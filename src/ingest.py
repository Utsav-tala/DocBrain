"""
ingest.py — DocBrain v2
Categorical Loader System (replaces naive single DirectoryLoader from v1)

Key Changes from v1:
  - CATEGORY_CONFIG drives all ingestion rules per content type
  - Per-category chunking: chunk_size tuned per content type (600-2000 chars)
  - Rich metadata: 12 fields per chunk (was 2 in v1)
  - Contextual chunking: [Category | Framework | Topic | File] header
    prepended to every chunk before embedding for precise semantic identity
  - Hard exclusions: javascript/, frontend/, contributing/, providers/, tools/
  - Integration filtered to: vectorstores/, retrievers/, embeddings/, document_loaders/
  - Priority field (1/2/3) stored in metadata for retrieval re-ranking
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
CONCEPTUAL_BASE  = "data/langchain_conceptual_docs/src/oss"
CODEBASE_BASE    = "data/langchain_codebase/libs/core/langchain_core"
DB_PATH          = "db/chroma_langchain"
ISSUES_SAVE_PATH = "data/langchain_issues.json"
COLLECTION_NAME  = "langchain_docs"

# ── Category Configuration ────────────────────────────────────────────────────
CATEGORY_CONFIG = {

    # TIER 1: Core Knowledge — priority=1 gets score boost in retriever

    "core_guide": {
        "path"          : f"{CONCEPTUAL_BASE}/langchain",
        "glob"          : "**/*.mdx",
        "exclude_paths" : ["frontend", "errors"],
        "doc_type"      : "core_guide",
        "framework"     : "langchain",
        "language"      : "python",
        "priority"      : 1,
        "category_label": "Core Guide",
        "chunk_size"    : 1500,
        "chunk_overlap" : 200,
    },

    "core_concepts": {
        "path"          : f"{CONCEPTUAL_BASE}/concepts",
        "glob"          : "*.mdx",
        "doc_type"      : "core_concepts",
        "framework"     : "langchain",
        "language"      : "agnostic",
        "priority"      : 1,
        "category_label": "Core Concepts",
        "chunk_size"    : 1500,
        "chunk_overlap" : 200,
    },

    "error_reference": {
        "path"          : f"{CONCEPTUAL_BASE}/langchain/errors",
        "glob"          : "*.mdx",
        "doc_type"      : "error_reference",
        "framework"     : "langchain",
        "language"      : "python",
        "priority"      : 1,
        "category_label": "Error Reference",
        "chunk_size"    : 600,
        "chunk_overlap" : 80,
    },

    "langgraph_guide": {
        "path"          : f"{CONCEPTUAL_BASE}/langgraph",
        "glob"          : "**/*.mdx",
        "exclude_files" : ["changelog-js.mdx", "studio.mdx"],
        "doc_type"      : "langgraph_guide",
        "framework"     : "langgraph",
        "language"      : "python",
        "priority"      : 1,
        "category_label": "LangGraph Guide",
        "chunk_size"    : 1500,
        "chunk_overlap" : 200,
    },

    "migration_guide": {
        "path"          : f"{CONCEPTUAL_BASE}/python/migrate",
        "glob"          : "*.mdx",
        "doc_type"      : "migration_guide",
        "framework"     : "langchain",
        "language"      : "python",
        "priority"      : 1,
        "category_label": "Migration Guide",
        "chunk_size"    : 2000,
        "chunk_overlap" : 300,
    },

    # TIER 2: Supplementary Knowledge

    "deepagents": {
        "path"          : f"{CONCEPTUAL_BASE}/deepagents",
        "glob"          : "**/*.mdx",
        "exclude_paths" : ["frontend"],
        "doc_type"      : "deepagents",
        "framework"     : "langgraph",
        "language"      : "python",
        "priority"      : 2,
        "category_label": "Deep Agents",
        "chunk_size"    : 1200,
        "chunk_overlap" : 150,
    },

    "source_code": {
        "path"          : CODEBASE_BASE,
        "glob"          : "**/*.py",
        "doc_type"      : "source_code",
        "framework"     : "langchain_core",
        "language"      : "python",
        "priority"      : 2,
        "category_label": "Source Code",
        "chunk_size"    : 1500,
        "chunk_overlap" : 200,
        "use_py_splitter": True,
    },
}

# Integration: only RAG-relevant subcategories
# Excluded: providers/(178), tools/(91), chat/(58), llms/(21)
INTEGRATION_SUBCATEGORIES = [
    "vectorstores",
    "document_loaders",
    "retrievers",
    "embeddings",
]


# ── Metadata Helpers ──────────────────────────────────────────────────────────

def extract_title(file_path: str) -> str:
    """Extract title from MDX frontmatter 'title:' or first '# ' heading."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            in_frontmatter = False
            for i, line in enumerate(f):
                stripped = line.strip()
                if i == 0 and stripped == "---":
                    in_frontmatter = True
                    continue
                if in_frontmatter:
                    if stripped == "---":
                        in_frontmatter = False
                        continue
                    if stripped.startswith("title:"):
                        return stripped.split("title:", 1)[1].strip().strip('"').strip("'")
                if stripped.startswith("# "):
                    return stripped[2:].strip()
    except Exception:
        pass
    return Path(file_path).stem


def build_chunk_metadata(source_path: str, config: dict) -> dict:
    """Build the full metadata dict for a chunk."""
    p = Path(source_path)
    return {
        "doc_type"      : config["doc_type"],
        "framework"     : config["framework"],
        "language"      : config["language"],
        "priority"      : config["priority"],
        "category_label": config["category_label"],
        "subfolder"     : p.parent.name,
        "file_name"     : p.stem,
        "topic"         : p.stem.replace("-", "_").replace(" ", "_").lower(),
        "title"         : extract_title(source_path),
    }


def build_context_header(metadata: dict) -> str:
    """
    Build the context header prepended to each chunk before embedding.
    This encodes document identity into the embedding vector itself.

    Example output:
      [Core Guide | Framework: langchain | Topic: retrieval | File: retrieval]
    """
    return (
        f"[{metadata['category_label']} | "
        f"Framework: {metadata['framework']} | "
        f"Topic: {metadata['topic']} | "
        f"File: {metadata['file_name']}]\n"
    )


# ── Exclusion Filter ──────────────────────────────────────────────────────────

def filter_excluded_docs(docs: list, config: dict) -> list:
    """Remove docs whose source path contains any excluded subfolder or filename."""
    exclude_paths = set(config.get("exclude_paths", []))
    exclude_files = set(config.get("exclude_files", []))

    if not exclude_paths and not exclude_files:
        return docs

    filtered = []
    for doc in docs:
        source   = doc.metadata.get("source", "")
        path_obj = Path(source)

        if exclude_paths and exclude_paths.intersection(set(path_obj.parts)):
            continue
        if exclude_files and path_obj.name in exclude_files:
            continue

        filtered.append(doc)

    removed = len(docs) - len(filtered)
    if removed > 0:
        logger.info(f"   Filtered out {removed} excluded docs")

    return filtered


# ── Category Loader ───────────────────────────────────────────────────────────

def load_category(category_name: str, config: dict) -> list:
    """
    Load, split, and enrich one category of documents.
    Steps: load -> filter -> split -> enrich metadata -> add context header
    """
    path = config["path"]

    if not Path(path).exists():
        logger.warning(f"   Path does not exist, skipping: {path}")
        return []

    logger.info(f"\n{'='*55}")
    logger.info(f"  [{category_name.upper()}]")
    logger.info(f"  doc_type: {config['doc_type']} | priority: {config['priority']}")
    logger.info(f"  path: {path}")

    loader = DirectoryLoader(
        path,
        glob=config.get("glob", "**/*.mdx"),
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        silent_errors=True,
    )
    docs = loader.load()
    logger.info(f"  Loaded: {len(docs)} files")

    docs = filter_excluded_docs(docs, config)
    logger.info(f"  After exclusion: {len(docs)} files")

    if not docs:
        return []

    if config.get("use_py_splitter"):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

    chunks = splitter.split_documents(docs)
    logger.info(f"  Chunks: {len(chunks)} (size={config['chunk_size']}, overlap={config['chunk_overlap']})")

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        meta   = build_chunk_metadata(source, config)
        chunk.metadata.update(meta)
        chunk.page_content = build_context_header(chunk.metadata) + chunk.page_content

    logger.info(f"  Metadata + context headers applied")
    return chunks


# ── Integration Loader ────────────────────────────────────────────────────────

def load_integrations() -> list:
    """
    Load only selected integration subcategories.
    Excluded: providers/, tools/, chat/, llms/ (and all others)
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"  [INTEGRATION] (selective subcategories only)")
    logger.info(f"  Subcategories: {INTEGRATION_SUBCATEGORIES}")

    cfg = {
        "doc_type"      : "integration",
        "framework"     : "langchain",
        "language"      : "python",
        "priority"      : 2,
        "category_label": "Integration",
        "chunk_size"    : 1000,
        "chunk_overlap" : 150,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )

    all_chunks = []
    base = f"{CONCEPTUAL_BASE}/python/integrations"

    for sub in INTEGRATION_SUBCATEGORIES:
        subpath = f"{base}/{sub}"
        if not Path(subpath).exists():
            logger.warning(f"  Subcategory missing: {subpath}")
            continue

        loader = DirectoryLoader(
            subpath,
            glob="**/*.mdx",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            silent_errors=True,
        )
        docs   = loader.load()
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            meta   = build_chunk_metadata(source, cfg)
            meta["subfolder"] = sub
            chunk.metadata.update(meta)
            chunk.page_content = build_context_header(chunk.metadata) + chunk.page_content

        logger.info(f"  [{sub}] {len(docs)} files → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"  Integration total: {len(all_chunks)} chunks")
    return all_chunks


# ── GitHub Issues Loader ──────────────────────────────────────────────────────

def load_github_issues(max_pages: int = 5) -> list:
    """
    Fetch closed bug issues from langchain-ai/langchain GitHub.
    Adds issue_title metadata for lexical matching on specific error names.
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"  [TROUBLESHOOTING] GitHub Issues (max_pages={max_pages})")

    all_issues = []
    headers    = {"Accept": "application/vnd.github.v3+json"}

    github_token = os.getenv("GITHUB_TOKEN", "")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
        logger.info("  Using GitHub token (rate limit: 5000/hr)")
    else:
        logger.warning("  No GitHub token — rate limit: 60/hr")

    for page in range(1, max_pages + 1):
        url = (
            "https://api.github.com/repos/langchain-ai/langchain/issues"
            f"?state=closed&per_page=100&page={page}&labels=bug"
        )
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code != 200:
            logger.warning(f"  GitHub API returned {resp.status_code} on page {page}. Stopping.")
            break

        page_issues = resp.json()
        if not page_issues:
            break

        all_issues.extend(page_issues)
        logger.info(f"  Page {page}: {len(page_issues)} issues")

    logger.info(f"  Total issues fetched: {len(all_issues)}")

    with open(ISSUES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2)

    documents = []
    for issue in all_issues:
        title  = issue.get("title", "")
        body   = issue.get("body") or ""
        url    = issue.get("html_url", "")
        number = issue.get("number", "")

        if len(body.strip()) < 50:
            continue

        content = f"Issue #{number}: {title}\n\n{body}"
        documents.append(Document(
            page_content=content,
            metadata={
                "source"        : url,
                "doc_type"      : "troubleshooting",
                "framework"     : "langchain",
                "language"      : "python",
                "priority"      : 3,
                "category_label": "Troubleshooting",
                "subfolder"     : "github_issues",
                "file_name"     : f"issue_{number}",
                "topic"         : "bug_report",
                "title"         : title,
                "issue_title"   : title,
                "issue_number"  : str(number),
            }
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_documents(documents)

    for chunk in chunks:
        issue_title = chunk.metadata.get("issue_title", "bug report")
        header = f"[Troubleshooting | Framework: langchain | Issue: {issue_title}]\n"
        chunk.page_content = header + chunk.page_content

    logger.info(f"  {len(chunks)} troubleshooting chunks ready")
    return chunks


# ── Embed & Store ─────────────────────────────────────────────────────────────

def embed_and_store(all_chunks: list):
    """Embed all chunks and persist to ChromaDB in batches of 500."""
    logger.info(f"\n{'='*55}")
    logger.info(f"  EMBEDDING {len(all_chunks)} chunks into ChromaDB")
    logger.info(f"  Model: text-embedding-3-small")
    logger.info(f"  Dest : {DB_PATH} | Collection: {COLLECTION_NAME}")

    embedding   = OpenAIEmbeddings(model="text-embedding-3-small")
    BATCH_SIZE  = 500
    vectorstore = None

    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = all_chunks[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    logger.info(f"  ChromaDB ready at '{DB_PATH}'")
    return vectorstore


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    total = sum(results.values())
    print("\n" + "=" * 60)
    print("       DOCBRAIN v2 — INGESTION SUMMARY")
    print("=" * 60)
    for cat, count in results.items():
        label = f"  {cat}"
        print(f"{label:<32} {count:>6} chunks")
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<32} {total:>6} chunks")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("DocBrain v2 Ingestion — Categorical Loader System")

    all_chunks = []
    results    = {}

    # Load all standard categories
    for name, cfg in CATEGORY_CONFIG.items():
        chunks = load_category(name, cfg)
        all_chunks.extend(chunks)
        results[name] = len(chunks)

    # Load selective integrations
    int_chunks = load_integrations()
    all_chunks.extend(int_chunks)
    results["integration"] = len(int_chunks)

    # Load GitHub Issues
    issue_chunks = load_github_issues(max_pages=5)
    all_chunks.extend(issue_chunks)
    results["troubleshooting"] = len(issue_chunks)

    print_summary(results)
    embed_and_store(all_chunks)

    logger.info(f"DocBrain v2 ingestion complete. Total chunks: {len(all_chunks)}")
