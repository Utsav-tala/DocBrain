# DocBrain v2 — RAG Re-Architecture Plan

## Problem Statement

The current ingestion pipeline is a single `DirectoryLoader` that blindly ingests all 950 .mdx files from `src/oss/`. This produces a contaminated, unbalanced vector space with three critical failure modes:

1. **JavaScript Contamination** — 177 JS files pollute a Python-only tool
2. **Integration Page Dominance** — 576 integration pages drown out 33 core guide files
3. **Semantic Identity Loss** — chunks are stripped of their document context after splitting

The result: MMR fetches 50 candidates but 45 of them are from integration pages, so the "diverse" top-4 returned are still all integration noise.

---

## Data Audit Summary (Ground Truth)

### Layer 1 — `langchain_conceptual_docs/src/oss/` (950 files total)

| Folder | Files | Avg Size | Content Value | Action |
|---|---|---|---|---|
| `langchain/` (excl. errors/, frontend/) | ~20 | 30-82KB | ⭐ Highest — core Python LangChain guides | **INGEST as `core_guide`** |
| `langchain/errors/` | 7 | ~2KB | ⭐ High — specific error pages | **INGEST as `error_reference`** |
| `langgraph/` (excl. changelog-js) | ~33 | 10-40KB | ⭐ High — LangGraph guides | **INGEST as `langgraph_guide`** |
| `concepts/` | 4 | 12-21KB | ⭐ High — context, memory, models | **INGEST as `core_concepts`** |
| `python/migrate/` | 2 | 32KB | ⭐ High — LLMChain migration, v1 changes | **INGEST as `migration_guide`** |
| `python/integrations/` | 576 | 3-11KB | 🟡 Useful only for integration-specific queries | **INGEST as `integration`** |
| `deepagents/` | 30 | 5-20KB | 🟡 Niche advanced content | **INGEST as `deepagents`** |
| `javascript/` | 177 | 3-11KB | 🔴 Wrong language | **EXCLUDE** |
| `langchain/frontend/` | 12 | 5-15KB | 🔴 UI/frontend only, not for Python devs | **EXCLUDE** |
| `contributing/` | 8 | varies | 🔴 Contributor docs, not user-facing | **EXCLUDE** |
| `reference/` | 9 | tiny stubs | 🔴 Redirect stubs, no content | **EXCLUDE** |
| `langgraph/changelog-js.mdx` | 1 | small | 🔴 JS changelog | **EXCLUDE** |

### Layer 2 — `langchain_codebase/libs/core/langchain_core/` (180 .py files)

All kept. Well-structured. Key modules: `runnables/`, `prompts/`, `output_parsers/`, `retrievers.py`, `agents.py`. Tag with module-level metadata.

### Layer 3 — `langchain_issues.json` (500 issues)

Keep. Add `issue_title` to metadata. Expand to 2000+ in v2.1.

---

## Architecture: Categorical Loader System

### Core Principle: Configuration-Driven Ingestion

Replace the single `DirectoryLoader` with a `CATEGORY_CONFIG` dictionary that defines ingestion rules per category. Each category has its own:
- Source path + glob pattern + exclusion patterns
- `doc_type` tag
- Chunk size + overlap (tuned for that content type)
- Additional metadata fields to extract

### The `CATEGORY_CONFIG` Dictionary

```python
CONCEPTUAL_BASE = "data/langchain_conceptual_docs/src/oss"
CODEBASE_BASE   = "data/langchain_codebase/libs/core/langchain_core"

CATEGORY_CONFIG = {

    # ── TIER 1: Highest Priority — Core Conceptual Knowledge ──────────────
    "core_guide": {
        "path"         : f"{CONCEPTUAL_BASE}/langchain",
        "glob"         : "**/*.mdx",
        "exclude_paths": ["frontend/", "errors/"],     # handled separately
        "doc_type"     : "core_guide",
        "framework"    : "langchain",
        "language"     : "python",
        "chunk_size"   : 1500,
        "chunk_overlap": 200,
        "priority"     : 1,
        # WHY large chunks: files are 30-82KB rich guides with interconnected concepts.
        # Small chunks destroy context (e.g., splitting a RAG pipeline explanation in half).
    },

    "core_concepts": {
        "path"         : f"{CONCEPTUAL_BASE}/concepts",
        "glob"         : "*.mdx",
        "doc_type"     : "core_concepts",
        "framework"    : "langchain",
        "language"     : "agnostic",
        "chunk_size"   : 1500,
        "chunk_overlap": 200,
        "priority"     : 1,
    },

    "error_reference": {
        "path"         : f"{CONCEPTUAL_BASE}/langchain/errors",
        "glob"         : "*.mdx",
        "doc_type"     : "error_reference",
        "framework"    : "langchain",
        "language"     : "python",
        "chunk_size"   : 600,
        "chunk_overlap": 80,
        "priority"     : 1,
        # WHY small chunks: error files are 1-3KB each. One chunk per file is ideal.
        # We want the full error explanation together, not split.
    },

    "langgraph_guide": {
        "path"         : f"{CONCEPTUAL_BASE}/langgraph",
        "glob"         : "**/*.mdx",
        "exclude_paths": ["changelog-js.mdx"],
        "doc_type"     : "langgraph_guide",
        "framework"    : "langgraph",
        "language"     : "python",
        "chunk_size"   : 1500,
        "chunk_overlap": 200,
        "priority"     : 1,
    },

    "migration_guide": {
        "path"         : f"{CONCEPTUAL_BASE}/python/migrate",
        "glob"         : "*.mdx",
        "doc_type"     : "migration_guide",
        "framework"    : "langchain",
        "language"     : "python",
        "chunk_size"   : 2000,
        "chunk_overlap": 300,
        "priority"     : 1,
        # WHY largest chunks: migration guides have long before/after code comparisons.
        # Cutting them produces orphaned "before" or "after" halves with no context.
    },

    # ── TIER 2: Medium Priority — Integration-Specific Queries ────────────
    "integration": {
        "path"         : f"{CONCEPTUAL_BASE}/python/integrations",
        "glob"         : "**/*.mdx",
        "doc_type"     : "integration",
        "framework"    : "langchain",
        "language"     : "python",
        "chunk_size"   : 1000,
        "chunk_overlap": 150,
        "priority"     : 2,
        # WHY medium chunks: integration files are 3-11KB how-to guides.
        # 1000 chars captures a full setup section without being too broad.
    },

    "deepagents": {
        "path"         : f"{CONCEPTUAL_BASE}/deepagents",
        "glob"         : "**/*.mdx",
        "exclude_paths": ["frontend/"],
        "doc_type"     : "deepagents",
        "framework"    : "langgraph",
        "language"     : "python",
        "chunk_size"   : 1200,
        "chunk_overlap": 150,
        "priority"     : 2,
    },

    # ── TIER 2: Source Code ────────────────────────────────────────────────
    "source_code": {
        "path"         : CODEBASE_BASE,
        "glob"         : "**/*.py",
        "doc_type"     : "source_code",
        "framework"    : "langchain_core",
        "language"     : "python",
        "chunk_size"   : 1500,
        "chunk_overlap": 200,
        "priority"     : 2,
        # Use RecursiveCharacterTextSplitter.from_language(Language.PYTHON)
    },

    # ── TIER 3: Troubleshooting ────────────────────────────────────────────
    "troubleshooting": {
        "source"       : "github_issues_api",
        "doc_type"     : "troubleshooting",
        "framework"    : "langchain",
        "language"     : "python",
        "chunk_size"   : 800,
        "chunk_overlap": 100,
        "priority"     : 3,
    },
}

# ── HARD EXCLUDE LIST ──────────────────────────────────────────────────────
EXCLUDE_PATHS = [
    "src/oss/javascript/",        # Wrong language — JS
    "src/oss/langchain/frontend/",# UI/frontend, not for Python devs
    "src/oss/contributing/",      # Contributor docs
    "src/oss/reference/",         # Empty redirect stubs
    "src/oss/langgraph/studio.mdx",# Studio is a paid product
    "src/oss/langchain/studio.mdx",
]
```

---

## Metadata Schema (Per Chunk)

Every chunk in ChromaDB must carry these fields:

```python
{
    # ── Existing (keep) ────────────────────────────────────────────
    "source"         : str,   # local file path OR GitHub URL
    "doc_type"       : str,   # from CATEGORY_CONFIG (e.g., "core_guide")

    # ── NEW: Structural Metadata ───────────────────────────────────
    "framework"      : str,   # "langchain" | "langgraph" | "langchain_core"
    "language"       : str,   # "python" | "agnostic"
    "priority"       : int,   # 1=highest, 2=medium, 3=lower
    "subfolder"      : str,   # direct parent folder name (e.g., "langchain", "integrations")
    "file_name"      : str,   # bare filename without extension (e.g., "agents")
    "topic"          : str,   # same as file_name for docs; module name for .py files

    # ── NEW: Content Metadata (extracted from file) ────────────────
    "title"          : str,   # H1 heading or frontmatter title field
    "category_label" : str,   # human-readable (e.g., "Core Guide", "Integration")

    # ── Layer 3 only ───────────────────────────────────────────────
    "issue_title"    : str,   # GitHub issue title (enables lexical match on error name)
    "issue_number"   : str,   # GitHub issue number
}
```

### How to Extract Each Field

```python
def extract_metadata(source_path: str, category: str, config: dict) -> dict:
    meta = {
        "doc_type"     : config["doc_type"],
        "framework"    : config["framework"],
        "language"     : config["language"],
        "priority"     : config["priority"],
        "subfolder"    : Path(source_path).parent.name,
        "file_name"    : Path(source_path).stem,
        "topic"        : Path(source_path).stem.replace("-", "_"),
        "category_label": category.replace("_", " ").title(),
    }

    # Extract title from frontmatter or first H1
    meta["title"] = extract_title(source_path)

    return meta


def extract_title(file_path: str) -> str:
    """Read first line that matches 'title:' in frontmatter or '# ' H1."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("title:"):
                    return line.split("title:", 1)[1].strip().strip('"')
                if line.startswith("# "):
                    return line[2:].strip()
    except Exception:
        pass
    return Path(file_path).stem
```

---

## Contextual Chunking — The Key Fix

After splitting but **before embedding**, prepend a context header to every chunk's `page_content`:

```python
def add_context_header(chunk, metadata: dict) -> str:
    """
    Prepend topic identity to chunk so the embedding encodes WHAT this chunk is about.

    Before: "A retriever is an interface that returns documents..."
            → embedding is similar to ALL retriever-related text

    After:  "[LangChain Core Guide | Topic: retrieval | File: retrieval.mdx]
             A retriever is an interface that returns documents..."
            → embedding encodes "this is an OFFICIAL CORE GUIDE about retrievers"
    """
    header = (
        f"[{metadata['category_label']} | "
        f"Framework: {metadata['framework']} | "
        f"Topic: {metadata['topic']} | "
        f"File: {metadata['file_name']}]\n"
    )
    return header + chunk.page_content
```

**Result:** A chunk from `python/integrations/retrievers/dappier.mdx` gets embedded as:
```
[Integration | Framework: langchain | Topic: dappier | File: dappier]
Dappier makes it easy to connect...
```

A chunk from `langchain/retrieval.mdx` gets embedded as:
```
[Core Guide | Framework: langchain | Topic: retrieval | File: retrieval]
A retriever is an interface that returns documents given an unstructured query...
```

These two embeddings are now **semantically distinct** even though both contain the word "retriever".

---

## Updated Retriever Architecture

### Query → Intent Classification (Improved)

```python
INTENT_RULES = {
    "error_query": {
        "keywords": ["error", "exception", "traceback", "fix", "bug", "why am i getting",
                     "not working", "fails", "crash", "attributeerror", "valueerror",
                     "typeerror", "importerror", "validationerror"],
        "preferred_doc_types": ["error_reference", "troubleshooting"],
        "fallback_doc_types" : ["core_guide", "source_code"],
    },
    "concept_query": {
        "keywords": ["what is", "explain", "what are", "overview", "difference between",
                     "when to use", "why use", "define", "understand"],
        "preferred_doc_types": ["core_guide", "core_concepts", "langgraph_guide"],
        "fallback_doc_types" : ["migration_guide", "integration"],
    },
    "code_query": {
        "keywords": ["how to", "implement", "syntax", "parameter", "args", "class",
                     "method", "function", "example", "code", "use"],
        "preferred_doc_types": ["core_guide", "source_code", "integration"],
        "fallback_doc_types" : ["langgraph_guide"],
    },
    "migration_query": {
        "keywords": ["migration", "upgrade", "v1", "deprecated", "llmchain",
                     "legacy", "changed", "breaking"],
        "preferred_doc_types": ["migration_guide"],
        "fallback_doc_types" : ["core_guide"],
    },
    "integration_query": {
        "keywords": ["chroma", "pinecone", "openai", "anthropic", "faiss", "weaviate",
                     "install", "pip install", "setup"],
        "preferred_doc_types": ["integration", "core_guide"],
        "fallback_doc_types" : ["source_code"],
    },
}
```

### Two-Stage Retrieval with Priority Re-ranking

```
Stage 1: Classify query intent → get preferred doc_types
Stage 2: MMR search (fetch_k=60) across ALL chunks
Stage 3: Re-rank — among the 60 candidates:
           a. Score each by: embedding similarity + priority boost
           b. priority=1 chunks get +0.1 score boost
           c. chunks matching preferred doc_type get +0.15 boost
           d. integration chunks get -0.1 penalty on concept queries
Stage 4: Return top-k=5 after re-ranking (increased from 4)
Stage 5: If top-5 has fewer than 2 preferred doc_type chunks → run targeted search
          with metadata filter on preferred doc_types and merge results
```

### Priority Boosting Logic

```python
def rerank_results(docs, preferred_doc_types: list, query_intent: str) -> list:
    scored = []
    for doc in docs:
        score = doc.metadata.get("_similarity_score", 0.5)

        # Boost by priority tier
        priority = doc.metadata.get("priority", 3)
        if priority == 1:   score += 0.15
        elif priority == 2: score += 0.05

        # Boost if doc_type matches query intent
        if doc.metadata.get("doc_type") in preferred_doc_types:
            score += 0.15

        # Penalize integrations on concept queries
        if query_intent == "concept_query" and doc.metadata.get("doc_type") == "integration":
            score -= 0.10

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:5]]
```

---

## Expected Chunk Volume After Re-ingestion

| Category | Est. Files | Est. Chunks | Priority |
|---|---|---|---|
| `core_guide` | ~20 | ~1,500 | 1 |
| `core_concepts` | 4 | ~300 | 1 |
| `error_reference` | 7 | ~50 | 1 |
| `langgraph_guide` | ~33 | ~2,500 | 1 |
| `migration_guide` | 2 | ~200 | 1 |
| `integration` | 576 | ~8,000 | 2 |
| `deepagents` | ~22 | ~1,500 | 2 |
| `source_code` | 180 | ~2,448 | 2 |
| `troubleshooting` | 500 issues | ~5,349 | 3 |
| **TOTAL** | | **~21,847** | |

> **Note:** Total drops from 29,182 → ~21,847 (25% reduction) by removing JS and junk files. Signal-to-noise ratio improves dramatically.

---

## Prompt Template Improvements

### Fix 1 — Partial Context Handling

Current prompt jumps straight to "I could not find information" if context is weak. Add a middle ground:

```
If the context contains PARTIAL information:
- Answer what you can from the context
- Clearly state: "Note: This answer may be incomplete. The full documentation may have more details."
- Still include the Source URL
```

### Fix 2 — Metadata-Aware Context Formatting

Update `format_context()` in chain.py to expose the richer metadata:

```
--- Chunk 1 ---
Type     : Core Guide
Framework: LangChain
Topic    : retrieval
Source   : https://python.langchain.com/docs/langchain/retrieval/
Content  :
{chunk text}
```

This helps the LLM understand which chunks are authoritative vs supplementary.

### Fix 3 — Add doc_type Priority Instruction to Prompt

```
When multiple chunks are provided:
- Treat "Core Guide" and "Core Concepts" chunks as primary authoritative sources
- Treat "Integration" chunks as supplementary examples
- Treat "Source Code" chunks as API-level technical detail
- Prefer information from higher-priority chunks when there is any contradiction
```

---

## Evaluation Plan

### Test Set — 20 Queries (to run before and after)

| # | Query | Target doc_type | Success Metric |
|---|---|---|---|
| 1 | What is a retriever in LangChain? | `core_guide` | Top chunk from `retrieval.mdx` |
| 2 | What is LCEL? | `core_guide` | Top chunk from `langchain/` guides |
| 3 | What are document loaders? | `core_guide` | From core guide, NOT integration |
| 4 | What is a vector store? | `core_guide` | From core guide, NOT integration |
| 5 | What is LangGraph and when to use it? | `langgraph_guide` | From `langgraph/overview.mdx` |
| 6 | How to use RecursiveCharacterTextSplitter? | `core_guide` + `integration` | Code example returned |
| 7 | How to implement streaming in LangChain? | `core_guide` | From `streaming.mdx` |
| 8 | What are output parsers? | `core_guide` + `source_code` | API + guide combined |
| 9 | How to use Chroma as vector store? | `integration` | From `integrations/vectorstores/chroma.mdx` |
| 10 | How to use OpenAI embeddings? | `integration` | From `integrations/embeddings/` |
| 11 | Why am I getting OUTPUT_PARSING_FAILURE? | `error_reference` | From `errors/OUTPUT_PARSING_FAILURE.mdx` |
| 12 | Why am I getting MODEL_RATE_LIMIT error? | `error_reference` | From `errors/MODEL_RATE_LIMIT.mdx` |
| 13 | Why am I getting MODEL_NOT_FOUND? | `error_reference` | From `errors/MODEL_NOT_FOUND.mdx` |
| 14 | What is the difference between LLMChain and LCEL? | `migration_guide` | From `python/migrate/langchain-v1.mdx` |
| 15 | How to migrate from LangChain v0 to v1? | `migration_guide` | From migration guide |
| 16 | What is structured output in LangChain? | `core_guide` | From `structured-output.mdx` |
| 17 | What is context engineering? | `core_guide` + `concepts` | From `context-engineering.mdx` or `concepts/context.mdx` |
| 18 | How to add memory to a LangGraph agent? | `langgraph_guide` | From `langgraph/add-memory.mdx` |
| 19 | What is tool calling in LangChain? | `core_guide` | From `tools.mdx` |
| 20 | What are checkpointers in LangGraph? | `langgraph_guide` | From `langgraph/checkpointers.mdx` |

### Metrics to Track

| Metric | Definition | Target |
|---|---|---|
| **Hit Rate @5** | Is the correct file in top-5 chunks? | >85% |
| **Correct doc_type Rate** | Does query intent match returned doc_type? | >80% |
| **Integration Contamination Rate** | % of concept queries where top chunk is from `integration/` | <10% |
| **Answer Quality Score** | Human 1-5 rating on 7-part format completeness | >3.5 avg |

---

## Files to Modify

### [MODIFY] `src/ingest.py`
- Replace single `DirectoryLoader` with `CategoryLoader` using `CATEGORY_CONFIG`
- Add `extract_metadata()` and `extract_title()` helper functions
- Add `add_context_header()` contextual chunking function
- Add exclusion path filtering before loading
- Separate Python splitter for `source_code` category

### [MODIFY] `src/retriever.py`
- Replace simple `classify_query()` with intent-based `INTENT_RULES` system
- Add `rerank_results()` function for priority boosting
- Update `retrieve()` to use two-stage MMR + rerank
- Update logging to show new metadata fields

### [MODIFY] `src/chain.py`
- Update `format_context()` to display `framework`, `category_label`, `topic`, `priority`

### [MODIFY] `src/prompt_templates.py`
- Add partial-context handling instructions
- Add doc_type priority guidance for LLM
- Tighten code block enforcement

### [MODIFY] `PROJECT_MEMORY.md`
- Document all decisions from this plan

---

## Open Questions for User

> [!IMPORTANT]
> **Q1 — deepagents inclusion:** The `deepagents/` folder (30 files) is about an advanced autonomous agent product. Should it be included? It's real content but niche. Excluding it keeps the vector space cleaner.

> [!IMPORTANT]
> **Q2 — Integration volume:** 576 integration files will still produce ~8,000 chunks (even with smaller chunk size). This is still the largest single category. Do you want to further filter integrations (e.g., only keep `vectorstores/`, `retrievers/`, `embeddings/`, `document_loaders/`) and exclude `providers/` (178 files of pure provider marketing pages)?

> [!IMPORTANT]
> **Q3 — Re-ingest cost:** Re-ingesting ~21,847 chunks at `text-embedding-3-small` rates = ~$0.45. Old ChromaDB must be deleted first (`db/chroma_langchain/`). Confirm before running.

> [!IMPORTANT]
> **Q4 — GitHub Issues:** Increase from 500 → 2000 (costs more API calls, no $ cost with token). Do you want to do this during this re-ingest run or keep at 500 for now?
