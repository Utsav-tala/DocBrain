"""
retriever.py — DocBrain v3
Fixes applied:
  B-08: MMR k != fetch_k — k=5, fetch_k=60 so diversity algorithm has a real pool
  B-09: Multi-intent scoring — all rules scored, top intent wins (no early-exit)
  B-10: classify_intent() returns intent string alongside rule so chain.py can
        route to the correct prompt template
  B-11: strip_context_header() removes the embedded [Category|...] prefix from
        page_content before sending to LLM (it was needed for embedding, not LLM)
"""

import os
import re
import requests
from dotenv import load_dotenv
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── URL verification cache ────────────────────────────────────────────────────
# convert_source_to_url() GUESSES a live URL from the local file path. That guess
# is frequently wrong (doc reorganizations, renamed paths) and was producing
# confidently-wrong source links. We now verify with a cheap HEAD request and
# cache the result so repeated chunks from the same file don't re-hit the network.
_URL_VERIFY_CACHE: dict[str, bool] = {}
_VERIFY_TIMEOUT = 3  # seconds — fail fast, don't block the response on a slow HEAD


def _url_resolves(url: str) -> bool:
    if url in _URL_VERIFY_CACHE:
        return _URL_VERIFY_CACHE[url]
    try:
        resp = requests.head(url, timeout=_VERIFY_TIMEOUT, allow_redirects=True)
        ok = resp.status_code < 400
    except Exception:
        ok = False
    _URL_VERIFY_CACHE[url] = ok
    return ok

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH         = "db/chroma_langchain"
COLLECTION_NAME = "langchain_docs"
K               = 5      # final chunks returned to LLM
FETCH_K         = 60     # MMR candidate pool — MUST be > K for diversity to work
MIN_PREFERRED   = 2      # min preferred-type chunks before fallback triggers

# ── Intent Classification Rules ───────────────────────────────────────────────
# Each rule has a weight per keyword match — higher weight = stronger signal.
# All rules are scored. Top-scoring rule wins. No early-exit.

INTENT_RULES = [
    {
        "intent"   : "error_query",
        "keywords" : {
            "error": 2, "exception": 2, "traceback": 2, "fix": 1, "bug": 1,
            "why am i getting": 3, "not working": 1, "fails": 1, "crash": 2,
            "attributeerror": 3, "valueerror": 3, "typeerror": 3,
            "importerror": 3, "validationerror": 3,
            "output_parsing_failure": 3, "model_not_found": 3,
            "model_rate_limit": 3, "model_authentication": 3,
        },
        "preferred": ["error_reference", "troubleshooting"],
        "fallback" : ["core_guide", "source_code"],
        "penalty"  : [],
    },
    {
        "intent"   : "migration_query",
        "keywords" : {
            "migration": 2, "migrate": 2, "upgrade": 2, "v1": 1, "v0": 1,
            "deprecated": 2, "llmchain": 3, "legacy": 2,
            "breaking change": 3, "difference between llmchain": 3, "old api": 2,
        },
        "preferred": ["migration_guide"],
        "fallback" : ["core_guide"],
        "penalty"  : ["integration", "troubleshooting"],
    },
    {
        "intent"   : "concept_query",
        "keywords" : {
            "what is": 2, "what are": 2, "explain": 2, "overview": 1,
            "difference between": 2, "when to use": 2, "why use": 1,
            "define": 2, "understand": 1, "concept": 1,
            "compare": 2, "vs": 1, "versus": 2,
        },
        "preferred": ["core_guide", "core_concepts", "langgraph_guide"],
        "fallback" : ["migration_guide", "deepagents"],
        "penalty"  : ["integration"],
    },
    {
        "intent"   : "code_query",
        "keywords" : {
            "how to": 2, "implement": 2, "syntax": 2, "parameter": 1,
            "args": 1, "class": 1, "method": 1, "function": 1,
            "example": 1, "code": 1, "import": 1, "install": 1,
            "show me": 2, "give me": 1, "write": 1, "create": 1, "build": 1,
        },
        "preferred": ["core_guide", "source_code", "integration"],
        "fallback" : ["langgraph_guide"],
        "penalty"  : [],
    },
    {
        "intent"   : "langgraph_query",
        "keywords" : {
            "langgraph": 3, "state graph": 3, "stategraph": 3,
            "checkpointer": 2, "state machine": 2, "workflow": 1,
            "node": 1, "edge": 1, "graph": 1,
        },
        "preferred": ["langgraph_guide", "core_guide"],
        "fallback" : ["deepagents"],
        "penalty"  : ["integration"],
    },
    {
        "intent"   : "integration_query",
        "keywords" : {
            "chroma": 3, "pinecone": 3, "weaviate": 3, "faiss": 3,
            "openai embedding": 2, "anthropic": 2, "cohere": 2,
            "huggingface": 2, "pip install": 2, "setup": 1,
            "configuration": 1, "api key": 2,
        },
        "preferred": ["integration", "core_guide"],
        "fallback" : ["source_code"],
        "penalty"  : [],
    },
]

DEFAULT_RULE = {
    "intent"   : "general",
    "preferred": ["core_guide", "core_concepts", "langgraph_guide"],
    "fallback" : ["source_code", "integration"],
    "penalty"  : [],
}


# ── Load VectorStore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
    )


# ── B-09 Fix: Multi-intent Scoring ───────────────────────────────────────────
def classify_intent(query: str) -> dict:
    """
    Score ALL rules against the query. Top-scoring rule wins.
    Returns the full rule dict including intent string for downstream routing.

    Old behaviour: first-match-wins (early exit on first keyword hit).
    New behaviour: accumulate weighted keyword scores across all rules,
                   return the rule with the highest total score.
    """
    query_lower = query.lower()
    best_rule   = DEFAULT_RULE
    best_score  = 0

    for rule in INTENT_RULES:
        score = 0
        for keyword, weight in rule["keywords"].items():
            if keyword in query_lower:
                score += weight
        if score > best_score:
            best_score = score
            best_rule  = rule

    logger.info(f"   Intent: {best_rule['intent']} (score={best_score})")
    logger.info(f"   Preferred: {best_rule['preferred']}")
    return best_rule


# ── B-11 Fix: Strip Embedded Context Header from page_content ────────────────
def strip_context_header(text: str) -> str:
    """
    Remove the [Category | Framework: X | Topic: Y | File: Z] header
    that was prepended during ingestion for embedding purposes.
    We expose this metadata separately in format_context() — no need to
    repeat it inside the content body sent to the LLM.
    """
    return re.sub(r'^\[.*?\]\n', '', text, count=1)


# ── Priority Re-ranking ───────────────────────────────────────────────────────
def rerank_results(docs: list, scores: list, rule: dict) -> list:
    """
    Re-rank MMR results using priority metadata + intent alignment.
    Now also attaches the rerank_score to each doc's metadata so
    format_context() can expose it to the LLM.

    Score per chunk:
      Base          : 1.0
      priority=1    : +0.3
      priority=2    : +0.1
      preferred type: +0.2
      penalty type  : -0.2
    """
    preferred = set(rule.get("preferred", []))
    penalty   = set(rule.get("penalty", []))

    # scores list may be shorter than docs if MMR doesn't return scores
    score_map = {}
    for i, doc in enumerate(docs):
        base     = 1.0
        doc_type = doc.metadata.get("doc_type", "")
        priority = int(doc.metadata.get("priority", 3))

        if priority == 1:
            base += 0.3
        elif priority == 2:
            base += 0.1

        if doc_type in preferred:
            base += 0.2
        if doc_type in penalty:
            base -= 0.2

        score_map[i] = base

    # Sort by rerank score
    ranked_indices = sorted(score_map, key=lambda i: score_map[i], reverse=True)

    result = []
    for rank, idx in enumerate(ranked_indices[:K]):
        doc = docs[idx]
        doc.metadata["rerank_score"] = round(score_map[idx], 2)
        doc.metadata["rerank_rank"]  = rank + 1
        # B-11: strip context header from page_content
        doc.page_content = strip_context_header(doc.page_content)

        logger.info(
            f"   [{rank+1}] rerank={score_map[idx]:.2f} | "
            f"type={doc.metadata.get('doc_type')} | "
            f"priority={doc.metadata.get('priority')} | "
            f"topic={doc.metadata.get('topic', '')} | "
            f"file={doc.metadata.get('file_name', '')}"
        )
        result.append(doc)

    return result


# ── Fallback Search ───────────────────────────────────────────────────────────
def targeted_fallback(query: str, vectorstore, preferred_types: list) -> list:
    """
    Targeted MMR search filtered to preferred doc_types.
    k=3, fetch_k=20 — small, fast, corrective.
    """
    results = []
    for doc_type in preferred_types[:2]:
        try:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k"      : 3,
                    "fetch_k": 20,           # fetch_k > k — diversity works
                    "filter" : {"doc_type": doc_type},
                }
            )
            found = retriever.invoke(query)
            results.extend(found)
            if results:
                logger.info(f"   Fallback: {len(found)} chunks from doc_type={doc_type}")
                break
        except Exception as e:
            logger.warning(f"   Fallback failed for doc_type={doc_type}: {e}")

    return results


# ── Main Retrieve Function ────────────────────────────────────────────────────
def retrieve(query: str, vectorstore=None) -> tuple[list, dict]:
    """
    Full retrieval pipeline. Returns (docs, rule) tuple.

    B-08 fix: k=K (5), fetch_k=FETCH_K (60) — MMR pool is 12× the return size.
    B-09 fix: multi-intent scoring via classify_intent().
    B-10 fix: returns rule dict so chain.py can select the right prompt template.
    B-11 fix: rerank_results() strips context headers from page_content.

    Returns:
        docs : list[Document] — top-K chunks with rerank_score in metadata
        rule : dict           — intent rule for prompt routing
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    q_preview = query[:60] + "..." if len(query) > 60 else query
    logger.info(f"\n--- Retrieval: '{q_preview}' ---")

    # Stage 1: Multi-intent classification
    rule = classify_intent(query)

    # Stage 2: MMR search — B-08 fix: k=K, fetch_k=FETCH_K (not both=FETCH_K)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k"      : FETCH_K,   # get large pool for reranking
            "fetch_k": FETCH_K,   # ChromaDB MMR internal pool
        },
    )
    candidates = retriever.invoke(query)
    logger.info(f"   Candidates from MMR: {len(candidates)}")

    # Stage 3: Priority re-rank, strip headers
    results = rerank_results(candidates, [], rule)

    # Stage 4: Fallback if preferred types underrepresented
    preferred_types = set(rule.get("preferred", []))
    preferred_count = sum(1 for d in results if d.metadata.get("doc_type") in preferred_types)

    if preferred_count < MIN_PREFERRED and rule["intent"] != "general":
        logger.warning(f"   Only {preferred_count}/{K} preferred — triggering fallback")
        fallback_docs = targeted_fallback(query, vectorstore, rule["preferred"])
        if fallback_docs:
            # Strip headers from fallback docs too
            for doc in fallback_docs:
                doc.page_content = strip_context_header(doc.page_content)
            combined = candidates + fallback_docs
            results  = rerank_results(combined, [], rule)

    logger.info(f"   Final chunks: {len(results)} | intent: {rule['intent']}")
    return results, rule     # B-10: return rule for prompt routing


# ── Source URL Converter ──────────────────────────────────────────────────────
# Known-good top-level fallback pages, by framework. Used when a guessed URL
# doesn't resolve — better to point at a real, relevant index page than a
# confidently-wrong 404 or unrelated article.
_FALLBACK_PAGES = {
    "langchain"        : "https://python.langchain.com/docs/introduction/",
    "langgraph"        : "https://langchain-ai.github.io/langgraph/",
    "langchain_core"    : "https://python.langchain.com/api_reference/core/index.html",
}


def convert_source_to_url(source_path: str, verify: bool = True) -> str:
    """
    Convert a local file path to a live doc URL.

    IMPORTANT: this is a GUESS based on the local mirror's folder structure
    matching the live site's URL structure. That assumption breaks whenever
    docs get reorganized. When verify=True, we HEAD-check the guess and fall
    back to a known-good index page rather than return a link that 404s or
    silently resolves to the wrong/unrelated page.
    """
    if not source_path:
        return ""
    if source_path.startswith("https://"):
        return source_path

    guessed = None
    framework = "langchain"

    if "langchain_conceptual_docs" in source_path:
        try:
            relative = source_path.split("src/oss/")[-1]
            relative = relative.replace(".mdx", "").replace(".md", "")
            guessed = f"https://python.langchain.com/docs/{relative}/"
            if "langgraph" in relative:
                framework = "langgraph"
        except Exception:
            guessed = None

    elif "langchain_codebase" in source_path:
        try:
            relative = source_path.split("langchain_codebase/")[-1]
            guessed = f"https://github.com/langchain-ai/langchain/blob/master/{relative}"
            framework = "langchain_core"
        except Exception:
            guessed = None

    if not guessed:
        return source_path

    if not verify:
        return guessed

    if _url_resolves(guessed):
        return guessed

    logger.warning(f"   [url-verify] Guessed URL did not resolve, using fallback: {guessed}")
    return _FALLBACK_PAGES.get(framework, _FALLBACK_PAGES["langchain"])