"""
retriever.py — DocBrain v5

Pipeline:
  classify intent → scored candidate pool (FETCH_K) → re-rank → top K

Ranking model (v5):
  final = semantic_relevance  +  metadata_nudge

  Semantic relevance is the PRIMARY signal and dominates. Metadata (priority,
  intent-preferred doc_type) is a small nudge that only breaks ties between
  chunks that are already comparably relevant to the question.

Why this changed (v4 → v5):
  v4's rerank_results() scored chunks on metadata ALONE. It never looked at the
  query and discarded the vector store's similarity score entirely, so a
  weakly-relevant priority=1 chunk reliably outranked a highly-relevant
  priority=3 one. That is the root cause of the off-topic chunks (and hence the
  off-topic source links) — link_resolver.py was fixing the symptom downstream
  while the LLM was still being handed the wrong documents to read.

  v4 also asked MMR for k == fetch_k == 60, which left MMR's diversity step no
  headroom to select from — it just returned the whole pool. Diversity is now an
  explicit per-source-file cap (MAX_PER_FILE), which does the job MMR was
  supposed to do and is far easier to reason about and tune.

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
K               = 5      # final chunks returned to LLM (rerank truncates to this)
FETCH_K         = 60     # candidate pool pulled from the vector store pre-rerank
MIN_PREFERRED   = 2      # min preferred-type chunks before fallback triggers
FALLBACK_K      = 5      # chunks pulled per doc_type in the targeted fallback
MAX_PER_FILE    = 2      # diversity cap: max chunks one source file may contribute

# ── Re-rank weights ───────────────────────────────────────────────────────────
# Relevance is normalized to [0,1] within the candidate pool, so the metadata
# nudge below must stay well under 1.0 or it starts overriding relevance again —
# which is exactly the v4 bug. Max possible nudge here is +0.18 / -0.10, meaning
# metadata can only reorder chunks whose relevance is already within ~0.18 of
# each other. It cannot lift an off-topic chunk over an on-topic one.
#
# These are deliberately module-level knobs: Phase 3 ablations will sweep them.
PRIORITY_BOOST  = {1: 0.10, 2: 0.05}   # priority 3+ gets nothing
PREFERRED_BOOST = 0.08                 # doc_type is in the intent's preferred list
PENALTY_WEIGHT  = 0.10                 # doc_type is in the intent's penalty list

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

# Pre-compile each keyword as a WHOLE-WORD pattern.
#
# v4 used `if keyword in query_lower`, a bare substring test, so keywords fired on
# fragments of unrelated words: "graph" matched para-GRAPH, "class" matched
# CLASS-ification, "vs" matched inside "versus" (double-counting it alongside the
# "versus" keyword). A bad intent is expensive here — it picks the prompt template
# AND decides which doc_types get the preferred/penalty nudge in rerank_results().
for _rule in INTENT_RULES:
    _rule["_patterns"] = [
        (re.compile(rf"\b{re.escape(kw)}\b"), weight)
        for kw, weight in _rule["keywords"].items()
    ]


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

    Keywords match on whole-word boundaries (see _patterns above), not substrings.
    Scores accumulate across every rule — no early exit on first hit.
    """
    query_lower = query.lower()
    best_rule   = DEFAULT_RULE
    best_score  = 0

    for rule in INTENT_RULES:
        score = sum(w for pattern, w in rule["_patterns"] if pattern.search(query_lower))
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


# ── Re-ranking ────────────────────────────────────────────────────────────────
def _dedupe(scored_docs: list) -> list:
    """
    Drop duplicate chunks, keeping the best-scoring (lowest-distance) copy.

    The fallback path merges a second, filtered search into the main pool, and the
    same chunk can legitimately surface in both. Left unmerged it would occupy two
    slots in the final K and double-count against the MAX_PER_FILE cap.
    """
    best: dict = {}
    for doc, distance in scored_docs:
        key = (doc.metadata.get("source", ""), doc.page_content)
        if key not in best or distance < best[key][1]:
            best[key] = (doc, distance)
    return list(best.values())


def _normalize_relevance(distances: list) -> list:
    """
    Map raw vector-store distances onto a [0,1] relevance score (1.0 = closest).

    We min-max normalize *within the pool* rather than assuming a formula like
    `1 - distance`, because Chroma's raw score is a distance whose scale depends
    on the collection's configured space (cosine / l2 / ip). Normalizing in-pool
    is metric-agnostic and always lands in [0,1], which is what the nudge weights
    above are calibrated against.

    Caveat: this is a RELATIVE score, not an absolute confidence. Every pool has a
    1.0 and a 0.0 even if every chunk in it is junk — so never threshold on it to
    decide "do we have a good answer?". That question needs an absolute score.
    """
    if not distances:
        return []
    d_min, d_max = min(distances), max(distances)
    span = d_max - d_min
    if span <= 0:                      # all identical (or a single candidate)
        return [1.0] * len(distances)
    return [(d_max - d) / span for d in distances]


def rerank_results(scored_docs: list, rule: dict) -> list:
    """
    Rank candidates by semantic relevance, nudged by metadata, capped for diversity.

    Args:
        scored_docs: list of (Document, raw_distance) from the vector store
        rule:        the intent rule (supplies preferred / penalty doc_types)

    Returns:
        top-K Documents, with rerank_rank / rerank_score / semantic_score attached.

    Does NOT mutate page_content — retrieve() strips context headers once, after
    ranking is settled, so this stays safe to call twice on the fallback path.
    """
    if not scored_docs:
        return []

    preferred = set(rule.get("preferred", []))
    penalty   = set(rule.get("penalty", []))

    pool      = _dedupe(scored_docs)
    relevance = _normalize_relevance([distance for _, distance in pool])

    scored = []
    for (doc, distance), rel in zip(pool, relevance):
        doc_type = doc.metadata.get("doc_type", "")
        try:
            priority = int(doc.metadata.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3

        nudge = PRIORITY_BOOST.get(priority, 0.0)
        if doc_type in preferred:
            nudge += PREFERRED_BOOST
        if doc_type in penalty:
            nudge -= PENALTY_WEIGHT

        scored.append({
            "doc"      : doc,
            "final"    : rel + nudge,
            "relevance": rel,
            "nudge"    : nudge,
            "distance" : distance,
        })

    scored.sort(key=lambda s: s["final"], reverse=True)

    # Diversity cap. Contextual chunking means adjacent chunks of one page are near
    # duplicates in embedding space, so a pure top-K can hand the LLM five slices of
    # the same doc and nothing else. Capping per source file forces coverage across
    # documents — this is the job MMR was nominally doing, done explicitly.
    selected: list = []
    per_file: dict = {}
    for item in scored:
        if len(selected) >= K:
            break
        source = item["doc"].metadata.get("source", "")
        if per_file.get(source, 0) >= MAX_PER_FILE:
            continue
        per_file[source] = per_file.get(source, 0) + 1
        selected.append(item)

    # If the cap starved the result set (small or duplicate-heavy pool), backfill by
    # score so we always return K chunks when K are available.
    if len(selected) < K:
        chosen = {id(item["doc"]) for item in selected}
        for item in scored:
            if len(selected) >= K:
                break
            if id(item["doc"]) not in chosen:
                selected.append(item)

    results = []
    for rank, item in enumerate(selected, start=1):
        doc = item["doc"]
        doc.metadata["rerank_rank"]    = rank
        doc.metadata["rerank_score"]   = round(item["final"], 3)
        doc.metadata["semantic_score"] = round(item["relevance"], 3)

        # semantic_score is normalized WITHIN the pool, so its top result is always
        # 1.0 — even when every candidate is junk. raw_distance is the absolute,
        # cross-query-comparable signal, and it's the only one that can answer "does
        # the corpus actually cover this question?" Carried through so Phase 2 can
        # calibrate a real "local context is too weak, go search the web" threshold
        # against the golden set, instead of leaving that call to the agent's vibes.
        doc.metadata["raw_distance"]   = round(item["distance"], 4)

        logger.info(
            f"   [{rank}] final={item['final']:.3f} "
            f"(relevance={item['relevance']:.3f} nudge={item['nudge']:+.3f}) | "
            f"type={doc.metadata.get('doc_type')} | "
            f"priority={doc.metadata.get('priority')} | "
            f"topic={doc.metadata.get('topic', '')} | "
            f"file={doc.metadata.get('file_name', '')}"
        )
        results.append(doc)

    return results


# ── Fallback Search ───────────────────────────────────────────────────────────
def targeted_fallback(query: str, vectorstore, preferred_types: list) -> list:
    """
    Targeted search filtered to the intent's preferred doc_types.

    Returns (Document, distance) tuples, NOT bare Documents. The caller merges
    these into the main candidate pool and re-ranks the whole thing in one pass —
    which only works if these carry distances on the same scale. They do: same
    embedding space, same metric, just a metadata filter applied.
    """
    for doc_type in preferred_types[:2]:
        try:
            found = vectorstore.similarity_search_with_score(
                query, k=FALLBACK_K, filter={"doc_type": doc_type}
            )
            if found:
                logger.info(f"   Fallback: {len(found)} chunks from doc_type={doc_type}")
                return found
        except Exception as e:
            logger.warning(f"   Fallback failed for doc_type={doc_type}: {e}")

    return []


# ── Main Retrieve Function ────────────────────────────────────────────────────
def retrieve(query: str, vectorstore=None) -> tuple[list, dict]:
    """
    Full retrieval pipeline. Returns (docs, rule) tuple.

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

    # Stage 2: scored candidate pool. similarity_search_with_score keeps the
    # distance that as_retriever(search_type="mmr") discards — that distance IS
    # the relevance signal the v4 reranker was missing.
    candidates = vectorstore.similarity_search_with_score(query, k=FETCH_K)
    logger.info(f"   Candidates: {len(candidates)}")

    # Stage 3: relevance-first re-rank
    results = rerank_results(candidates, rule)

    # Stage 4: fallback if the intent's preferred doc_types are underrepresented
    preferred_types = set(rule.get("preferred", []))
    preferred_count = sum(1 for d in results if d.metadata.get("doc_type") in preferred_types)

    if preferred_count < MIN_PREFERRED and rule["intent"] != "general":
        logger.warning(f"   Only {preferred_count}/{len(results)} preferred — triggering fallback")
        extra = targeted_fallback(query, vectorstore, rule["preferred"])
        if extra:
            # Re-rank the merged pool in one pass rather than splicing two ranked
            # lists — distances share a scale, so they normalize together correctly.
            results = rerank_results(candidates + extra, rule)

    # Stage 5: B-11 — strip the ingest-time [Category|...] header. Done once, here,
    # after ranking is settled, so the fallback path can't double-strip.
    for doc in results:
        doc.page_content = strip_context_header(doc.page_content)

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