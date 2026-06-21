"""
link_resolver.py — DocBrain v4.1
Builds the "For deeper understanding" link row shown under each answer.

Problem this solves:
  Previously, source links were either (a) guessed from local file paths with
  no relevance check (could be topically wrong — e.g. a "Philosophy" doc
  linked for an "LCEL" question), or (b) URLs the LLM wrote into the answer
  body that were never verified at all (hallucinated "Read full documentation"
  links).

This module produces a small, ranked, DEDUPED list of links that are:
  1. Topically relevant to the actual user question (not just top-ranked-by-MMR)
  2. Verified to resolve (no dead/wrong-page links)
  3. Labeled with a short, human-readable topic name for the button

Sources considered, in priority order:
  1. Locally retrieved docs whose topic/title overlaps with the question
  2. URLs the agent actually visited via scrape_url / returned via web_search
     during this turn (these are real, already-verified-to-exist pages)
"""

import re
from dataclasses import dataclass
from loguru import logger

from src.retriever import convert_source_to_url, _url_resolves

MAX_LINKS = 4
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "how", "do", "i", "to", "in",
    "for", "of", "and", "or", "vs", "versus", "use", "using", "when",
    "should", "can", "does", "with", "between", "difference",
}


@dataclass
class SourceLink:
    label: str
    url: str
    relevance: float


def _keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _topic_overlap_score(query_kw: set, doc_meta: dict) -> float:
    """Score how much a doc's topic/title/file_name overlaps the query keywords."""
    doc_text = " ".join([
        doc_meta.get("topic", ""),
        doc_meta.get("title", ""),
        doc_meta.get("file_name", ""),
        doc_meta.get("category_label", ""),
    ])
    doc_kw = _keywords(doc_text)
    if not doc_kw or not query_kw:
        return 0.0
    overlap = len(query_kw & doc_kw)
    return overlap / max(len(query_kw), 1)


def _label_from_meta(meta: dict) -> str:
    title = meta.get("title", "") or meta.get("file_name", "source")
    label = title.replace("_", " ").replace("-", " ").strip()
    label = " ".join(w.capitalize() for w in label.split())
    if len(label) > 28:
        label = label[:26] + "…"
    return label or "Documentation"


def resolve_links(
    question: str,
    docs: list,
    web_results: list | None = None,
) -> list[SourceLink]:
    """
    Build the final ranked, deduped, verified link list.

    Args:
        question:    the (rewritten) user question
        docs:        locally retrieved Document objects (with rerank_score in metadata)
        web_results: optional list of dicts {"title": ..., "url": ...} the agent
                     actually visited/found this turn (from web_search/scrape_url
                     tool calls) — these are treated as pre-verified real pages

    Returns:
        list[SourceLink], ranked by relevance, deduped by URL, capped at MAX_LINKS
    """
    query_kw = _keywords(question)
    candidates: list[SourceLink] = []
    seen_urls = set()

    # ── Local docs: score by topical overlap with the actual question ───────
    for doc in docs:
        meta = doc.metadata
        url = convert_source_to_url(meta.get("source", ""), verify=True)
        if not url or not url.startswith("http") or url in seen_urls:
            continue

        score = _topic_overlap_score(query_kw, meta)
        # Small boost for priority-1 docs, but relevance dominates — this is
        # the key fix: we no longer just take "whatever ranked #1 in MMR".
        if int(meta.get("priority", 3)) == 1:
            score += 0.05

        if score <= 0:
            continue  # not actually relevant to the question — don't show it

        candidates.append(SourceLink(
            label=_label_from_meta(meta),
            url=url,
            relevance=score,
        ))
        seen_urls.add(url)

    # ── Web results the agent actually used this turn ────────────────────────
    for result in (web_results or []):
        url = result.get("url", "")
        title = result.get("title", "")
        if not url or url in seen_urls:
            continue
        if not _url_resolves(url):
            logger.warning(f"   [link-resolver] Web result did not resolve, skipping: {url}")
            continue

        label = " ".join(w.capitalize() for w in title.split()[:4]) or "Web Reference"
        if len(label) > 28:
            label = label[:26] + "…"

        candidates.append(SourceLink(
            label=label,
            url=url,
            relevance=_topic_overlap_score(query_kw, {"title": title}) + 0.02,
        ))
        seen_urls.add(url)

    # ── Rank, cap, return ─────────────────────────────────────────────────────
    candidates.sort(key=lambda c: c.relevance, reverse=True)
    final = candidates[:MAX_LINKS]

    if not final:
        logger.warning("   [link-resolver] No relevant verified links found for this answer")

    logger.info(f"   [link-resolver] {len(final)} link(s): {[c.label for c in final]}")
    return final
