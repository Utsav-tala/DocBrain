"""
output_refiner.py — DocBrain v4.2
Post-generation verification pass. Runs AFTER the agent produces an answer,
BEFORE it's returned to the user.

Four independent checks:
  1. Groundedness    — does every code/factual claim trace back to context? (LLM call)
  2. Format           — did the answer follow the style rules? (regex, no LLM call)
  3. Source line      — the trailing "Source: <url>" line is now REMOVED from the
                         body entirely (the UI renders verified links separately via
                         link_resolver.py — see chain.py). We just strip it here so
                         it doesn't show twice.
  4. Inline link scan — any OTHER http(s) URL the LLM wrote into the prose (e.g. a
                         "Read full documentation" style link) is checked against the
                         verified URL set. Anything not verified is stripped from the
                         text rather than shown to the user — this was the source of
                         the hallucinated-link bug (LLM writing a plausible-looking
                         URL that was never actually retrieved or searched).

Design principle: cheap checks (format, links) run first and are pure functions —
no LLM cost. Groundedness is the only LLM call, and only runs if the answer has
code blocks (the highest-risk hallucination surface).
"""

import re
from dataclasses import dataclass, field
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import convert_source_to_url

# ── Banned headings (mirrors prompt_templates.py rules) ──────────────────────
BANNED_HEADINGS = [
    "what is it", "why does it exist", "how it works", "worked example",
    "example", "key features", "summary", "answer", "explanation",
    "overview", "key differences", "sources",
]

_URL_RE = re.compile(r"https?://[^\s\)\]]+")

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checker for a documentation assistant.

You will be given a generated ANSWER and the CONTEXT it was supposed to be based on.

Check ONLY the code blocks and specific technical claims (function names, parameter
names, import paths, class names) in the ANSWER. For each one, decide if it is
directly supported by the CONTEXT.

Output format — ONLY this, nothing else:
GROUNDED: yes
or
GROUNDED: no
ISSUES: <one short bullet per unsupported claim, e.g. "- `from langchain.foo import Bar` not present in context">

If everything is supported, output only "GROUNDED: yes" with no ISSUES line."""),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


@dataclass
class RefinerReport:
    is_grounded: bool = True
    groundedness_issues: list = field(default_factory=list)
    format_ok: bool = True
    format_issues: list = field(default_factory=list)
    links_stripped: list = field(default_factory=list)  # unverified URLs removed from body
    final_answer: str = ""


# ── Check 1: Groundedness (LLM call, only if code present) ──────────────────
def check_groundedness(answer: str, context: str, llm) -> tuple[bool, list]:
    if "```" not in answer:
        # No code blocks — lowest hallucination risk, skip the LLM call
        return True, []

    try:
        chain = GROUNDEDNESS_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"context": context, "answer": answer}).strip()

        is_grounded = result.lower().startswith("grounded: yes")
        issues = []
        if not is_grounded:
            issues_match = re.search(r"ISSUES:\s*(.+)", result, re.DOTALL)
            if issues_match:
                issues = [
                    line.strip("- ").strip()
                    for line in issues_match.group(1).strip().splitlines()
                    if line.strip()
                ]
        return is_grounded, issues

    except Exception as e:
        logger.warning(f"   [refiner] Groundedness check failed ({e}), passing through")
        return True, []


# ── Check 2: Format compliance (pure regex, no LLM) ──────────────────────────
def check_format_compliance(answer: str) -> tuple[bool, list]:
    issues = []

    # Banned headings
    for heading in BANNED_HEADINGS:
        pattern = rf"^#{1,4}\s*{re.escape(heading)}\s*$"
        if re.search(pattern, answer, re.IGNORECASE | re.MULTILINE):
            issues.append(f"Banned heading found: '## {heading}'")

    # Bold intro on first non-empty line
    first_line = next((l for l in answer.strip().splitlines() if l.strip()), "")
    if not re.search(r"\*\*.+?\*\*", first_line):
        issues.append("First line is not bolded (style rule #1)")

    # Unclosed code fences
    if answer.count("```") % 2 != 0:
        issues.append("Unclosed code block (odd number of ``` fences)")

    return len(issues) == 0, issues


# ── Check 3+4: Strip Source: line, validate/strip any other inline URLs ─────
def sanitize_links(answer: str, verified_urls: set) -> tuple[str, list]:
    """
    Two jobs:
      1. Remove the trailing "Source: <url>" line entirely — the UI now shows
         verified links as a separate button row (see link_resolver.py), so a
         second link in the body is redundant and was the original wrong-page bug.
      2. Scan all OTHER http(s) URLs left in the body (e.g. an LLM-written
         "see https://..." reference). Any URL not in verified_urls gets
         stripped out — this is what catches a hallucinated "Read full
         documentation" style link before it ever reaches the user.

    Returns (cleaned_answer, list_of_stripped_urls) for logging/eval.
    """
    stripped = []

    # 1. Remove the Source: line
    answer = re.sub(
        r"^Source:\s*\S+\s*$", "", answer,
        flags=re.IGNORECASE | re.MULTILINE,
    ).rstrip()

    # 2. Scan remaining URLs, strip anything unverified
    def _check(match):
        url = match.group(0)
        if url in verified_urls:
            return url
        stripped.append(url)
        return "[link removed — could not be verified]"

    answer = _URL_RE.sub(_check, answer)

    if stripped:
        logger.warning(f"   [refiner] Stripped unverified link(s) from body: {stripped}")

    return answer, stripped


# ── Orchestrator ───────────────────────────────────────────────────────────
def refine(answer: str, docs: list, context: str, llm, resolved_links: list = None) -> RefinerReport:
    """
    Run all checks. Strips the Source: line and any unverified inline URLs
    (auto-fix, always applied). Does NOT auto-fix groundedness or format
    issues (those need regeneration — left to the caller to decide).

    Args:
        resolved_links: list[SourceLink] from link_resolver.resolve_links(),
                         already computed for this turn. Their URLs form the
                         "verified" set — anything else in the body gets stripped.
    """
    report = RefinerReport()

    # Cheap checks first
    report.format_ok, report.format_issues = check_format_compliance(answer)

    verified_urls = {link.url for link in (resolved_links or [])}
    answer, report.links_stripped = sanitize_links(answer, verified_urls)

    # Expensive check last, only if code present
    report.is_grounded, report.groundedness_issues = check_groundedness(answer, context, llm)

    report.final_answer = answer

    if not (report.is_grounded and report.format_ok) or report.links_stripped:
        logger.warning(
            f"   [refiner] Issues found | grounded={report.is_grounded} "
            f"format_ok={report.format_ok} links_stripped={len(report.links_stripped)}"
        )

    return report