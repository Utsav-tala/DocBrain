"""
evaluation.py — retrieval evaluation harness.

Run:  python -m src.evaluation
      python -m src.evaluation --k 5 --label v5

WHY NOT RAGAS (it's in requirements.txt but unused):
  1. It doesn't import in this venv — langchain_community version conflict.
  2. More importantly, retrieval quality needs no LLM at all. recall@k and MRR are
     pure set comparison against labels: deterministic, free, instant, reproducible.
     An LLM judge adds cost, variance, and a second thing to debug — and it cannot
     answer the question this harness exists to answer, which is "did we retrieve the
     RIGHT DOCUMENT?" Answer-quality metrics (faithfulness, relevancy) do need a judge
     and are a separate, later concern. Measure retrieval first: it is upstream of
     everything else, and if it is wrong, nothing downstream can be right.

LABELS ARE AT SOURCE-FILE GRANULARITY, NOT CHUNK GRANULARITY:
  - 13,093 chunks is not hand-labellable. 103 conceptual doc files is.
  - Chunk IDs shift on every re-ingest; file paths are stable across re-embeds.
  - "Did we retrieve from the right document?" is the question that actually matters.
  `relevant_sources` entries are matched as SUBSTRINGS of a chunk's `source` path, so
  a label like "oss/langchain/streaming" matches regardless of directory prefix.

COVERAGE-GAP QUESTIONS:
  Some questions have no correct answer in the corpus (e.g. "What is LCEL?" — the
  scrape contains no LCEL explainer). Scoring those as recall=0 would be wrong: it
  blames the retriever for a hole in the DATA. They are tagged coverage="gap",
  excluded from the recall/MRR aggregates, and scored on a different question
  entirely — "did the system NOTICE it couldn't answer?" — which is what
  calibrate_gap_threshold() measures, using absolute raw_distance.
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.retriever import load_vectorstore, retrieve

GOLDEN_DEFAULT = "eval_data/golden_set.jsonl"
RESULTS_DIR    = "results"


# ── Data ──────────────────────────────────────────────────────────────────────
@dataclass
class QuestionResult:
    id: str
    question: str
    coverage: str
    intent: str
    relevant_sources: list
    retrieved_sources: list
    hit: float               # 1.0 if any labelled source landed in the top k
    recall: float            # fraction of labelled sources found in the top k
    mrr: float               # 1 / rank of first labelled source (0.0 if none)
    top1_distance: float     # absolute distance of the top chunk — the gap signal
    notes: str = ""


@dataclass
class Report:
    k: int
    covered: list = field(default_factory=list)
    gaps: list = field(default_factory=list)


# ── Golden set ────────────────────────────────────────────────────────────────
def load_golden_set(path: str) -> list:
    entries = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON — {e}") from e

            if item.get("coverage", "covered") == "covered" and not item.get("relevant_sources"):
                raise ValueError(
                    f"{path}:{line_no} ({item.get('id')}) is coverage='covered' but has no "
                    f"relevant_sources. Either label it, or mark it coverage='gap'."
                )
            entries.append(item)
    return entries


def _matches(source: str, patterns: list) -> bool:
    return any(p in source for p in patterns)


# ── Metrics ───────────────────────────────────────────────────────────────────
def score_one(item: dict, docs: list) -> QuestionResult:
    """
    hit@k    : did ANY labelled source appear in the top k?
    recall@k : what FRACTION of the labelled sources appeared? A question can have
               several legitimately-correct docs — memory is documented across three
               separate files — so demanding one exact file would be a mislabel, not
               a miss.
    MRR      : 1 / rank of the first labelled source. Rewards ranking it FIRST, not
               merely somewhere in the top k. This is the metric the v4→v5 reranker
               fix should move most, because v4's failure was one of ORDERING: it
               retrieved the right chunk into the pool and then sorted it away.
    """
    retrieved = [d.metadata.get("source", "") for d in docs]
    relevant  = item.get("relevant_sources", [])

    hit_ranks = [i for i, src in enumerate(retrieved) if _matches(src, relevant)]
    found     = {p for p in relevant if any(p in src for src in retrieved)}

    return QuestionResult(
        id                = item["id"],
        question          = item["question"],
        coverage          = item.get("coverage", "covered"),
        intent            = "",
        relevant_sources  = relevant,
        retrieved_sources = retrieved,
        hit               = 1.0 if hit_ranks else 0.0,
        recall            = len(found) / len(relevant) if relevant else 0.0,
        mrr               = 1.0 / (hit_ranks[0] + 1) if hit_ranks else 0.0,
        top1_distance     = docs[0].metadata.get("raw_distance", float("nan")) if docs else float("nan"),
        notes             = item.get("notes", ""),
    )


def calibrate_gap_threshold(covered: list, gaps: list) -> dict:
    """
    Fit a raw_distance threshold separating "the corpus covers this" from "it doesn't"
    — i.e. the point where the agent should stop trusting local docs and search the web.

    That decision is currently left to the agent's judgment. This turns it into a
    measurement.

    Sweeps every candidate threshold and picks the one maximising Youden's J
    (sensitivity + specificity - 1), and reports the confusion counts alongside it, so
    a BAD separation shows up as a bad J instead of hiding behind a confident-looking
    number. A threshold fit on a handful of points is a hypothesis, not a result.
    """
    if not covered or not gaps:
        return {"threshold": None, "reason": "need both covered and gap questions to calibrate"}

    cov_d = [r.top1_distance for r in covered if r.top1_distance == r.top1_distance]
    gap_d = [r.top1_distance for r in gaps    if r.top1_distance == r.top1_distance]
    if not cov_d or not gap_d:
        return {"threshold": None, "reason": "no distances recorded"}

    best = {"threshold": None, "j": -1.0}
    for t in sorted(set(cov_d + gap_d)):
        # predict "this is a coverage gap" when top1_distance >= t
        tp = sum(1 for d in gap_d if d >= t)      # gap correctly flagged
        fn = len(gap_d) - tp
        fp = sum(1 for d in cov_d if d >= t)      # covered question wrongly flagged
        tn = len(cov_d) - fp
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        j = sensitivity + specificity - 1
        if j > best["j"]:
            best = {
                "threshold"   : round(t, 4),
                "j"           : round(j, 3),
                "sensitivity" : round(sensitivity, 3),
                "specificity" : round(specificity, 3),
                "gaps_caught" : f"{tp}/{len(gap_d)}",
                "false_alarms": f"{fp}/{len(cov_d)}",
            }

    best["covered_distance_median"] = round(statistics.median(cov_d), 4)
    best["gap_distance_median"]     = round(statistics.median(gap_d), 4)
    return best


# ── Runner ────────────────────────────────────────────────────────────────────
def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run(golden_path: str = GOLDEN_DEFAULT, k: int = 5, label: str = "current") -> Report:
    golden = load_golden_set(golden_path)
    vs     = load_vectorstore()
    report = Report(k=k)

    print(f"Evaluating {len(golden)} questions from {golden_path} ...")

    for item in golden:
        docs, rule = retrieve(item["question"], vs)
        result = score_one(item, docs[:k])
        result.intent = rule["intent"]
        (report.gaps if result.coverage == "gap" else report.covered).append(result)

    _print_report(report, label)
    _save_report(report, label, golden_path, k)
    return report


def _print_report(report: Report, label: str) -> None:
    cov, gaps, k = report.covered, report.gaps, report.k

    print()
    print("=" * 74)
    print(f"RETRIEVAL EVALUATION — {label}   (k={k})")
    print("=" * 74)

    if cov:
        print(f"\nCOVERED QUESTIONS ({len(cov)}) — is the right doc retrieved, and ranked high?")
        print(f"  hit@{k}     {_mean([r.hit for r in cov]):.3f}   (any labelled doc in top {k})")
        print(f"  recall@{k}  {_mean([r.recall for r in cov]):.3f}   (fraction of labelled docs found)")
        print(f"  MRR       {_mean([r.mrr for r in cov]):.3f}   (1/rank of first labelled doc)")

        by_intent = defaultdict(list)
        for r in cov:
            by_intent[r.intent].append(r)
        print("\n  By intent (the aggregate hides a lot — the corpus is 41% GitHub issues):")
        print(f"    {'intent':<18} {'n':>3}  {'hit':>5} {'recall':>7} {'MRR':>6}")
        for intent, rs in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
            print(f"    {intent:<18} {len(rs):>3}  {_mean([r.hit for r in rs]):>5.2f} "
                  f"{_mean([r.recall for r in rs]):>7.2f} {_mean([r.mrr for r in rs]):>6.2f}")

        misses = [r for r in cov if r.hit == 0.0]
        if misses:
            print(f"\n  COMPLETE MISSES ({len(misses)}) — labelled doc never retrieved at all:")
            for r in misses:
                got = [s.split("/")[-1] for s in r.retrieved_sources[:3]]
                print(f"    [{r.id}] {r.question}")
                print(f"          want: {r.relevant_sources}")
                print(f"          got : {got}")

    if gaps:
        print(f"\nCOVERAGE-GAP QUESTIONS ({len(gaps)}) — excluded from the metrics above.")
        print("  Scored on a different question: does the system NOTICE it can't answer?")
        calib = calibrate_gap_threshold(cov, gaps)
        if calib.get("threshold") is None:
            print(f"  Not calibratable: {calib.get('reason')}")
        else:
            print(f"\n  Top-1 distance (median):  covered={calib['covered_distance_median']}   "
                  f"gap={calib['gap_distance_median']}")
            print(f"  Best separating threshold: raw_distance >= {calib['threshold']}  →  search the web")
            print(f"    catches {calib['gaps_caught']} gaps, {calib['false_alarms']} false alarms "
                  f"on covered questions   (Youden's J = {calib['j']})")
            if calib["j"] < 0.5:
                print("    ⚠️  J < 0.5 — distance does NOT separate these cleanly. Do not ship this threshold.")

    print()


def _save_report(report: Report, label: str, golden_path: str, k: int) -> None:
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    cov = report.covered
    out = {
        "label"     : label,
        "golden_set": golden_path,
        "k"         : k,
        "n_covered" : len(cov),
        "n_gaps"    : len(report.gaps),
        "metrics"   : {
            f"hit@{k}"   : round(_mean([r.hit for r in cov]), 4),
            f"recall@{k}": round(_mean([r.recall for r in cov]), 4),
            "mrr"        : round(_mean([r.mrr for r in cov]), 4),
        },
        "gap_calibration": calibrate_gap_threshold(cov, report.gaps),
        "per_question": [
            {
                "id": r.id, "question": r.question, "coverage": r.coverage,
                "intent": r.intent, "hit": r.hit, "recall": round(r.recall, 3),
                "mrr": round(r.mrr, 3), "top1_distance": r.top1_distance,
                "relevant_sources" : r.relevant_sources,
                "retrieved_sources": r.retrieved_sources,
            }
            for r in cov + report.gaps
        ],
    }
    path = Path(RESULTS_DIR) / f"eval_{label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  → saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocBrain retrieval evaluation")
    parser.add_argument("--golden", default=GOLDEN_DEFAULT, help="path to golden set JSONL")
    parser.add_argument("--k", type=int, default=5, help="cutoff for hit@k / recall@k")
    parser.add_argument("--label", default="current",
                        help="name this run (e.g. v4-baseline, v5) — used in the results/ filename")
    args = parser.parse_args()

    logger.remove()  # the retriever is chatty; the report is the output
    run(args.golden, args.k, args.label)


if __name__ == "__main__":
    main()
