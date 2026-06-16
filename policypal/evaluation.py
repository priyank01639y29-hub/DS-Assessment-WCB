"""Evaluation framework (§5 of the guide).

Runs a curated Q/A/evidence set through each approach and scores four mechanical
metrics plus an optional LLM-judge faithfulness metric. All but faithfulness are
deterministic and need no API key, so the harness itself is testable offline.

Metrics (per approach):
  * abstention_accuracy — answerable→answered, unanswerable→abstained
  * answer_correctness  — every expected key phrase/number present in the answer
  * evidence_recall     — a gold evidence phrase appears in a *cited* source's text
  * citation_validity   — fraction of citations whose quote mechanically verifies (§1.3)
  * faithfulness        — (optional, LLM judge) claims actually supported by their quotes

The contradiction trap (Section 1 "3 days/week" superseded by the June-2024 amendment
to "1 day/week, Fridays") and the unanswerables (no bereavement/parental/401k policy)
are what separate a trustworthy pipeline from a plausible-sounding one.
"""
import json
from dataclasses import dataclass, field

from .agent import SourceResolver
from .contract import Answer
from .util import parse_json
from .verify import normalize, verify_quote


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool
    expected: list[str] = field(default_factory=list)   # all must appear → correct
    evidence: list[str] = field(default_factory=list)    # any in a cited source → recall hit
    note: str = ""


@dataclass
class CaseResult:
    case_id: str
    approach: str
    answerable: bool
    abstained: bool
    abstention_ok: bool
    correctness: bool | None          # None when not applicable (unanswerable/abstained)
    evidence_recall: bool | None
    citations_total: int
    citations_valid: int
    faithfulness: float | None
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    answer: Answer | None = None   # the full answer, so callers can display Q/A/result


def load_cases(path: str) -> list[EvalCase]:
    with open(path) as f:
        return [EvalCase(**c) for c in json.load(f)]


def evaluate_case(pal, case: EvalCase, approach: str, judge: bool = False) -> CaseResult:
    ans: Answer = pal.answer(case.question, approach)
    resolver = SourceResolver(pal.store, pal.node_index)

    abstention_ok = ans.abstained == (not case.answerable)
    cits = [c for cl in ans.claims for c in cl.citations]
    cits_valid = sum(1 for c in cits if verify_quote(c.quote, resolver.text(c.source_id)))

    correctness = evidence_recall = None
    if case.answerable:
        if ans.abstained:
            correctness, evidence_recall = False, False
        else:
            pool = normalize(" ".join([cl.text for cl in ans.claims]
                                      + [c.quote for c in cits]))
            correctness = all(normalize(e) in pool for e in case.expected) \
                if case.expected else None
            src_pool = normalize(" ".join(resolver.text(c.source_id) for c in cits))
            evidence_recall = any(normalize(e) in src_pool for e in case.evidence) \
                if case.evidence else None

    faith = None
    if judge and not ans.abstained and ans.claims:
        faith = judge_faithfulness(pal.llm, case.question, ans, resolver)

    return CaseResult(case.id, approach, case.answerable, ans.abstained, abstention_ok,
                      correctness, evidence_recall, len(cits), cits_valid, faith,
                      in_tokens=ans.usage.get("input_tokens", 0),
                      out_tokens=ans.usage.get("output_tokens", 0),
                      cost_usd=ans.cost_usd, latency_s=ans.latency_s, answer=ans)


JUDGE_SYSTEM = """You are a strict grader. For each claim, decide if the quote(s) actually
support the claim's statement. Return JSON only: {"verdicts": [true|false, ...]} with one
boolean per claim, in order. Judge support only — not whether the claim is true in general."""


def judge_faithfulness(llm, question: str, answer: Answer, resolver) -> float:
    items = []
    for i, cl in enumerate(answer.claims):
        quotes = " | ".join(f'"{c.quote}"' for c in cl.citations)
        items.append(f"{i+1}. CLAIM: {cl.text}\n   QUOTES: {quotes}")
    body = f"Question: {question}\n\n" + "\n".join(items)
    try:
        resp = llm.chat(system=JUDGE_SYSTEM,
                        messages=[{"role": "user", "content": body}], max_tokens=300)
        verdicts = parse_json(resp.text).get("verdicts", [])
        verdicts = [bool(v) for v in verdicts][: len(answer.claims)]
        return sum(verdicts) / len(answer.claims) if answer.claims else None
    except Exception:
        return None


def evaluate(pal, cases: list[EvalCase], approaches=("traditional", "pageindex", "agentic"),
             judge: bool = False) -> dict:
    """Returns {approach: {metric: value, "results": [CaseResult, ...]}}."""
    out = {}
    for approach in approaches:
        results = [evaluate_case(pal, c, approach, judge) for c in cases]
        out[approach] = {**_aggregate(results), "results": results}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _aggregate(results: list[CaseResult]) -> dict:
    tot = sum(r.citations_total for r in results)
    valid = sum(r.citations_valid for r in results)
    n = len(results) or 1
    return {
        "n": len(results),
        "abstention_accuracy": _mean([r.abstention_ok for r in results]),
        "answer_correctness": _mean([r.correctness for r in results]),
        "evidence_recall": _mean([r.evidence_recall for r in results]),
        "citation_validity": (valid / tot) if tot else None,
        "faithfulness": _mean([r.faithfulness for r in results]),
        "tokens_per_query": sum(r.in_tokens + r.out_tokens for r in results) / n,
        "cost_per_query": sum(r.cost_usd for r in results) / n,
        "total_cost": sum(r.cost_usd for r in results),
        "latency_per_query": sum(r.latency_s for r in results) / n,
        "total_latency": sum(r.latency_s for r in results),
    }


def evaluate_retrieval(store, cases: list[EvalCase], k: int = 6) -> dict:
    """Retrieval-only recall@k per method — no LLM needed.

    Quantifies the hybrid-search payoff: a method "hits" a case if a gold evidence
    phrase appears in the top-k retrieved chunk texts. Compares dense (vector) vs
    keyword (BM25) vs the two fusions. The point: hybrid should never trail the best
    single retriever, and keyword rescues exact-identifier/number cases vectors miss.
    """
    from .retrieve import hybrid_search

    methods = {
        "dense": lambda q: store.dense(q, k),
        "keyword": lambda q: store.lexical(q, k),
        "hybrid-rrf": lambda q: hybrid_search(store, q, k_final=k, method="rrf"),
        "hybrid-weighted": lambda q: hybrid_search(store, q, k_final=k, method="weighted"),
    }
    answerable = [c for c in cases if c.answerable and c.evidence]
    out = {"k": k, "n": len(answerable), "recall": {}, "per_case": {}}
    for name, fn in methods.items():
        hits = 0
        per = []
        for c in answerable:
            ids = fn(c.question)
            pool = normalize(" ".join(store.get_text(i) for i in ids))
            hit = any(normalize(e) in pool for e in c.evidence)
            hits += hit
            per.append((c.id, hit))
        out["recall"][name] = hits / len(answerable) if answerable else None
        out["per_case"][name] = per
    return out


def render_retrieval(report: dict) -> str:
    lines = [f"retrieval recall@{report['k']} over {report['n']} answerable cases:"]
    for name, r in report["recall"].items():
        bar = "█" * round((r or 0) * 20)
        lines.append(f"  {name:18} {('n/a' if r is None else f'{r:.2f}'):>5}  {bar}")
    return "\n".join(lines)


def render_report(report: dict) -> str:
    quality = ["abstention_accuracy", "answer_correctness", "evidence_recall",
               "citation_validity", "faithfulness"]
    approaches = list(report)
    w = 22
    lines = ["metric".ljust(w) + "".join(a.ljust(14) for a in approaches)]
    lines.append("-" * (w + 14 * len(approaches)))
    for m in quality:
        row = m.ljust(w)
        for a in approaches:
            v = report[a][m]
            row += ("  n/a" if v is None else f"{v:6.2f}").ljust(14)
        lines.append(row)
    lines.append("-" * (w + 14 * len(approaches)))
    # cost rows (different formatting)
    row = "tokens/query".ljust(w)
    for a in approaches:
        row += f"{report[a]['tokens_per_query']:6.0f}".ljust(14)
    lines.append(row)
    row = "cost/query ($)".ljust(w)
    for a in approaches:
        row += f"{report[a]['cost_per_query']:8.5f}".ljust(14)
    lines.append(row)
    row = "latency/query (s)".ljust(w)
    for a in approaches:
        row += f"{report[a]['latency_per_query']:8.2f}".ljust(14)
    lines.append(row)
    return "\n".join(lines)
