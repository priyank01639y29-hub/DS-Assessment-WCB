"""Offline tests — run with: python -m pytest tests/  (forces the mock LLM).

These exercise parsing → chunking → tree → all three retrieval approaches and the
mechanical quote verifier, without any network/API key.
"""
import os
import sys

import pytest

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["STORE_BACKEND"] = "memory"   # keep tests hermetic (no DB dependency)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from policypal import PolicyPal
from policypal.verify import normalize, verify_quote

_ROOT = os.path.dirname(os.path.dirname(__file__))
HANDBOOK = os.path.join(_ROOT, "OmniCorp_Handbook_Challenge V2.pdf")
CASES = os.path.join(_ROOT, "eval", "handbook_cases.json")


# --- verifier unit tests (no PDF needed) ---------------------------------------
def test_verify_exact():
    assert verify_quote("five days", "Employees receive five days of leave.")


def test_verify_whitespace_insensitive():
    assert verify_quote("five   days", "Employees receive\nfive days.")


def test_verify_fuzzy_hyphenation():
    # ligature/hyphenation artifact tolerated by the fuzzy fallback
    assert verify_quote("co-operation between teams",
                        "We value cooperation between teams at all times", fuzz=0.85)


def test_verify_rejects_absent():
    assert not verify_quote("ten weeks of sabbatical", "There is no such policy here.")


def test_normalize():
    assert normalize("  Foo   BAR\n") == "foo bar"


# --- end-to-end over the real handbook, mock LLM -------------------------------
@pytest.fixture(scope="module")
def pal():
    if not os.path.exists(HANDBOOK):
        pytest.skip(f"handbook not found: {HANDBOOK}")
    return PolicyPal(HANDBOOK)


def test_parsing_produces_structure(pal):
    assert len(pal.blocks) > 0
    assert len(pal.chunks) > 0
    assert len(pal.node_index) > 1            # root + at least one section


@pytest.mark.parametrize("approach", ["traditional", "pageindex", "agentic"])
def test_each_approach_runs_and_verifies(pal, approach):
    ans = pal.answer("What is the leave policy?", approach)
    assert ans.retrieval.get("approach") == approach
    # the mock quotes real source text, so any claim it makes must verify mechanically
    for claim in ans.claims:
        assert claim.verified, f"{approach}: unverified claim {claim.text!r}"


@pytest.mark.parametrize("approach", ["traditional", "pageindex", "agentic"])
def test_token_usage_and_cost_populated(pal, approach):
    ans = pal.answer("What is the remote work policy?", approach)
    assert ans.usage["input_tokens"] > 0
    assert ans.usage["output_tokens"] > 0
    assert ans.usage["calls"] >= 1
    # cost = tokens × configured per-1M prices (default 3/15)
    assert ans.cost_usd > 0
    assert "tokens" in ans.usage_line() and "$" in ans.usage_line()


def test_citation_shows_page_number(pal):
    ans = pal.answer("What is the remote work policy?", "traditional")
    cits = [c for cl in ans.claims for c in cl.citations]
    assert cits, "expected at least one citation"
    for c in cits:
        assert c.page >= 1
        assert c.page_label.startswith("p")     # "p.N" or "pp.N-M"
        assert c.page_label in ans.render()


def test_llm_sampling_params_config_and_assembly(monkeypatch):
    from policypal.config import load_config
    from policypal.llm import MockLLM

    # The "unset" case below must not depend on ambient env: the deployment .env
    # (e.g. in the Docker container) sets TEMPERATURE/TOP_P/SEED, which load_config
    # would otherwise read. Strip them so this stays hermetic.
    for var in ("TEMPERATURE", "TOP_P", "SEED"):
        monkeypatch.delenv(var, raising=False)

    # unset -> None -> not sent
    base = MockLLM(load_config())
    assert base.temperature is None and base.top_p is None and base.seed is None
    assert base._sampling(include_seed=True) == {}

    # set -> carried and assembled; seed only for openai-style (include_seed)
    cfg = load_config(temperature=0.0, top_p=0.9, seed=42, max_tokens=1234)
    llm = MockLLM(cfg)
    assert llm.default_max_tokens == 1234
    assert llm._sampling(include_seed=True) == {"temperature": 0.0, "top_p": 0.9, "seed": 42}
    assert llm._sampling(include_seed=False) == {"temperature": 0.0, "top_p": 0.9}  # no seed (Anthropic)


def test_cost_estimate_formula():
    from policypal.config import estimate_cost, load_config
    cfg = load_config()
    cost = estimate_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000}, cfg)
    assert abs(cost - (cfg.cost_input_per_1m + cfg.cost_output_per_1m)) < 1e-9


# --- keyword search / analyzer (no PDF needed) ----------------------------------
def test_analyzer_preserves_identifiers_and_numbers():
    from policypal.keyword import analyze
    toks = analyze("Ref: POL-853-MEN requires exactly 14 characters and $1,000")
    assert "pol-853-men" in toks      # code kept verbatim, not split/stemmed
    assert "14" in toks               # number kept
    assert "1000" in toks             # thousands separator stripped, one token


def test_analyzer_stopwords_and_stemming():
    from policypal.keyword import analyze
    toks = analyze("What are the policies for passwords?")
    assert "the" not in toks and "are" not in toks    # stopwords removed
    assert "policy" in toks            # policies -> policy
    assert "password" in toks          # passwords -> password


def test_keyword_index_finds_exact_term():
    from policypal.keyword import KeywordIndex
    docs = [("a", "Remote work and flexible hours policy."),
            ("b", "All passwords must be exactly 14 characters long."),
            ("c", "Mental health and wellbeing resources, EAP counseling.")]
    idx = KeywordIndex(docs)
    top = idx.search("password length 14 characters", k=1)
    assert top and top[0][0] == "b"


def test_hybrid_never_trails_best_single_retriever(pal):
    from policypal.evaluation import evaluate_retrieval, load_cases
    rep = evaluate_retrieval(pal.store, load_cases(CASES), k=6)
    r = rep["recall"]
    best_single = max(r["dense"], r["keyword"])
    assert r["hybrid-rrf"] >= best_single - 1e-9     # fusion shouldn't lose recall


def test_parse_json_tolerates_raw_newlines_and_braces():
    from policypal.util import parse_json
    # a verbatim quote with a literal newline AND a brace — both are illegal under
    # strict JSON but LLMs emit them when copying multi-line source spans
    raw = ('{"abstained": false, "claims": [{"text": "x", "citations": '
           '[{"source_id": "c1", "quote": "line one\nline two {brace}"}]}]}')
    data = parse_json(raw)
    assert data["claims"][0]["citations"][0]["quote"] == "line one\nline two {brace}"
    # with leading prose (forces the brace-aware fallback scanner, not the fast path)
    data2 = parse_json("Sure, here you go:\n" + raw)
    assert data2["abstained"] is False


def test_eval_harness_computes_metrics(pal):
    from policypal.evaluation import evaluate, load_cases

    cases = load_cases(CASES)
    report = evaluate(pal, cases[:4], approaches=["traditional"], judge=False)
    m = report["traditional"]
    # mock quotes real source text → every citation must mechanically verify
    assert m["citation_validity"] == 1.0
    for key in ("abstention_accuracy", "evidence_recall"):
        assert 0.0 <= m[key] <= 1.0
