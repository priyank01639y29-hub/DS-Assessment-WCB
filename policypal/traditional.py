"""Approach 1 — Traditional RAG (§2).

    query → [dense top-k] ⊕ [BM25 top-k] → RRF → top-n → generate(JSON claims) → verify

One LLM call per query: cheapest and lowest latency. Single-shot retrieval is the
trade-off — multi-part questions may surface only one facet (that's what the
agentic loop in §4 is for).
"""
from .contract import Answer
from .generate import generate
from .retrieve import hybrid_search
from .verify import verify_answer


def answer_traditional(llm, store, query: str, k_each: int = 20,
                       k_final: int = 6) -> Answer:
    ids = hybrid_search(store, query, k_each=k_each, k_final=k_final)
    sources = [store.by_id[i] for i in ids]
    answer = generate(llm, query, sources)
    answer = verify_answer(answer, lambda c: store.get_text(c.source_id))
    answer.retrieval = {"approach": "traditional", "retrieved_ids": ids}
    return answer
