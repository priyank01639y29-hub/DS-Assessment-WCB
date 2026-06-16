"""Approach 2 — PageIndex pipeline (§3).

    query → LLM tree-search over node titles/summaries → node_ids
          → load full node text → generate(JSON claims) → verify

Citations land on real document sections ("4.2 Bereavement Leave, p.23") — exactly
how an HR specialist would cite — and the tree-search "thinking" is logged as the
audit trail.
"""
from .contract import Answer
from .generate import generate
from .tree import TreeNode, index_nodes, tree_search
from .verify import verify_answer


def answer_pageindex(llm, root: TreeNode, query: str,
                     node_index: dict[str, TreeNode] | None = None) -> Answer:
    idx = node_index or index_nodes(root)
    node_ids, trace = tree_search(llm, query, root)
    nodes = [idx[n] for n in node_ids if n in idx]
    if not nodes:
        ans = Answer(claims=[], abstained=True)
        ans.retrieval = {"approach": "pageindex", "node_ids": node_ids, "trace": trace}
        return ans
    answer = generate(llm, query, nodes)            # TreeNode satisfies the source shape
    answer = verify_answer(answer, lambda c: idx[c.source_id].text if c.source_id in idx else "")
    answer.retrieval = {"approach": "pageindex", "node_ids": node_ids, "trace": trace}
    return answer
