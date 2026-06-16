"""§3.2 / §3.3 PageIndex — vectorless, reasoning-based retrieval.

A hierarchical table-of-contents tree is built from the heading structure (the
guide's §3.2 Option B self-built tree — no extra dependency). Retrieval is then an
LLM reading node titles/summaries and *reasoning* about which sections plausibly
hold the answer, the way a person navigates a handbook's TOC — no embeddings, no
chunking, no vector DB.

Each node carries .id/.text/.section_path/.page_start/.page_end, so it plugs into
the shared generate() and verify() unchanged.
"""
from dataclasses import dataclass, field

from .parse import Block
from .util import parse_json

SEARCH_PROMPT = """You are searching an employee handbook for sections that answer a question.
Below is the handbook's tree structure (node_id, title, summary, pages).
Return JSON only: {"thinking": "...", "node_ids": ["...", ...]}
Select ALL nodes needed — a question may span several policies (e.g. leave days AND support services).
If a selected section references another section, include the referenced node too.
If no node can answer, return {"node_ids": []}."""


@dataclass
class TreeNode:
    id: str
    title: str
    section_path: str
    page_start: int
    page_end: int
    text: str = ""                       # "[section_path]\n<body>" once finalized
    summary: str = ""
    bboxes: list = field(default_factory=list)
    children: list = field(default_factory=list)
    _blocks: list = field(default_factory=list, repr=False)


def build_tree(blocks: list[Block]) -> TreeNode:
    root = TreeNode(id="root", title="Employee Handbook", section_path="",
                    page_start=1, page_end=1)
    stack: list[tuple[float, TreeNode]] = [(float("inf"), root)]
    counter = 0
    for b in blocks:
        if b.is_heading:
            counter += 1
            while len(stack) > 1 and stack[-1][0] <= b.font_size:
                stack.pop()
            parent = stack[-1][1]
            path = f"{parent.section_path} > {b.text}" if parent.section_path else b.text
            node = TreeNode(id=f"n{counter:04d}", title=b.text, section_path=path,
                            page_start=b.page, page_end=b.page)
            parent.children.append(node)
            stack.append((b.font_size, node))
        else:
            stack[-1][1]._blocks.append(b)
    _finalize(root)
    return root


def _finalize(node: TreeNode) -> int:
    """Compute node text, summary, bboxes, and propagate page_end upward.

    node.text is the *whole section* — the node's own body plus all descendant
    text — so that selecting a parent section ("4 Leave Policies") still yields the
    content that physically lives in its subsections ("4.2 Bereavement Leave").
    """
    body = "\n".join(b.text for b in node._blocks)
    prefix = f"[{node.section_path}]\n" if node.section_path else ""
    own = f"{prefix}{body}" if body else ""
    node.bboxes = [(b.page, b.bbox) for b in node._blocks]
    node.summary = _summarize(node, body)
    last = max((b.page for b in node._blocks), default=node.page_start)
    child_texts = []
    for child in node.children:
        last = max(last, _finalize(child))
        child_texts.append(child.text)
    parts = [p for p in [own, *child_texts] if p.strip()]
    node.text = "\n\n".join(parts) if parts else node.title
    node.page_end = last
    return last


def _summarize(node: TreeNode, body: str) -> str:
    """Heuristic summary keeps index-building offline (cheap index, §3.1).

    Swap in an LLM call here for higher-quality node summaries — the rest of the
    pipeline is unchanged."""
    snippet = body.strip().replace("\n", " ")
    return (snippet[:160] + "…") if len(snippet) > 160 else (snippet or node.title)


def index_nodes(root: TreeNode) -> dict[str, TreeNode]:
    idx = {}
    def walk(n):
        idx[n.id] = n
        for c in n.children:
            walk(c)
    walk(root)
    return idx


def render_toc(root: TreeNode) -> str:
    lines = []
    def walk(n, depth):
        if n.id != "root":
            indent = "  " * (depth - 1)
            lines.append(f"{indent}{n.id} | {n.title} | {n.summary} | pp.{n.page_start}-{n.page_end}")
        for c in n.children:
            walk(c, depth + 1)
    walk(root, 0)
    return "\n".join(lines)


def tree_search(llm, query: str, root: TreeNode) -> tuple[list[str], dict]:
    """Single-call tree search — a 50–150pp handbook's TOC fits in one prompt (§3.3)."""
    toc = render_toc(root)
    resp = llm.chat(
        system=SEARCH_PROMPT,
        messages=[{"role": "user", "content": f"<toc>\n{toc}\n</toc>\n\nQuestion: {query}"}],
        max_tokens=800,
    )
    out = parse_json(resp.text)
    trace = {"thinking": out.get("thinking", ""), "node_ids": out.get("node_ids", [])}
    return out.get("node_ids", []), trace
