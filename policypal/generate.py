"""§2.4 Citation-constrained generation — shared by traditional RAG and PageIndex.

The LLM answers using ONLY the provided excerpts and must attach a verbatim quote
to every claim. The quote is what the verifier (§1.3) mechanically checks; the LLM
never decides whether a claim is "verified".

`sources` is any object exposing .id, .text, .section_path, .page_start — a Chunk
works directly, and PageIndex wraps tree nodes in the same shape.
"""
from .contract import Answer, Citation, Claim
from .util import parse_json

GEN_SYSTEM = """You answer HR policy questions using ONLY the provided handbook excerpts.
Return JSON only:
{"abstained": bool,
 "claims": [{"text": "...", "citations": [{"source_id": "...", "quote": "<VERBATIM text copied from that excerpt>"}]}]}
Rules:
- Every claim must carry >=1 citation. quote must be copied character-for-character from the excerpt.
- If the excerpts do not contain the answer, set abstained=true and claims=[].
- Never use knowledge outside the excerpts."""


def render_context(sources) -> str:
    return "\n\n".join(
        f"<chunk id='{s.id}' section='{s.section_path}' pages='{s.page_start}-{s.page_end}'>\n"
        f"{s.text}\n</chunk>"
        for s in sources
    )


def generate(llm, query: str, sources) -> Answer:
    by_id = {s.id: s for s in sources}
    ctx = render_context(sources)
    resp = llm.chat(
        system=GEN_SYSTEM,
        messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {query}"}],
        max_tokens=1500,
    )
    return parse_answer(resp.text, by_id)


def build_claims(raw_claims, resolve) -> list[Claim]:
    """Assemble Claim/Citation objects from the LLM's raw JSON claims.

    `resolve(source_id)` returns the cited source object (a Chunk or tree node)
    or None; the citation inherits its page/section metadata so the verifier and
    UI see one uniform shape regardless of which retrieval strategy produced it.
    Shared by traditional/PageIndex generation and the agentic loop.
    """
    claims = []
    for c in raw_claims:
        citations = []
        for cit in c.get("citations", []):
            sid = cit.get("source_id") or cit.get("chunk_id") or cit.get("node_id", "")
            src = resolve(sid)
            citations.append(Citation(
                source_id=sid,
                page=getattr(src, "page_start", 0) if src else 0,
                page_end=getattr(src, "page_end", 0) if src else 0,
                section_path=getattr(src, "section_path", "") if src else "",
                quote=cit.get("quote", ""),
            ))
        claims.append(Claim(text=c.get("text", ""), citations=citations))
    return claims


def parse_answer(text: str, by_id: dict) -> Answer:
    data = parse_json(text)
    if data.get("abstained"):
        return Answer(claims=[], abstained=True)
    claims = build_claims(data.get("claims", []), by_id.get)
    return Answer(claims=claims, abstained=not claims)
