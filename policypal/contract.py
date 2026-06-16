"""§1.2 The citation contract — the shared output schema for all three pipelines.

Writing this once means the verification layer and any UI are written once too,
regardless of which retrieval strategy produced the answer.
"""
from dataclasses import dataclass, field


@dataclass
class Citation:
    source_id: str       # chunk id (traditional/agentic) or tree node_id (PageIndex)
    page: int            # first page of the cited source (1-based)
    section_path: str    # e.g. "4 Leave Policies > 4.2 Bereavement Leave"
    quote: str           # verbatim span the model claims supports the answer
    page_end: int = 0    # last page if the source spans multiple pages (else == page)

    @property
    def page_label(self) -> str:
        if self.page_end and self.page_end != self.page:
            return f"pp.{self.page}-{self.page_end}"
        return f"p.{self.page}"


@dataclass
class Claim:
    text: str                              # one atomic statement in the answer
    citations: list[Citation] = field(default_factory=list)
    verified: bool = False                 # set by the verifier, never by the LLM


@dataclass
class Answer:
    claims: list[Claim] = field(default_factory=list)
    abstained: bool = False                # True => "not found in handbook"
    retrieval: dict = field(default_factory=dict)   # approach-specific audit trail
    usage: dict = field(default_factory=dict)       # {input_tokens, output_tokens, calls}
    cost_usd: float = 0.0                   # estimated cost from .env price config
    latency_s: float = 0.0                  # wall-clock seconds for the whole answer (set by pipeline)

    @property
    def fully_verified(self) -> bool:
        return self.abstained or (
            bool(self.claims) and all(c.verified for c in self.claims)
        )

    def usage_line(self) -> str:
        u = self.usage or {}
        lat = f"  ·  {self.latency_s:.2f}s" if self.latency_s else ""
        return (f"— tokens: {u.get('input_tokens', 0):,} in + "
                f"{u.get('output_tokens', 0):,} out  ·  {u.get('calls', 0)} LLM call(s)"
                f"{lat}  ·  est. cost ${self.cost_usd:.4f}")

    def render(self) -> str:
        """Human-readable answer with a verification flag and page number per claim."""
        lines = []
        if self.abstained:
            lines.append("ABSTAINED — the handbook does not appear to contain this answer.")
        for claim in self.claims:
            mark = "✅" if claim.verified else "⚠️ UNVERIFIED"
            lines.append(f"{mark} {claim.text}")
            for c in claim.citations:
                where = c.section_path or c.source_id
                lines.append(f"     ↳ [{where}, {c.page_label}] “{c.quote}”")
        if self.usage:
            lines.append(self.usage_line())
        return "\n".join(lines)
