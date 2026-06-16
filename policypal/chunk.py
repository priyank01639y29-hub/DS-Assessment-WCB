"""§2.2 Section-aware chunking.

Heading-bounded chunks mean a chunk never straddles two policies — the dominant
failure mode of fixed-size chunking on policy documents (a chunk holding the tail
of "Vacation" and the head of "Sick Leave" yields confidently wrong day counts).
"""
from dataclasses import dataclass, field

from .parse import Block


@dataclass
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    section_path: str
    bboxes: list = field(default_factory=list)   # [(page, bbox), ...] per block


def _approx_tokens(buf: list[Block]) -> int:
    return sum(len(b.text) for b in buf) // 4     # ~4 chars/token


def chunk_by_section(blocks: list[Block], max_tokens: int = 450,
                     overlap_blocks: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: list[tuple[float, str]] = []   # (font_size, title) hierarchy
    buf: list[Block] = []

    def path() -> list[str]:
        return [t for _, t in heading_stack]

    def flush():
        nonlocal buf
        if not buf:
            return
        sect = " > ".join(path())
        prefix = f"[{sect}]\n" if sect else ""
        chunks.append(
            Chunk(
                id=f"c{len(chunks):04d}",
                text=f"{prefix}" + "\n".join(b.text for b in buf),
                page_start=buf[0].page,
                page_end=buf[-1].page,
                section_path=sect,
                bboxes=[(b.page, b.bbox) for b in buf],
            )
        )
        buf = []

    for b in blocks:
        if b.is_heading:
            flush()
            # pop siblings/deeper headings (>= this size), then nest under parent
            while heading_stack and heading_stack[-1][0] <= b.font_size:
                heading_stack.pop()
            heading_stack.append((b.font_size, b.text))
            continue
        buf.append(b)
        if _approx_tokens(buf) > max_tokens:
            overlap = buf[-overlap_blocks:] if overlap_blocks else []
            flush()
            buf = list(overlap)
    flush()
    return chunks
