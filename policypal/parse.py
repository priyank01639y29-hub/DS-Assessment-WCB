"""§1.1 Structured extraction with provenance.

Every downstream citation depends on capturing {page, bbox, font_size, heading}
at parse time — you cannot retrofit bounding boxes after chunking.
"""
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Block:
    text: str
    page: int            # 1-based
    bbox: tuple          # (x0, y0, x1, y1) on page — enables highlight rendering
    font_size: float
    is_heading: bool


def extract_blocks(pdf_path: str) -> list[Block]:
    doc = fitz.open(pdf_path)
    body_size = _modal_font_size(doc)          # most common size = body text
    blocks: list[Block] = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:                  # skip images
                continue
            spans = [s for l in b["lines"] for s in l["spans"]]
            if not spans:
                continue
            text = " ".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            bold = any(s["flags"] & 16 for s in spans)
            blocks.append(
                Block(
                    text=text,
                    page=pno,
                    bbox=tuple(b["bbox"]),
                    font_size=round(size, 1),
                    is_heading=(size > body_size * 1.15) or (bold and len(text) < 80),
                )
            )
    doc.close()
    return blocks


def _modal_font_size(doc) -> float:
    """Most common font size, weighted by character count = body text size."""
    sizes: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"])
    return sizes.most_common(1)[0][0] if sizes else 12.0


def print_outline(blocks: list[Block]) -> None:
    """Design-note sanity check: eyeball the inferred headings against the PDF TOC
    before trusting downstream chunking/tree building (§1.1 design note)."""
    for b in blocks:
        if b.is_heading:
            print(f"p{b.page:>3}  sz{b.font_size:>5}  {b.text}")
