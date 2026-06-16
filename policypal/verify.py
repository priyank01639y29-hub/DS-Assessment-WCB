"""§1.3 Mechanical quote verification — the single highest-leverage trust feature.

The LLM asserts a quote; *code* (not another LLM) checks it exists in the cited
source. Runs identically over chunks (traditional/agentic) or tree-node text
(PageIndex). A claim with verified=False is flagged, never silently shown.
"""
import re
from difflib import SequenceMatcher

from .contract import Answer


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote: str, source_text: str, fuzz: float = 0.92) -> bool:
    q, src = normalize(quote), normalize(source_text)
    if not q:
        return False
    if q in src:
        return True
    # fuzzy fallback: tolerate hyphenation/ligature artifacts from PDF extraction
    m = SequenceMatcher(None, q, src).find_longest_match(0, len(q), 0, len(src))
    return m.size / max(len(q), 1) >= fuzz


def _numbers_ok(quote: str, claim_text: str) -> bool:
    """Stricter rule for numeric claims: every number asserted in the claim must
    appear in the cited quote (§1.3 note)."""
    claim_nums = set(re.findall(r"\d+", claim_text))
    if not claim_nums:
        return True
    quote_nums = set(re.findall(r"\d+", quote))
    return claim_nums <= quote_nums


def verify_answer(answer: Answer, get_source_text) -> Answer:
    """Set claim.verified by checking every citation's quote against its source.

    `get_source_text(citation) -> str` resolves a citation to the text it cites
    (chunk text or tree-node text, depending on the approach).
    """
    for claim in answer.claims:
        ok = bool(claim.citations)
        for c in claim.citations:
            src = get_source_text(c)
            if not verify_quote(c.quote, src) or not _numbers_ok(c.quote, claim.text):
                ok = False
                break
        claim.verified = ok
    return answer
