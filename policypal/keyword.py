"""Keyword (lexical) search — the BM25 half of hybrid retrieval.

This is the technique behind "hybrid search" that NotebookLM and most production RAG
systems use: run a sparse keyword retriever (BM25) alongside the dense/vector
retriever and fuse the results. Keyword search supplies what embeddings are weakest
at — exact matches on rare, discriminative tokens: identifiers, codes, numbers,
acronyms (e.g. "POL-853-MEN", "401k", "14 characters", "$1,000"). A semantic vector
may drift right past those; BM25 nails them.

BM25 scoring (handled by rank_bm25's BM25Okapi):
    score(d, q) = Σ_t  IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
  * IDF(t)  — rarer terms across the corpus weigh more (probabilistic IDF).
  * k1      — term-frequency saturation: the 1st occurrence matters most,
              repeats give diminishing returns (typical 1.2–2.0).
  * b       — document-length normalization: stops long chunks winning on length
              alone (0 = none, 1 = full; typical 0.75).

The part that actually decides keyword-search quality — and that most tutorials skip
— is the ANALYZER: how text becomes tokens. The index and the query MUST be analyzed
identically, and the analyzer must NOT destroy the exact tokens keyword search exists
to catch. That is the design tension encoded below.
"""
import re
import unicodedata

from rank_bm25 import BM25Okapi

# A compact English stopword list. Deliberately conservative — we keep words that can
# carry meaning in policy questions (e.g. "many", "before", "after", "up", "down").
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "for", "with", "as", "by", "at", "from", "into", "is", "are", "was",
    "were", "be", "been", "being", "am", "this", "that", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "their", "do", "does", "did", "doing", "have", "has", "had", "having", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "what",
    "which", "who", "whom", "when", "where", "why", "how", "there", "here", "about",
    "so", "than", "too", "very", "just", "also", "any", "all", "no", "not",
}

# Strip thousands separators inside numbers so "$1,000" -> "1000" (one token).
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
# A token is an alnum run that may contain internal - _ / (keeps codes like POL-853-MEN).
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*")


def _is_identifier(tok: str) -> bool:
    """Identifiers/numbers/codes — kept verbatim (never stemmed or stop-listed),
    because exact-matching them is keyword search's whole advantage over vectors."""
    return any(ch.isdigit() for ch in tok) or any(c in tok for c in "-_/")


def _light_stem(w: str) -> str:
    """Deliberately conservative suffix stripper (plurals + common verb endings).

    Stemming raises recall ("policies" matches "policy") but over-stemming hurts
    precision, so this stays minimal. The only hard requirement is that index and
    query are stemmed the SAME way. For production, swap in a Snowball/Porter stemmer.
    """
    if len(w) <= 3:
        return w
    for suf, cut, add in (("sses", 2, ""), ("ies", 3, "y"), ("ied", 3, "y"),
                          ("ing", 3, ""), ("ed", 2, ""), ("es", 2, ""), ("s", 1, "")):
        if w.endswith(suf) and len(w) - cut >= 2:
            return w[:-cut] + add
    return w


def analyze(text: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Text -> tokens. Used identically at index time and query time."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _THOUSANDS.sub("", text)
    tokens = []
    for tok in _TOKEN.findall(text):
        if _is_identifier(tok):
            tokens.append(tok)                       # keep codes/numbers verbatim
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(_light_stem(tok) if stem else tok)
    return tokens


class KeywordIndex:
    """BM25 over an analyzed corpus, with tunable k1/b and a consistent analyzer."""

    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75,
                 stem: bool = True, remove_stopwords: bool = True):
        self.ids = [d[0] for d in docs]
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        corpus = [analyze(text, stem, remove_stopwords) for _, text in docs]
        # rank_bm25 needs at least one non-empty doc; guard the empty-corpus case.
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b) if any(corpus) else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        q = analyze(query, self.stem, self.remove_stopwords)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:k] if s > 0]
