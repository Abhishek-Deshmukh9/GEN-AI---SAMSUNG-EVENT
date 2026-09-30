"""Sparse stage: BM25 (bm25s) over a code-aware tokenizer.

The tokenizer is shared by queries and documents so natural-language words in a problem
statement can meet identifiers in code: `maxSubarraySum` and `max_subarray_sum` both
yield `max`, `subarray`, `sum` (plus the joined identifier itself).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a an the and or of to in on at for with by from as is are was were be been being
    it its this that these those there their then than so such if else elif not no nor
    do does did done can could should would will shall may might must have has had
    you your we our they them he she his her i me my which who whom what when where why
    how all any each every some one also only just into over under about after before
    between during out up down very more most other same both few own again further once
    here while self def return print input int str range len list map split strip
    import from pass none true false""".split()
)


def code_tokenize(text: str) -> List[str]:
    """Lowercased identifier sub-tokens plus the full identifier, minus stopwords."""
    tokens: List[str] = []
    for ident in _IDENT.findall(text):
        parts = [p.lower() for piece in ident.split("_") for p in _CAMEL.findall(piece)]
        whole = ident.lower().strip("_")
        if len(parts) > 1 and whole not in STOPWORDS:
            tokens.append(whole)
        tokens.extend(p for p in parts if len(p) > 1 and p not in STOPWORDS)
    return tokens


class BM25Index:
    """bm25s index over `code_tokenize`; returns dense score vectors over the corpus."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.num_docs = 0

    def build(self, texts: Sequence[str]) -> None:
        import bm25s

        corpus_tokens = [code_tokenize(t) or ["<empty>"] for t in texts]
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(corpus_tokens, show_progress=False)
        self.num_docs = len(texts)

    def score(self, query: str, extra_terms: Sequence[str] = ()) -> np.ndarray:
        """BM25 score of every document for one query (zeros when nothing matches)."""
        if self.retriever is None:
            raise ValueError("BM25 index not built")
        tokens = code_tokenize(query)
        for term in extra_terms:
            tokens.extend(code_tokenize(term))
        if not tokens:
            return np.zeros(self.num_docs, dtype=np.float32)
        return np.asarray(self.retriever.get_scores(tokens), dtype=np.float32)

    def stats(self) -> Dict[str, int]:
        return {"docs": self.num_docs}
