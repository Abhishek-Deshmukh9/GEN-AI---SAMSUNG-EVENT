"""P1 + Bonus: retrieval over multiple versions of a codebase.

P1 (retrieval on a given version): each version gets its own HybridRetriever, but all of
them share one encoder and one content-hash embedding store, so indexing a new version
embeds only snippets whose content changed; BM25 is rebuilt from scratch (sub-second).

Bonus (evolutionary retrieval across all versions): one index over the *unique contents*
seen in any version. Identical content across versions is embedded and scored once, then
results are collapsed per logical snippet (doc id): the best-scoring variant wins, and
near-ties (within a relative `tie_eps`) go to the newest version, so near-duplicates never crowd the
top-k. Each hit lists every version it appears in and which versions changed it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.dense import DenseEncoder, EmbeddingStore
from src.retriever import HybridRetriever, RetrieverConfig


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class VersionDiff:
    version: str
    added: int
    modified: int
    removed: int
    unchanged: int
    embedded: int
    reused: int
    seconds: float


@dataclass
class EvolutionHit:
    doc_id: str
    score: float
    version: str
    text: str
    versions_present: List[str] = field(default_factory=list)
    versions_changed: List[str] = field(default_factory=list)


class VersionedCodeIndex:
    """Keeps every version of a corpus and answers per-version or cross-version queries."""

    def __init__(self, config: RetrieverConfig | None = None, store: EmbeddingStore | None = None,
                 tie_eps: float = 0.01) -> None:
        self.config = config or RetrieverConfig()
        self.store = store if store is not None else EmbeddingStore()
        self.encoder = (
            DenseEncoder(self.config.model_name, self.config.max_seq_length,
                         self.config.query_max_tokens, self.config.quantize, store=self.store)
            if self.config.use_dense else None
        )
        self.tie_eps = tie_eps
        self.order: List[str] = []
        self.versions: Dict[str, Dict[str, str]] = {}
        self.retrievers: Dict[str, HybridRetriever] = {}
        self._all: Optional[HybridRetriever] = None
        self._owners: Dict[str, List[Tuple[str, str]]] = {}

    def _new_retriever(self, config: RetrieverConfig) -> HybridRetriever:
        return HybridRetriever(config, store=self.store, encoder=self.encoder)

    def add_version(self, name: str, docs: Dict[str, str]) -> VersionDiff:
        """Index a new version; only changed or new content is embedded."""
        start = time.perf_counter()
        prev = self.versions[self.order[-1]] if self.order else {}
        added = sum(1 for d in docs if d not in prev)
        modified = sum(1 for d in docs if d in prev and prev[d] != docs[d])
        removed = sum(1 for d in prev if d not in docs)
        retr = self._new_retriever(self.config)
        stats = retr.index_corpus(list(docs), list(docs.values()))
        self.versions[name] = dict(docs)
        self.order.append(name)
        self.retrievers[name] = retr
        self._all = None  # cross-version index is rebuilt lazily
        return VersionDiff(name, added, modified, removed, len(docs) - added - modified,
                           int(stats["embedded"]), int(stats["reused"]),
                           round(time.perf_counter() - start, 2))

    def search(self, query: str, version: str | None = None, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieval against one version (default: newest)."""
        name = version or self.order[-1]
        return self.retrievers[name].retrieve([query], top_k=k)[0]

    def _build_all(self) -> HybridRetriever:
        owners: Dict[str, List[Tuple[str, str]]] = {}
        texts: Dict[str, str] = {}
        for name in self.order:
            for doc_id, text in self.versions[name].items():
                h = _h(text)
                owners.setdefault(h, []).append((name, doc_id))
                texts[h] = text
        retr = self._new_retriever(self.config)
        retr.index_corpus(list(texts), list(texts.values()))
        self._owners = owners
        return retr

    def search_all(self, query: str, k: int = 10, depth: int = 200) -> List[EvolutionHit]:
        """Evolutionary retrieval across every version, collapsed per logical snippet."""
        if self._all is None:
            self._all = self._build_all()
        rank = {name: i for i, name in enumerate(self.order)}
        best: Dict[str, EvolutionHit] = {}
        for content_hash, score in self._all.retrieve([query], top_k=depth)[0]:
            for version, doc_id in self._owners[content_hash]:
                hit = best.get(doc_id)
                newer = hit is not None and rank[version] > rank[hit.version]
                tie = hit is not None and abs(score - hit.score) <= self.tie_eps * max(abs(score), abs(hit.score))
                if hit is None or (not tie and score > hit.score) or (tie and newer):
                    best[doc_id] = EvolutionHit(doc_id, score, version, self.versions[version][doc_id])
        hits = sorted(best.values(), key=lambda h: (-h.score, -rank[h.version]))[:k]
        for hit in hits:
            present = [v for v in self.order if hit.doc_id in self.versions[v]]
            hit.versions_present = present
            hit.versions_changed = [
                v for prev, v in zip(present, present[1:])
                if self.versions[prev][hit.doc_id] != self.versions[v][hit.doc_id]
            ]
        return hits
