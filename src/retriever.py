"""Role 3: hybrid code retriever implementing MTEB's SearchProtocol.

Pipeline per query:
  1. Role 1 QueryParser categorizes the query (and optionally supplies expanded terms).
  2. Dense pass: CodeRankEmbed embeddings, exact dot product over the whole corpus
     (optionally over Role 2 chunks, max-pooled back to documents).
  3. Optional pseudo-relevance feedback: Rocchio update of the query vector with the
     top dense hits, then a second dense pass.
  4. Sparse pass: BM25 over a code-aware tokenizer.
  5. Weighted reciprocal-rank fusion of the dense and sparse rankings.
  6. Optional Role 4 reranker over the fused top-k.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.dense import DenseEncoder, EmbeddingStore
from src.interfaces import Candidate, IdentityReranker, QueryExpander, Reranker, SnippetProvider
from src.preprocessor import QueryParser
from src.sparse import BM25Index

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Every knob tuned on the dev split; serialized next to each result file."""

    model_name: str = "nomic-ai/CodeRankEmbed"
    max_seq_length: int = 512
    query_max_tokens: int = 512
    quantize: str = "none"
    use_dense: bool = True
    use_bm25: bool = True
    w_dense: float = 1.0
    w_bm25: float = 0.3
    rrf_k: int = 60
    prf_docs: int = 0
    prf_beta: float = 0.0
    role1_bm25_terms: bool = False
    role1_dense_query: bool = False
    doc_view: str = "raw"  # "raw" | "chunks" (Role 2 summary views, max-pooled)
    rerank_top_k: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "RetrieverConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every column in each row (rank 1 = highest score)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


class HybridRetriever:
    """Dense + BM25 retriever with RRF fusion, usable standalone or via `mteb.evaluate`."""

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        expander: QueryExpander | None = None,
        snippets: SnippetProvider | None = None,
        reranker: Reranker | None = None,
        store: EmbeddingStore | None = None,
        encoder: DenseEncoder | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self.expander = expander or QueryParser()
        self.snippets = snippets
        self.reranker = reranker or IdentityReranker()
        self.store = store if store is not None else EmbeddingStore()
        self.encoder = encoder
        if self.encoder is None and self.config.use_dense:
            self.encoder = DenseEncoder(
                self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                query_max_tokens=self.config.query_max_tokens,
                quantize=self.config.quantize,
                store=self.store,
            )
        self.bm25 = BM25Index()
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.unit_matrix = np.zeros((0, 0), dtype=np.float32)
        self.unit_to_doc = np.zeros(0, dtype=np.int64)
        self.last_index_stats: Dict[str, float] = {}
        self._mteb_meta = None

    # ------------------------------------------------------------------ indexing
    def _units(self) -> Tuple[List[str], np.ndarray]:
        """Texts to embed and their parent document index."""
        if self.config.doc_view == "raw" or self.snippets is None:
            return self.doc_texts, np.arange(len(self.doc_texts))
        texts: List[str] = []
        owners: List[int] = []
        for i, (doc_id, text) in enumerate(zip(self.doc_ids, self.doc_texts)):
            for snippet in self.snippets.get_or_compute(text, doc_id):
                texts.append(snippet.summary_view)
                owners.append(i)
        return texts, np.asarray(owners)

    def index_corpus(self, doc_ids: Sequence[str], texts: Sequence[str]) -> Dict[str, float]:
        """(Re)build all indexes. Only content not already in the embedding cache is embedded."""
        start = time.perf_counter()
        self.doc_ids = list(doc_ids)
        self.doc_texts = [t or "" for t in texts]
        embedded = 0
        units = len(self.doc_texts)
        if self.config.use_dense and self.encoder is not None:
            unit_texts, self.unit_to_doc = self._units()
            units = len(unit_texts)
            self.unit_matrix = self.encoder.encode(unit_texts, "document", show_progress=True)
            embedded = self.encoder.last_misses
        dense_time = time.perf_counter() - start
        if self.config.use_bm25:
            self.bm25.build(self.doc_texts)
        self.last_index_stats = {
            "docs": len(self.doc_ids),
            "units": units,
            "embedded": embedded,
            "reused": units - embedded,
            "dense_seconds": round(dense_time, 2),
            "total_seconds": round(time.perf_counter() - start, 2),
        }
        logger.info("Index built: %s", self.last_index_stats)
        return self.last_index_stats

    # ------------------------------------------------------------------ scoring
    def _dense_scores(self, q: np.ndarray) -> np.ndarray:
        unit_scores = q @ self.unit_matrix.T
        if len(self.unit_to_doc) == unit_scores.shape[1] and self.config.doc_view == "chunks":
            doc_scores = np.full((q.shape[0], len(self.doc_ids)), -np.inf, dtype=np.float32)
            for row in range(q.shape[0]):
                np.maximum.at(doc_scores[row], self.unit_to_doc, unit_scores[row])
            return doc_scores
        return unit_scores

    def score_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Fused relevance matrix of shape (num_queries, num_docs); higher is better."""
        cfg = self.config
        parsed = [self.expander.parse(q) for q in queries]
        fused = np.zeros((len(queries), len(self.doc_ids)), dtype=np.float32)

        if cfg.use_dense:
            dense_inputs = [p.processed_query if cfg.role1_dense_query else q for p, q in zip(parsed, queries)]
            q = self.encoder.encode(dense_inputs, "query", show_progress=len(queries) > 64)
            dense = self._dense_scores(q)
            if cfg.prf_docs > 0 and cfg.prf_beta > 0:
                top = np.argsort(-dense, axis=1)[:, : cfg.prf_docs]
                doc_vecs = self._doc_vectors(top)
                q = q + cfg.prf_beta * doc_vecs.mean(axis=1)
                q /= np.linalg.norm(q, axis=1, keepdims=True)
                dense = self._dense_scores(q)
            if not cfg.use_bm25:
                return dense
            fused += cfg.w_dense / (cfg.rrf_k + _ranks(dense))

        if cfg.use_bm25:
            sparse = np.stack(
                [
                    self.bm25.score(q, p.expanded_identifiers if cfg.role1_bm25_terms else ())
                    for p, q in zip(parsed, queries)
                ]
            )
            if not cfg.use_dense:
                return sparse
            fused += cfg.w_bm25 / (cfg.rrf_k + _ranks(sparse))
        return fused

    def _doc_vectors(self, doc_idx: np.ndarray) -> np.ndarray:
        """Document-level vectors for PRF (first unit of each document when chunked)."""
        if self.config.doc_view != "chunks":
            return self.unit_matrix[doc_idx]
        first_unit = np.full(len(self.doc_ids), -1)
        for u in range(len(self.unit_to_doc) - 1, -1, -1):
            first_unit[self.unit_to_doc[u]] = u
        return self.unit_matrix[first_unit[doc_idx]]

    def retrieve(self, queries: Sequence[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Ranked (doc_id, score) lists, with the Role 4 reranker applied to the top-k."""
        if len(queries) > 512:
            if self.config.use_dense:  # embed all queries in one length-sorted pass
                self.encoder.encode(list(queries), "query", show_progress=True)
            out: List[List[Tuple[str, float]]] = []
            for start in range(0, len(queries), 512):
                out.extend(self.retrieve(queries[start : start + 512], top_k))
            return out
        scores = self.score_queries(queries)
        depth = min(max(top_k, self.config.rerank_top_k), scores.shape[1])
        top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
        results: List[List[Tuple[str, float]]] = []
        for row, query in enumerate(queries):
            idx = top[row][np.argsort(-scores[row, top[row]], kind="stable")]
            ranked = [(self.doc_ids[i], float(scores[row, i])) for i in idx]
            k = self.config.rerank_top_k
            if k > 0 and not isinstance(self.reranker, IdentityReranker):
                head = [Candidate(d, self.doc_texts[i], s) for (d, s), i in zip(ranked[:k], idx[:k])]
                reranked = self.reranker.rerank(query, head)
                # Keep reranked head strictly above the untouched tail.
                offset = (ranked[k][1] if len(ranked) > k else 0.0) + 1.0
                n = len(reranked)
                ranked = [(c.doc_id, offset + (n - j)) for j, c in enumerate(reranked)] + ranked[k:]
            results.append(ranked[:top_k])
        return results

    # ------------------------------------------------------------------ MTEB SearchProtocol
    @property
    def mteb_model_meta(self):
        if self._mteb_meta is None:
            from mteb.models.model_meta import ModelMeta

            cfg = self.config
            name = f"HybridRetriever-{cfg.model_name.split('/')[-1]}"
            self._mteb_meta = ModelMeta.create_empty(
                overwrites=dict(
                    name=f"code-intel/{name}",
                    revision="1.0.0",
                    languages=["eng", "python"],
                    framework=["Sentence Transformers", "PyTorch"],
                    modalities=["text"],
                    embed_dim=self.encoder.dim if self.encoder else None,
                )
            )
        return self._mteb_meta

    def index(self, corpus, *, task_metadata=None, hf_split="test", hf_subset="default",
              encode_kwargs=None, num_proc=None) -> None:
        texts = []
        for row in corpus:
            title = (row.get("title") or "").strip()
            text = row.get("text") or ""
            texts.append(f"{title}\n{text}" if title else text)
        self.index_corpus(corpus["id"], texts)

    def search(self, queries, *, task_metadata=None, hf_split="test", hf_subset="default",
               top_k: int = 100, encode_kwargs=None, top_ranked=None, num_proc=None):
        query_ids = list(queries["id"])
        texts = [t if isinstance(t, str) else " ".join(t) for t in queries["text"]]
        ranked = self.retrieve(texts, top_k=min(top_k, len(self.doc_ids)))
        return {qid: dict(r) for qid, r in zip(query_ids, ranked)}
