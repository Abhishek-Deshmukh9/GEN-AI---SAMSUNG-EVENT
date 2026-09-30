"""Interface contracts between the retrieval roles, plus stubs so no role blocks another.

Role 1 (query understanding)   -> QueryExpander:   query text -> StructuredQuery
Role 2 (corpus processing)     -> SnippetProvider: document   -> List[SnippetSchema]
Role 3 (core retriever)        -> src/retriever.py HybridRetriever consumes all three
Role 4 (reranking)             -> Reranker:        query + top-k candidates -> reordered candidates

Each contract is a `typing.Protocol`; the real implementations already in the repo
(`QueryParser`, `ChunkCache`) satisfy them structurally, and the stubs below are drop-in
defaults that keep the pipeline runnable while a teammate's module is unfinished.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, runtime_checkable

from src.chunker import SnippetMetadata, SnippetSchema
from src.preprocessor import StructuredQuery


@dataclass
class Candidate:
    """One retrieved document handed from first-stage retrieval to the reranker."""

    doc_id: str
    text: str
    score: float


@runtime_checkable
class QueryExpander(Protocol):
    """Role 1 contract: categorize and enrich a natural-language query."""

    def parse(self, query: str) -> StructuredQuery: ...


@runtime_checkable
class SnippetProvider(Protocol):
    """Role 2 contract: turn one corpus document into embeddable snippets."""

    def get_or_compute(self, source: str, doc_id: str) -> List[SnippetSchema]: ...


@runtime_checkable
class Reranker(Protocol):
    """Role 4 contract: reorder first-stage candidates for one query.

    Must return the same doc ids it was given, best first, with new scores.
    """

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> List[Candidate]: ...


class StubQueryExpander:
    """Passes the query through unchanged."""

    def parse(self, query: str) -> StructuredQuery:
        return StructuredQuery(
            raw_query=query,
            intent="behavior-lookup",
            intent_confidence=0.0,
            processed_query=query,
        )


class StubSnippetProvider:
    """Treats every document as a single snippet whose summary is the full text."""

    def get_or_compute(self, source: str, doc_id: str) -> List[SnippetSchema]:
        return [
            SnippetSchema(
                id=doc_id,
                doc_id=doc_id,
                summary_view=source,
                full_view=source,
                metadata=SnippetMetadata(chunk_index=0, num_chunks=1),
            )
        ]


class IdentityReranker:
    """Keeps first-stage order; the default until Role 4's reranker is plugged in."""

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> List[Candidate]:
        return list(candidates)


class CrossEncoderReranker:
    """Reference Role 4 implementation using a sentence-transformers CrossEncoder on CPU."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> List[Candidate]:
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, c.text) for c in candidates], batch_size=self.batch_size
        )
        rescored = [Candidate(c.doc_id, c.text, float(s)) for c, s in zip(candidates, scores)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)
