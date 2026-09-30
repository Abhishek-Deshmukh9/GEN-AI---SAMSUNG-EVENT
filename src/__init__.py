"""Agentic Code Intelligence - Phase 1 Prototype.

Modular query preprocessing, corpus chunking, and MTEB code retrieval encoder baseline.
"""

from src.preprocessor import IntentType, QueryParser, StructuredQuery
from src.chunker import (
    ChunkCache,
    LongDocChunker,
    SnippetSchema,
    StructuralExtractor,
    aggregate_scores,
)

__all__ = [
    "QueryParser",
    "StructuredQuery",
    "IntentType",
    "StructuralExtractor",
    "SnippetSchema",
    "LongDocChunker",
    "ChunkCache",
    "aggregate_scores",
]

# The encoder pulls in the heavy retrieval stack (torch, mteb, sentence-transformers).
# Keep it optional so the CPU-only analysis roles stay importable on their own.
try:
    from src.encoder import PrePostPipelineEncoder

    __all__.append("PrePostPipelineEncoder")
except ImportError:  # pragma: no cover - depends on optional retrieval deps
    pass
