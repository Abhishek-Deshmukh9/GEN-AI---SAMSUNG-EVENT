"""MTEB AbsEncoder implementation for Agentic Code Intelligence.

Role 1 & Retrieval Baseline: PrePostPipelineEncoder.
Combines:
- QueryParser preprocessing for queries (PromptType.query)
- SentenceTransformer CPU-optimized baseline embedding model (all-MiniLM-L6-v2)
- AbsEncoder conformance for MTEB benchmark execution
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Sequence, Union, cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, PromptType

from src.preprocessor import QueryParser, StructuredQuery

logger = logging.getLogger(__name__)


class PrePostPipelineEncoder(AbsEncoder):
    """MTEB-compatible encoder featuring rule-based query understanding preprocessing.

    Inherits directly from `mteb.models.abs_encoder.AbsEncoder`.
    Routes query embeddings through the Role 1 `QueryParser` to extract intents,
    expand code identifiers, and enrich search representations before dense vector encoding.
    """

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the PrePostPipelineEncoder.

        Args:
            model_name: HuggingFace model repository or local path for the baseline encoder.
            device: Computation device ('cpu' strictly satisfies minimal GPU resource constraints).
            **kwargs: Additional parameters passed to SentenceTransformer.
        """
        super().__init__()
        self.device = device
        self.base_model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        self.query_parser = QueryParser()

        # Build MTEB ModelMeta for evaluation harness registration
        base_meta = ModelMeta.from_sentence_transformer_model(self.model)
        embed_dim = (
            self.model.get_embedding_dimension()
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension()
        )
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(
                name="PrePostPipelineEncoder-all-MiniLM-L6-v2",
                revision=base_meta.revision or "1.0.0",
                embed_dim=embed_dim,
                languages=["eng", "python"],
                framework=["Sentence Transformers", "PyTorch"],
                modalities=["text"],
                similarity_fn_name=base_meta.similarity_fn_name,
                loader=type(self),
            )
        )

    def preprocess_query(self, query: str) -> StructuredQuery:
        """Run Role 1 Query Understanding preprocessor on a query string."""
        return self.query_parser.parse(query)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute similarity matrix between two collections of embeddings."""
        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            return cast(Array, self.model.similarity(embeddings1, embeddings2))
        return super().similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute pairwise similarity vector between corresponding embeddings."""
        if hasattr(self.model, "similarity_pairwise") and callable(self.model.similarity_pairwise):
            return cast(Array, self.model.similarity_pairwise(embeddings1, embeddings2))
        return super().similarity_pairwise(embeddings1, embeddings2)

    def _extract_texts(
        self, inputs: Union[DataLoader[Any], Sequence[Union[str, dict]], Iterable[Any]]
    ) -> List[str]:
        """Extract flat string texts from DataLoader, lists, or dictionary batches."""
        texts: List[str] = []
        if isinstance(inputs, DataLoader):
            for batch in inputs:
                if isinstance(batch, dict) and "text" in batch:
                    texts.extend([str(t) for t in batch["text"]])
                elif isinstance(batch, (list, tuple)):
                    for item in batch:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(str(item["text"]))
                        else:
                            texts.append(str(item))
                elif isinstance(batch, str):
                    texts.append(batch)
                else:
                    texts.extend([str(x) for x in batch])
        elif isinstance(inputs, (list, tuple)):
            for item in inputs:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
                else:
                    texts.append(str(item))
        else:
            for item in inputs:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
                else:
                    texts.append(str(item))
        return texts

    def encode(
        self,
        inputs: DataLoader[Any] | Sequence[str | dict] | Iterable[Any],
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode text inputs with integrated Role 1 query pre-processing.

        When `prompt_type == PromptType.query`, queries are enriched with regex-identified
        intents and expanded identifier casing variants (camelCase, PascalCase, snake_case).
        When `prompt_type` is document/corpus or None, code snippets are encoded directly.

        Args:
            inputs: Batch of text inputs (DataLoader, list of strings, or list of dicts).
            task_metadata: MTEB task metadata.
            hf_split: Dataset split (e.g. 'test').
            hf_subset: Dataset subset name.
            prompt_type: PromptType indicating 'query' or 'document'.
            **kwargs: Additional encode kwargs (batch_size, show_progress_bar, etc.)

        Returns:
            numpy.ndarray of shape (N, embedding_dim).
        """
        texts = self._extract_texts(inputs)
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        # Apply Role 1 Query Understanding preprocessor for queries
        if prompt_type == PromptType.query:
            processed_texts: List[str] = []
            for query_text in texts:
                structured: StructuredQuery = self.query_parser.parse(query_text)
                processed_texts.append(structured.processed_query)
            texts_to_encode = processed_texts
        else:
            texts_to_encode = texts

        # Extract encoding arguments safely
        batch_size = kwargs.get("batch_size", 64)
        show_progress_bar = kwargs.get("show_progress_bar", False)
        normalize_embeddings = kwargs.get("normalize_embeddings", True)

        embeddings = self.model.encode(
            texts_to_encode,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            device=self.device,
        )

        return cast(Array, embeddings)
