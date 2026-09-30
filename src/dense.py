"""Dense embedding stage: CPU encoder with a content-hash-keyed embedding cache.

Every embedded text is stored in SQLite under sha256(encoder key + text), where the
encoder key captures everything that changes the vector (model, prefix, max length,
quantization). Re-indexing a new corpus version therefore only embeds texts whose
content changed, and repeated tuning runs never re-embed the same query twice.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Instruction prefixes the models were trained with (queries only unless noted).
MODEL_PROMPTS: Dict[str, Dict[str, str]] = {
    "nomic-ai/CodeRankEmbed": {
        "query": "Represent this query for searching relevant code: ",
        "document": "",
    },
    "jinaai/jina-code-embeddings-0.5b": {
        "query": "Find the most relevant code snippet given the following query:\n",
        "document": "Candidate code snippet:\n",
    },
}

DEFAULT_CACHE_PATH = os.path.join(".cache", "embeddings.sqlite")


class EmbeddingStore:
    """SQLite map from content hash to a float32 embedding vector."""

    def __init__(self, db_path: str = DEFAULT_CACHE_PATH) -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.conn.commit()

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        found: Dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 900):  # SQLite variable limit
            chunk = list(keys[start : start + 900])
            marks = ",".join("?" * len(chunk))
            for key, blob in self.conn.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({marks})", chunk
            ):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: Dict[str, np.ndarray]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
        )
        self.conn.commit()


class DenseEncoder:
    """SentenceTransformer on CPU with model prompts, length caps, int8 option and caching.

    Args:
        model_name: HuggingFace model id.
        max_seq_length: Token cap for documents.
        query_max_tokens: Token cap for queries (APPS queries are long problem statements).
        quantize: "none" or "int8" (dynamic int8 quantization of all Linear layers).
        store: Embedding cache; pass None to disable caching.
        save_every: Cache misses are embedded and persisted in blocks of this size.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/CodeRankEmbed",
        max_seq_length: int = 512,
        query_max_tokens: int = 512,
        quantize: str = "none",
        store: EmbeddingStore | None = None,
        batch_size: int = 32,
        save_every: int = 512,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.query_max_tokens = query_max_tokens
        self.quantize = quantize
        self.batch_size = batch_size
        self.save_every = save_every
        self.prompts = MODEL_PROMPTS.get(model_name, {"query": "", "document": ""})
        self.store = store
        self.last_misses = 0

        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        if quantize == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif quantize != "none":
            raise ValueError(f"unknown quantize mode {quantize!r}")
        self.model.eval()
        self.dim = int(self.model.get_sentence_embedding_dimension() or 0)

    def _key_prefix(self, kind: str, max_len: int) -> str:
        return f"{self.model_name}|{kind}|{max_len}|{self.quantize}|{self.prompts[kind]}|"

    def _encode_raw(self, texts: List[str], max_len: int, show_progress: bool) -> np.ndarray:
        # sentence-transformers sorts inputs by length before batching, so padding waste
        # stays low even though APPS queries and solutions vary widely in length.
        previous = self.model.max_seq_length
        self.model.max_seq_length = max_len
        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)
        finally:
            self.model.max_seq_length = previous

    def encode(self, texts: Sequence[str], kind: str, show_progress: bool = False) -> np.ndarray:
        """Encode `texts` as "query" or "document", embedding only cache misses."""
        max_len = self.query_max_tokens if kind == "query" else self.max_seq_length
        prompted = [self.prompts[kind] + t for t in texts]
        if not prompted:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.store is None:
            return self._encode_raw(prompted, max_len, show_progress)

        prefix = self._key_prefix(kind, max_len)
        keys = [hashlib.sha256((prefix + t).encode("utf-8")).hexdigest() for t in texts]
        cached = self.store.get_many(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(k for k in keys if k not in cached))
        if missing:
            first = {k: i for i, k in reversed(list(enumerate(keys)))}
            # Longest texts first so the slowest blocks surface early in progress output.
            missing.sort(key=lambda k: len(prompted[first[k]]), reverse=True)
            start = time.perf_counter()
            for block in range(0, len(missing), self.save_every):
                block_keys = missing[block : block + self.save_every]
                vectors = self._encode_raw(
                    [prompted[first[k]] for k in block_keys], max_len, show_progress
                )
                new = dict(zip(block_keys, vectors))
                self.store.put_many(new)  # persist per block so interruptions keep progress
                cached.update(new)
            logger.info(
                "Embedded %d %s texts in %.1fs (%d cached)",
                len(missing), kind, time.perf_counter() - start, len(set(keys)) - len(missing),
            )
        self.last_misses = len(missing)
        return np.stack([cached[k] for k in keys]).astype(np.float32)
