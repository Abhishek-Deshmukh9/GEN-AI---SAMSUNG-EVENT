"""Step 1: lock the dev split, record the BM25 floor, and warm the embedding cache.

Embeds the full corpus once plus the dev queries at each query-length cap under test.
Every later tuning step and the test run then reuse these cached vectors.

    python scripts/prepare.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import load_apps, make_dev_split, score_run
from src.retriever import HybridRetriever, RetrieverConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    data = load_apps()
    dev = make_dev_split(data)
    dev_queries = [data.queries[q] for q in dev]
    print(f"Dev split: {len(dev)} train queries, corpus {len(data.doc_ids)} docs")

    os.makedirs("results", exist_ok=True)
    bm25 = HybridRetriever(RetrieverConfig(use_dense=False, use_bm25=True))
    bm25.index_corpus(data.doc_ids, data.doc_texts)
    t = time.perf_counter()
    floor = score_run(bm25.retrieve(dev_queries, top_k=100), dev, data.qrels["train"])
    floor["query_ms"] = round((time.perf_counter() - t) * 1000 / len(dev), 2)
    print("BM25 floor (dev):", floor)
    with open("results/bm25_floor_dev.json", "w", encoding="utf-8") as f:
        json.dump(floor, f, indent=2)

    dense = HybridRetriever(RetrieverConfig(use_bm25=False))
    stats = dense.index_corpus(data.doc_ids, data.doc_texts)
    print("Corpus embedded:", stats)
    for cap in (256, 512):
        dense.encoder.query_max_tokens = cap
        t = time.perf_counter()
        dense.encoder.encode(dev_queries, "query", show_progress=True)
        print(f"Dev queries @ {cap} tokens embedded in {time.perf_counter() - t:.0f}s")


if __name__ == "__main__":
    main()
