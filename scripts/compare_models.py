"""Step 3: one-off dense comparisons on a fixed dev sub-corpus.

Full-corpus runs of a 0.5B model take hours on a laptop CPU, so every variant here is
scored on the same sub-corpus: the gold solutions of the first N dev queries plus a
fixed random sample of distractors. Scores are only comparable within this table.

Variants: CodeRankEmbed fp32 | CodeRankEmbed dynamic int8 | CodeRankEmbed over Role 2
chunk summaries (max-pooled) | jina-code-embeddings-0.5b.

    python scripts/compare_models.py [--queries 150 --distractors 450]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker import ChunkCache
from src.dense import DenseEncoder, EmbeddingStore
from src.evaluation import load_apps, make_dev_split, score_run
from src.retriever import HybridRetriever, RetrieverConfig

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--distractors", type=int, default=450)
    ap.add_argument("--query-cap", type=int, default=256)
    ap.add_argument("--variants", default="cre,int8,chunks,jina")
    args = ap.parse_args()

    data = load_apps()
    dev = make_dev_split(data)[: args.queries]
    qrels = data.qrels["train"]
    gold = {d for q in dev for d in qrels[q]}
    others = [d for d in data.doc_ids if d not in gold]
    sub_ids = sorted(gold | set(random.Random(7).sample(others, args.distractors)),
                     key=lambda d: int(d.lstrip("d")))
    text = dict(zip(data.doc_ids, data.doc_texts))
    sub_texts = [text[d] for d in sub_ids]
    queries = [data.queries[q] for q in dev]
    store = EmbeddingStore()
    print(f"Sub-corpus: {len(dev)} queries, {len(sub_ids)} docs", flush=True)

    variants = {
        "cre": ("CodeRankEmbed fp32 (raw docs)", dict(), None),
        "int8": ("CodeRankEmbed dynamic int8", dict(quantize="int8"), None),
        "chunks": ("CodeRankEmbed over Role 2 chunk summaries", dict(doc_view="chunks"), ChunkCache(os.path.join(".cache", "chunks.sqlite"))),
        "jina": ("jina-code-embeddings-0.5b", dict(model_name="jinaai/jina-code-embeddings-0.5b"), None),
    }
    rows = []
    for key in args.variants.split(","):
        label, overrides, snippets = variants[key]
        cfg = RetrieverConfig(use_bm25=False, query_max_tokens=args.query_cap, **overrides)
        try:
            t0 = time.perf_counter()
            retr = HybridRetriever(cfg, snippets=snippets, store=store)
            load_s = time.perf_counter() - t0
            stats = retr.index_corpus(sub_ids, sub_texts)
            t = time.perf_counter()
            run = retr.retrieve(queries, top_k=100)
            q_s = time.perf_counter() - t
        except Exception as exc:  # keep the other variants running
            print(f"{label}: FAILED ({exc!r})", flush=True)
            rows.append({"variant": label, "error": repr(exc)})
            continue
        m = score_run(run, dev, qrels)
        row = {"variant": label, **m, "units": stats["units"], "docs_embedded": stats["embedded"],
               "doc_embed_s": stats["dense_seconds"], "query_s": round(q_s, 1),
               "params_note": overrides, "model_load_s": round(load_s, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del retr

    os.makedirs("results", exist_ok=True)
    with open("results/model_comparison_dev.json", "w", encoding="utf-8") as f:
        json.dump({"queries": len(dev), "docs": len(sub_ids), "query_cap": args.query_cap,
                   "rows": rows}, f, indent=2)
    with open("results/model_comparison_dev.md", "w", encoding="utf-8") as f:
        f.write(f"Sub-corpus: {len(dev)} dev queries x {len(sub_ids)} docs, query cap {args.query_cap}\n\n")
        f.write("| Variant | NDCG@10 | MRR@10 | Recall@100 | Doc embed s (misses) | Query s |\n|---|---|---|---|---|---|\n")
        for r in rows:
            if "error" in r:
                f.write(f"| {r['variant']} | failed | | | | |\n")
            else:
                f.write(f"| {r['variant']} | {r['ndcg@10']:.4f} | {r['mrr@10']:.4f} | {r['recall@100']:.3f} | "
                        f"{r['doc_embed_s']} ({r['docs_embedded']}) | {r['query_s']} |\n")


if __name__ == "__main__":
    main()
