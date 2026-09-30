"""Step 2: greedy tuning + ablation on the locked dev split (never touches test labels).

Each stage keeps a change only if dev NDCG@10 improves:
  BM25 floor -> dense (query cap 256 vs 512) -> weighted RRF grid -> PRF grid
  -> Role 1 expanded terms (BM25 side, dense side) -> Role 4 reranker (dev slice).

Writes results/ablation_dev.{json,md} and configs/best.json.

    python scripts/tune_dev.py [--skip-rerank]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import load_apps, make_dev_split, score_run
from src.retriever import HybridRetriever, RetrieverConfig, _ranks

logging.basicConfig(level=logging.WARNING)


def ranked_from_scores(scores: np.ndarray, doc_ids: List[str], k: int = 100):
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = []
    for row in range(scores.shape[0]):
        idx = top[row][np.argsort(-scores[row, top[row]], kind="stable")]
        out.append([(doc_ids[i], float(scores[row, i])) for i in idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--rerank-queries", type=int, default=100)
    ap.add_argument("--rerank-top-k", type=int, default=10)
    args = ap.parse_args()

    data = load_apps()
    dev = make_dev_split(data)
    queries = [data.queries[q] for q in dev]
    qrels = data.qrels["train"]
    rows: List[Dict] = []

    def record(stage: str, setting: str, metrics: Dict, kept: bool = False, **extra) -> Dict:
        row = {"stage": stage, "setting": setting, **metrics, "kept": kept, **extra}
        rows.append(row)
        print(f"[{stage:>9}] {setting:<46} ndcg@10={metrics['ndcg@10']:.4f} "
              f"mrr@10={metrics['mrr@10']:.4f} r@100={metrics['recall@100']:.3f}"
              + ("  <- kept" if kept else ""), flush=True)
        return row

    retr = HybridRetriever(RetrieverConfig())
    retr.index_corpus(data.doc_ids, data.doc_texts)
    cfg = retr.config
    parsed = [retr.expander.parse(q) for q in queries]

    # BM25 floor ---------------------------------------------------------------
    t = time.perf_counter()
    sparse = np.stack([retr.bm25.score(q) for q in queries])
    bm25_ms = (time.perf_counter() - t) * 1000 / len(queries)
    floor = score_run(ranked_from_scores(sparse, retr.doc_ids), dev, qrels)
    record("floor", "BM25 (code-aware tokenizer)", floor, query_ms=round(bm25_ms, 2))

    # Dense + query length cap -------------------------------------------------
    dense_by_cap = {}
    for cap in (256, 512):
        retr.encoder.query_max_tokens = cap
        t = time.perf_counter()
        q = retr.encoder.encode(queries, "query", show_progress=True)
        dense_by_cap[cap] = (q, retr._dense_scores(q))
        dense_by_cap[cap] += (score_run(ranked_from_scores(dense_by_cap[cap][1], retr.doc_ids), dev, qrels),)
    best_cap = max(dense_by_cap, key=lambda c: dense_by_cap[c][2]["ndcg@10"])
    for cap in (256, 512):
        record("dense", f"CodeRankEmbed, query cap {cap} tokens", dense_by_cap[cap][2], kept=cap == best_cap)
    cfg.query_max_tokens = best_cap
    retr.encoder.query_max_tokens = best_cap
    qvec, dense, best = dense_by_cap[best_cap]

    # Weighted RRF -------------------------------------------------------------
    dense_ranks = _ranks(dense)
    sparse_ranks = _ranks(sparse)
    grid = []
    for w, k in itertools.product((0.1, 0.2, 0.3, 0.5, 0.75, 1.0), (10, 30, 60, 100)):
        fused = 1.0 / (k + dense_ranks) + w / (k + sparse_ranks)
        grid.append((score_run(ranked_from_scores(fused, retr.doc_ids), dev, qrels), w, k))
    m, w, k = max(grid, key=lambda g: g[0]["ndcg@10"])
    for gm, gw, gk in sorted(grid, key=lambda g: -g[0]["ndcg@10"])[:5]:
        record("rrf", f"w_bm25={gw}, k={gk}", gm, kept=(gw, gk) == (w, k) and m["ndcg@10"] > best["ndcg@10"])
    if m["ndcg@10"] > best["ndcg@10"]:
        best, cfg.w_bm25, cfg.rrf_k, cfg.use_bm25 = m, w, k, True
    else:
        cfg.use_bm25 = False

    def evaluate_current(q_vecs: np.ndarray, extra_terms: bool = False) -> Dict:
        d = retr._dense_scores(q_vecs)
        if cfg.prf_docs and cfg.prf_beta:
            top = np.argsort(-d, axis=1)[:, : cfg.prf_docs]
            q2 = q_vecs + cfg.prf_beta * retr._doc_vectors(top).mean(axis=1)
            q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
            d = retr._dense_scores(q2)
        if not cfg.use_bm25:
            return score_run(ranked_from_scores(d, retr.doc_ids), dev, qrels)
        s = sparse if not extra_terms else np.stack(
            [retr.bm25.score(qt, p.expanded_identifiers) for qt, p in zip(queries, parsed)]
        )
        fused = cfg.w_dense / (cfg.rrf_k + _ranks(d)) + cfg.w_bm25 / (cfg.rrf_k + _ranks(s))
        return score_run(ranked_from_scores(fused, retr.doc_ids), dev, qrels)

    # Pseudo-relevance feedback ------------------------------------------------
    prf_grid = []
    for n, beta in itertools.product((1, 3, 5), (0.2, 0.4, 0.7)):
        cfg.prf_docs, cfg.prf_beta = n, beta
        prf_grid.append((evaluate_current(qvec), n, beta))
    m, n, beta = max(prf_grid, key=lambda g: g[0]["ndcg@10"])
    for gm, gn, gb in sorted(prf_grid, key=lambda g: -g[0]["ndcg@10"])[:3]:
        record("prf", f"Rocchio top-{gn}, beta={gb}", gm, kept=(gn, gb) == (n, beta) and m["ndcg@10"] > best["ndcg@10"])
    if m["ndcg@10"] > best["ndcg@10"]:
        best, cfg.prf_docs, cfg.prf_beta = m, n, beta
    else:
        cfg.prf_docs, cfg.prf_beta = 0, 0.0

    # Role 1: expanded identifiers ---------------------------------------------
    if cfg.use_bm25:
        m = evaluate_current(qvec, extra_terms=True)
        keep = m["ndcg@10"] > best["ndcg@10"]
        record("role1", "expanded identifiers added to BM25 query", m, kept=keep)
        if keep:
            best, cfg.role1_bm25_terms = m, True
    enriched = retr.encoder.encode([p.processed_query for p in parsed], "query", show_progress=True)
    m = evaluate_current(enriched, extra_terms=cfg.role1_bm25_terms)
    keep = m["ndcg@10"] > best["ndcg@10"]
    record("role1", "QueryParser enriched text as dense query", m, kept=keep)
    if keep:
        best, cfg.role1_dense_query, qvec = m, True, enriched

    record("final", "best dev configuration", best, kept=True)

    # Role 4 reranker on a dev slice (cross-encoder cost is the constraint on CPU) ---
    if not args.skip_rerank:
        from src.interfaces import CrossEncoderReranker

        n_q = args.rerank_queries
        sub_q, sub_ids = queries[:n_q], dev[:n_q]
        base = HybridRetriever(replace(cfg, rerank_top_k=0), store=retr.store, encoder=retr.encoder)
        base.index_corpus(data.doc_ids, data.doc_texts)
        base_run = base.retrieve(sub_q, top_k=100)
        record("rerank", f"no reranker ({n_q}-query slice)", score_run(base_run, sub_ids, qrels))
        base.reranker = CrossEncoderReranker()
        base.config.rerank_top_k = args.rerank_top_k
        t = time.perf_counter()
        rr_run = base.retrieve(sub_q, top_k=100)
        ms = (time.perf_counter() - t) * 1000 / n_q
        m = score_run(rr_run, sub_ids, qrels)
        record("rerank", f"ms-marco MiniLM cross-encoder, top-{args.rerank_top_k}", m,
               kept=False, query_ms=round(ms, 1))

    os.makedirs("results", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    cfg.save("configs/best.json")
    with open("results/ablation_dev.json", "w", encoding="utf-8") as f:
        json.dump({"dev_size": len(dev), "rows": rows, "best_config": asdict(cfg)}, f, indent=2)
    with open("results/ablation_dev.md", "w", encoding="utf-8") as f:
        f.write(f"| Stage | Setting | NDCG@10 | MRR@10 | Recall@100 | Kept |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['stage']} | {r['setting']} | {r['ndcg@10']:.4f} | {r['mrr@10']:.4f} | "
                    f"{r['recall@100']:.3f} | {'yes' if r['kept'] else ''} |\n")
    print("\nBest config:", json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()
