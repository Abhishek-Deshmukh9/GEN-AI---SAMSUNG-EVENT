"""MTEB AppsRetrieval (CoIR apps, test split) evaluation of the hybrid retriever.

Produces:
  appsretrieval_results.json        - MTEB result file to attach to the GitHub release
  results/appsretrieval_top10.csv   - top-10 ranked documents per test query
  results/predictions/              - MTEB's raw per-query predictions

    python evaluate.py [--config configs/best.json] [--dense-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import mteb

from src.retriever import HybridRetriever, RetrieverConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/best.json")
    ap.add_argument("--dense-only", action="store_true", help="Safety run: CodeRankEmbed only.")
    ap.add_argument("--output", default="appsretrieval_results.json")
    args = ap.parse_args()

    config = RetrieverConfig.load(args.config) if os.path.exists(args.config) else RetrieverConfig()
    if args.dense_only:
        config.use_bm25 = False
        config.prf_docs, config.prf_beta = 0, 0.0
    logger.info("Retriever config: %s", json.dumps(asdict(config)))
    model = HybridRetriever(config)

    task = mteb.get_task("AppsRetrieval")
    pred_dir = Path("results/predictions")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
        cache=None,  # never reuse a stale cached score for a changed config
        prediction_folder=pred_dir,
    )
    task_result = list(result.task_results)[0]
    result_dict = task_result.to_dict()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    os.makedirs("results", exist_ok=True)
    shutil.copy(args.output, "results/" + Path(args.output).stem + ("_dense_only" if args.dense_only else "") + ".json")
    with open("results/test_run_config.json", "w", encoding="utf-8") as f:
        json.dump({"dense_only": args.dense_only, **asdict(config)}, f, indent=2)

    for pred_file in pred_dir.rglob("*.json"):
        preds = json.loads(pred_file.read_text(encoding="utf-8"))
        preds = preds.get("test", preds) if isinstance(preds, dict) else preds
        preds = preds.get("default", preds) if isinstance(preds, dict) else preds
        with open("results/appsretrieval_top10.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "rank", "corpus_id", "score"])
            for qid, docs in preds.items():
                if not isinstance(docs, dict):
                    continue
                top = sorted(docs.items(), key=lambda kv: -kv[1])[:10]
                for rank, (doc_id, score) in enumerate(top, 1):
                    writer.writerow([qid, rank, doc_id, f"{score:.6f}"])
        break

    scores = result_dict["scores"]["test"][0]
    print("\n" + "=" * 60)
    print("MTEB AppsRetrieval (test)")
    print("=" * 60)
    for key in ("ndcg_at_10", "mrr_at_10", "map_at_10", "recall_at_10", "recall_at_100"):
        print(f"{key:>14}: {scores.get(key)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
