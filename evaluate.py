"""Evaluation script for MTEB AppsRetrieval task.

Runs MTEB evaluation using PrePostPipelineEncoder and outputs appsretrieval_results.json.
Measures NDCG@10, MRR, and related retrieval metrics on the CoIR apps test split.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import mteb
from src.encoder import PrePostPipelineEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run MTEB evaluation on AppsRetrieval task and save results."""
    logger.info("Initializing PrePostPipelineEncoder on CPU...")
    model = PrePostPipelineEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    logger.info("Loading MTEB task 'AppsRetrieval'...")
    task = mteb.get_task("AppsRetrieval")

    logger.info("Starting MTEB evaluation (batch_size=64)...")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
        overwrite_strategy="always",
    )

    # Extract task results
    task_results_list = list(result.task_results)
    if not task_results_list:
        raise RuntimeError("Evaluation completed but no task results were produced.")

    task_result = task_results_list[0]
    output_path = Path("appsretrieval_results.json")

    logger.info(f"Writing evaluation results to {output_path.resolve()}...")
    result_dict = task_result.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    logger.info("Evaluation results saved successfully!")

    # Display primary benchmark metrics (NDCG@10, MRR)
    print("\n" + "=" * 60)
    print("MTEB AppsRetrieval Evaluation Summary")
    print("=" * 60)
    scores = result_dict.get("scores", {})
    test_scores = scores.get("test", [])
    if isinstance(test_scores, list) and len(test_scores) > 0:
        metrics = test_scores[0]
        ndcg_10 = metrics.get("ndcg_at_10", metrics.get("ndcg@10", "N/A"))
        mrr_10 = metrics.get("mrr_at_10", metrics.get("mrr@10", "N/A"))
        map_10 = metrics.get("map_at_10", metrics.get("map@10", "N/A"))
        recall_10 = metrics.get("recall_at_10", metrics.get("recall@10", "N/A"))
        print(f"NDCG@10: {ndcg_10}")
        print(f"MRR@10:  {mrr_10}")
        print(f"MAP@10:  {map_10}")
        print(f"Recall@10: {recall_10}")
    else:
        print(f"Scores structure: {list(scores.keys())}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
