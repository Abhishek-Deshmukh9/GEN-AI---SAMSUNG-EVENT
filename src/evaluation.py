"""Dataset loading, the locked dev split, and NDCG@10 / MRR@10 for offline tuning.

CoIR Apps: 8,765 corpus solutions (5,000 train + 3,765 test), one relevant solution per
query. The dev split is a fixed sample of *train* queries searched against the *full*
corpus, so it mirrors test conditions without ever touching test labels.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

DATASET = "CoIR-Retrieval/apps"
DEV_SPLIT_PATH = os.path.join("splits", "dev_qids.json")


@dataclass
class AppsData:
    doc_ids: List[str]
    doc_texts: List[str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, Dict[str, int]]]  # split -> qid -> {doc_id: score}


def load_apps() -> AppsData:
    from datasets import load_dataset

    corpus = load_dataset(DATASET, "corpus")["corpus"]
    queries = load_dataset(DATASET, "queries")["queries"]
    qrels_ds = load_dataset(DATASET, "default")
    qrels: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split in qrels_ds:
        table: Dict[str, Dict[str, int]] = {}
        for row in qrels_ds[split]:
            table.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        qrels[split] = table
    return AppsData(
        doc_ids=list(corpus["_id"]),
        doc_texts=list(corpus["text"]),
        queries=dict(zip(queries["_id"], queries["text"])),
        qrels=qrels,
    )


def is_test_style(query: str) -> bool:
    """Stdin/stdout problem statements (97% of test, only 24% of train)."""
    return "-----Input-----" in query


def make_dev_split(data: AppsData, size: int = 500, seed: int = 13) -> List[str]:
    """Sample `size` test-style train query ids once and freeze them in splits/dev_qids.json.

    Train is dominated by function-signature (Codewars-style) problems whose examples
    name the target function, which BM25 matches trivially; test is almost entirely
    stdin/stdout problems. Sampling only test-style train queries keeps dev faithful.
    """
    if os.path.exists(DEV_SPLIT_PATH):
        with open(DEV_SPLIT_PATH, encoding="utf-8") as f:
            return json.load(f)["qids"]
    qids = sorted(
        (q for q in data.qrels["train"] if is_test_style(data.queries[q])),
        key=lambda q: int(q.lstrip("q")),
    )
    sample = sorted(random.Random(seed).sample(qids, size), key=lambda q: int(q.lstrip("q")))
    os.makedirs(os.path.dirname(DEV_SPLIT_PATH), exist_ok=True)
    with open(DEV_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"source": "train", "filter": "-----Input----- (test-style)", "seed": seed,
             "size": size, "qids": sample},
            f, indent=1,
        )
    return sample


def score_run(
    ranked: Sequence[Sequence[Tuple[str, float]]],
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """NDCG@k, MRR@k and Recall@k/100 with binary relevance (MTEB-compatible definitions)."""
    ndcg = mrr = rec_k = rec_100 = 0.0
    for qid, run in zip(qids, ranked):
        relevant = {d for d, s in qrels[qid].items() if s > 0}
        ids = [d for d, _ in run]
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ids[:k]) if d in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg += dcg / idcg if idcg else 0.0
        mrr += next((1 / (i + 1) for i, d in enumerate(ids[:k]) if d in relevant), 0.0)
        rec_k += len(relevant & set(ids[:k])) / len(relevant)
        rec_100 += len(relevant & set(ids[:100])) / len(relevant)
    n = max(len(qids), 1)
    return {
        f"ndcg@{k}": round(ndcg / n, 5),
        f"mrr@{k}": round(mrr / n, 5),
        f"recall@{k}": round(rec_k / n, 5),
        "recall@100": round(rec_100 / n, 5),
    }
