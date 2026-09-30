"""Unit tests for the hybrid retriever, BM25 tokenizer, metrics, and versioned index.

A deterministic bag-of-tokens fake encoder stands in for CodeRankEmbed, so these tests
run in seconds without downloading a model.
"""

import os
import sys
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dense import EmbeddingStore
from src.evaluation import score_run
from src.interfaces import (
    Candidate,
    IdentityReranker,
    QueryExpander,
    Reranker,
    SnippetProvider,
    StubQueryExpander,
    StubSnippetProvider,
)
from src.chunker import ChunkCache
from src.preprocessor import QueryParser
from src.retriever import HybridRetriever, RetrieverConfig, _ranks
from src.sparse import code_tokenize
from src.versioning import VersionedCodeIndex


class FakeEncoder:
    """Hashes code-aware tokens into a normalized 64-dim bag-of-words vector."""

    dim = 64

    def __init__(self):
        self.query_max_tokens = 512
        self.last_misses = 0
        self.seen = set()

    def encode(self, texts, kind, show_progress=False):
        self.last_misses = sum(1 for t in texts if (kind, t) not in self.seen)
        self.seen.update((kind, t) for t in texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in code_tokenize(text):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


DOCS = {
    "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "sort": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        pass\n    return sorted(arr)",
    "prime": "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, n))",
    "fib": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
}


def make_retriever(**overrides):
    cfg = RetrieverConfig(**overrides)
    return HybridRetriever(cfg, store=EmbeddingStore(":memory:"), encoder=FakeEncoder())


class TestTokenizer(unittest.TestCase):
    def test_splits_identifiers_and_drops_stopwords(self):
        toks = code_tokenize("def maxSubarraySum(arr): return the max_value")
        self.assertIn("maxsubarraysum", toks)
        self.assertIn("subarray", toks)
        self.assertIn("value", toks)
        self.assertNotIn("the", toks)
        self.assertNotIn("return", toks)


class TestMetrics(unittest.TestCase):
    def test_ndcg_and_mrr(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}
        runs = [[("d1", 1.0), ("d2", 0.5)], [("d2", 1.0), ("d9", 0.5)]]
        m = score_run(runs, ["q1", "q2"], qrels)
        self.assertAlmostEqual(m["mrr@10"], (1 + 0.5) / 2, places=4)
        self.assertAlmostEqual(m["ndcg@10"], (1 + 1 / np.log2(3)) / 2, places=4)

    def test_ranks(self):
        np.testing.assert_array_equal(_ranks(np.array([[0.1, 0.9, 0.5]])), [[3, 1, 2]])


class TestContracts(unittest.TestCase):
    def test_real_modules_and_stubs_satisfy_contracts(self):
        self.assertIsInstance(QueryParser(), QueryExpander)
        self.assertIsInstance(StubQueryExpander(), QueryExpander)
        self.assertIsInstance(ChunkCache(":memory:"), SnippetProvider)
        self.assertIsInstance(StubSnippetProvider(), SnippetProvider)
        self.assertIsInstance(IdentityReranker(), Reranker)


class TestHybridRetriever(unittest.TestCase):
    def test_modes_rank_the_right_document_first(self):
        for overrides in ({}, {"use_bm25": False}, {"use_dense": False},
                          {"prf_docs": 1, "prf_beta": 0.3}, {"role1_bm25_terms": True}):
            retr = make_retriever(**overrides)
            retr.index_corpus(list(DOCS), list(DOCS.values()))
            top = retr.retrieve(["compute the greatest common divisor gcd of a and b"], top_k=2)[0]
            self.assertEqual(top[0][0], "gcd", overrides)

    def test_chunk_view_max_pools_to_documents(self):
        cfg = RetrieverConfig(doc_view="chunks", use_bm25=False)
        retr = HybridRetriever(cfg, snippets=ChunkCache(":memory:"),
                               store=EmbeddingStore(":memory:"), encoder=FakeEncoder())
        retr.index_corpus(list(DOCS), list(DOCS.values()))
        ranked = retr.retrieve(["fibonacci sequence"], top_k=4)[0]
        self.assertEqual(ranked[0][0], "fib")
        self.assertEqual(len({d for d, _ in ranked}), 4)

    def test_reranker_reorders_head_only(self):
        class Reverse:
            def rerank(self, query, candidates):
                return [Candidate(c.doc_id, c.text, float(i)) for i, c in enumerate(candidates)][::-1]

        retr = make_retriever(rerank_top_k=2)
        retr.index_corpus(list(DOCS), list(DOCS.values()))
        plain = [d for d, _ in retr.retrieve(["gcd greatest common divisor"], top_k=4)[0]]
        retr.reranker = Reverse()
        reranked = [d for d, _ in retr.retrieve(["gcd greatest common divisor"], top_k=4)[0]]
        self.assertEqual(reranked[:2], plain[:2][::-1])
        self.assertEqual(reranked[2:], plain[2:])

    def test_mteb_search_protocol(self):
        from mteb.models.models_protocols import SearchProtocol

        retr = make_retriever()
        self.assertIsInstance(retr, SearchProtocol)
        corpus = {"id": list(DOCS), "text": list(DOCS.values())}
        rows = [{"id": i, "text": t, "title": ""} for i, t in DOCS.items()]

        class Corpus(list):
            def __getitem__(self, key):
                return corpus[key] if isinstance(key, str) else list.__getitem__(self, key)

        retr.index(Corpus(rows))
        out = retr.search({"id": ["q1"], "text": ["is this number prime"]}, top_k=3)
        self.assertEqual(next(iter(sorted(out["q1"].items(), key=lambda kv: -kv[1])))[0], "prime")


class TestVersioning(unittest.TestCase):
    def test_incremental_reindex_and_cross_version_collapse(self):
        index = VersionedCodeIndex(RetrieverConfig(use_dense=False))
        index.encoder = FakeEncoder()
        index.config.use_dense = True
        d1 = index.add_version("v1", DOCS)
        self.assertEqual((d1.added, d1.embedded), (4, 4))

        v2 = dict(DOCS)
        v2["gcd"] = DOCS["gcd"].replace("a, b = b, a % b", "a, b = b, a % b  # euclid")
        v2.pop("sort")
        v2["lcm"] = "def lcm(a, b):\n    return a * b // gcd(a, b)"
        d2 = index.add_version("v2", v2)
        self.assertEqual((d2.added, d2.modified, d2.removed, d2.unchanged), (1, 1, 1, 2))
        self.assertEqual(d2.embedded, 2)  # only the changed and the new snippet

        self.assertEqual(index.search("bubble sort", version="v1", k=1)[0][0], "sort")
        hits = index.search_all("greatest common divisor gcd euclid", k=5)
        ids = [h.doc_id for h in hits]
        self.assertEqual(len(ids), len(set(ids)))  # one hit per logical snippet
        gcd = next(h for h in hits if h.doc_id == "gcd")
        self.assertEqual(gcd.versions_present, ["v1", "v2"])
        self.assertEqual(gcd.versions_changed, ["v2"])


if __name__ == "__main__":
    unittest.main()
