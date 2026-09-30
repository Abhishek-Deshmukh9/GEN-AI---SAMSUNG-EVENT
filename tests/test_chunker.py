"""Unit tests for StructuralExtractor, SnippetSchema, LongDocChunker, and ChunkCache."""

import os
import sys
import tempfile
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker import (
    ChunkCache,
    LongDocChunker,
    SnippetSchema,
    StructuralExtractor,
    aggregate_scores,
    build_chunk_to_doc,
    compute_content_hash,
    compute_text_hash,
    estimate_tokens,
)


import unittest


SHORT_DOC = '''"""Utility helpers for normalizing user input."""

import re
from typing import Optional

WHITESPACE_RE = re.compile(r"\\s+")


def normalize(raw: str) -> str:
    """Collapse whitespace and lowercase the input for stable matching."""
    return WHITESPACE_RE.sub(" ", raw).strip().lower()


def check_locale(locale: str, expected: Optional[str] = None) -> bool:
    """Return True when a locale prefix matches the expected value."""
    prefix = locale.split("-")[0]
    return expected is None or prefix == expected
'''


def _make_long_doc(num_functions: int = 12) -> str:
    """Build a synthetic document that comfortably exceeds any sane token budget."""
    header = '"""A deliberately long module used to exercise structural chunking."""\n\nimport os\nimport sys\nfrom typing import Any, Dict, List\n\n\n'
    body: List[str] = []
    for index in range(num_functions):
        body.append(
            f'''
CONSTANT_{index} = "value-{index}"


def process_record_{index}(record: Dict[str, Any], index: int = {index}) -> List[str]:
    """Process record number {index} and return the normalized field values.

    The docstring deliberately carries a few extra words so that signature and
    documentation dominate the embedded summary view of this definition.
    """
    collected: List[str] = []
    for key in sorted(record):
        value = record[key]
        if value is None:
            continue
        text = str(value).strip().lower()
        if not text:
            continue
        collected.append("{{}}={{}}".format(key, text))
        if len(collected) > {index + 4}:
            break
    if not collected:
        collected.append(CONSTANT_{index})
    return collected
'''
        )
    footer = '''
def main(argv: List[str]) -> int:
    """Entry point kept out of the core logic region by the extractor."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        print("running with verbose output enabled for the caller")
    raw = input("path> ")
    return process_record_0({"path": raw})


def build_parser() -> Any:
    """Construct the argument parser used by the entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''
    return header + "".join(body) + footer


MALFORMED_DOC = '''def broken(:\n    this is not python at all\n  )]'''


class TestStructuralExtractor(unittest.TestCase):
    def test_imports_units_and_docstrings(self):
        extract = StructuralExtractor().extract(SHORT_DOC)

        self.assertFalse(extract.parse_failed)
        self.assertEqual(
            extract.imports, ["import re", "from typing import Optional"]
        )
        self.assertEqual([unit.name for unit in extract.units], ["normalize", "check_locale"])
        self.assertEqual(extract.units[0].kind, "function")
        self.assertTrue(
            extract.units[0].signature.startswith("def normalize(raw: str) -> str:")
        )
        self.assertEqual(
            extract.units[0].docstring,
            "Collapse whitespace and lowercase the input for stable matching.",
        )
        self.assertIn("return WHITESPACE_RE.sub", extract.units[0].source)
        self.assertIsNotNone(extract.module_docstring)

    def test_core_logic_strips_boilerplate(self):
        extract = StructuralExtractor().extract(_make_long_doc(num_functions=2))

        self.assertFalse(extract.parse_failed)
        self.assertNotIn("if __name__", extract.core_logic)
        self.assertNotIn("parse_args", extract.core_logic)
        self.assertNotIn("add_argument", extract.core_logic)
        self.assertNotIn("import argparse", extract.core_logic)
        self.assertEqual(len(extract.units), 4)

    def test_class_and_async_definitions_are_detected(self):
        source = (
            "class Base:\n"
            '    """Base class."""\n\n'
            "    def run(self):\n"
            '        """Run it."""\n'
            "        return 1\n\n\n"
            "async def fetch(url: str) -> str:\n"
            '    """Fetch a url."""\n'
            "    return url\n"
        )
        extract = StructuralExtractor().extract(source)

        kinds = {unit.name: unit.kind for unit in extract.units}
        self.assertEqual(kinds, {"Base": "class", "fetch": "async_function"})
        self.assertTrue(extract.units[0].source.endswith("return 1"))

    def test_unparseable_source_falls_back_to_raw_text(self):
        extract = StructuralExtractor().extract(MALFORMED_DOC)

        self.assertTrue(extract.parse_failed)
        self.assertEqual(extract.core_logic, MALFORMED_DOC)
        self.assertEqual(extract.units, [])
        self.assertEqual(extract.imports, [])

    def test_empty_source_is_safe(self):
        extract = StructuralExtractor().extract("")

        self.assertFalse(extract.parse_failed)
        self.assertEqual(extract.core_logic, "")


class TestContentHash(unittest.TestCase):
    def test_hash_ignores_whitespace_and_comments(self):
        compact = "def add(a, b):\n    return a + b\n"
        reformatted = (
            "# a helpful comment\n"
            "def add( a ,   b ):\n\n"
            "    # sums two numbers\n"
            "    return a+b\n"
        )

        self.assertEqual(compute_content_hash(compact), compute_content_hash(reformatted))
        self.assertNotEqual(compute_content_hash(compact), compute_content_hash("def sub(a, b): return a - b"))
        self.assertEqual(len(compute_content_hash(compact)), 64)

    def test_hash_falls_back_for_non_python_text(self):
        self.assertEqual(
            compute_content_hash("plain   text\nhere"),
            compute_content_hash("plain text here"),
        )

    def test_text_hash_is_whitespace_insensitive_and_exact(self):
        self.assertEqual(compute_text_hash("a  b\n\nc"), compute_text_hash("a b c"))
        self.assertNotEqual(compute_text_hash("a b"), compute_text_hash("a b # note"))
        self.assertEqual(
            compute_content_hash("def f():\n    return 1"),
            compute_content_hash("# comment\ndef f():\n        return 1"),
        )


class TestLongDocChunkerShortDoc(unittest.TestCase):
    def test_short_document_is_not_chunked(self):
        chunker = LongDocChunker(max_tokens=256)
        snippets = chunker.chunk(SHORT_DOC, "doc-short")

        self.assertEqual(len(snippets), 1)
        snippet = snippets[0]
        self.assertIsInstance(snippet, SnippetSchema)
        self.assertEqual(snippet.doc_id, "doc-short")
        self.assertEqual(snippet.full_view.strip(), SHORT_DOC.strip())
        self.assertIn("def normalize(raw: str) -> str:", snippet.summary_view)
        self.assertEqual(snippet.metadata.chunk_index, 0)
        self.assertEqual(snippet.metadata.num_chunks, 1)
        self.assertIn("import re", snippet.metadata.has_imports)
        self.assertFalse(snippet.metadata.parse_failed)
        self.assertTrue(snippet.metadata.structural)
        self.assertEqual(len(snippet.id), 64)

    def test_short_document_below_budget_is_untouched_by_low_budget_edges(self):
        chunker = LongDocChunker(max_tokens=estimate_tokens(SHORT_DOC) + 50)
        snippets = chunker.chunk(SHORT_DOC, "doc-short")

        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0].summary_view, snippets[0].full_view)

    def test_empty_document_still_yields_one_snippet(self):
        snippets = LongDocChunker().chunk("", "doc-empty")

        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0].doc_id, "doc-empty")
        self.assertEqual(snippets[0].metadata.num_chunks, 1)


class TestLongDocChunkerLongDoc(unittest.TestCase):
    def setUp(self):
        self.doc = _make_long_doc(num_functions=12)
        self.chunker = LongDocChunker(max_tokens=120, summary_max_lines=15)
        self.snippets = self.chunker.chunk(self.doc, "doc-long")

    def test_long_document_produces_multiple_chunks(self):
        self.assertGreater(len(self.snippets), 1)
        self.assertEqual(
            [s.metadata.chunk_index for s in self.snippets],
            list(range(len(self.snippets))),
        )
        for snippet in self.snippets:
            self.assertEqual(snippet.doc_id, "doc-long")
            self.assertEqual(snippet.metadata.num_chunks, len(self.snippets))
            self.assertTrue(snippet.metadata.structural)
            self.assertIn("import os", snippet.metadata.has_imports)
            self.assertTrue(snippet.full_view.strip())

    def test_chunk_ids_are_unique_and_content_addressed(self):
        ids = [snippet.id for snippet in self.snippets]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(
            {snippet.id for snippet in self.snippets},
            {compute_content_hash(snippet.full_view) for snippet in self.snippets},
        )

    def test_chunks_respect_the_token_budget(self):
        budget = self.chunker.max_tokens
        for snippet in self.snippets:
            self.assertLessEqual(
                estimate_tokens(snippet.full_view),
                budget * 2,
                "chunk body should be split near the token budget",
            )
        self.assertLess(
            max(estimate_tokens(s.full_view) for s in self.snippets),
            estimate_tokens(self.doc),
        )

    def test_chunks_split_on_function_boundaries(self):
        names = {s.metadata.name for s in self.snippets}
        self.assertIn("process_record_0", names)
        self.assertIn("process_record_11", names)
        self.assertIn("<module>", names)
        self.assertNotIn("", names)
        for snippet in self.snippets:
            if snippet.metadata.kind in ("function", "async_function", "class"):
                self.assertTrue(snippet.full_view.startswith("def "))
        module_pieces = [s for s in self.snippets if s.metadata.kind == "module"]
        self.assertTrue(
            any("deliberately long module" in s.full_view for s in module_pieces),
            "the module docstring should remain retrievable as its own chunk",
        )

    def test_entry_point_boilerplate_is_excluded_from_core_logic(self):
        summaries = " ".join(s.summary_view for s in self.snippets if s.metadata.kind == "module")
        self.assertNotIn("sys.exit(main(", summaries)
        self.assertNotIn("add_argument", summaries)

    def test_summary_view_is_truncated_but_full_view_is_complete(self):
        self.assertTrue(
            any(s.summary_view != s.full_view for s in self.snippets),
            "at least one chunk should have a truncated summary view",
        )
        for snippet in self.snippets:
            self.assertLessEqual(
                estimate_tokens(snippet.full_view), estimate_tokens(self.doc)
            )
            self.assertLessEqual(
                estimate_tokens(snippet.summary_view), estimate_tokens(snippet.full_view)
            )

    def test_summary_view_respects_the_line_budget(self):
        chunker = LongDocChunker(max_tokens=80, summary_max_lines=3)
        source = (
            "def wide() -> int:\n"
            '    """A function with a body far longer than three lines."""\n'
            "    total = 0\n"
            + "".join(f"    total += step_{i} * weight_{i} + carry_{i}\n" for i in range(40))
            + "    return total\n"
        )
        snippets = chunker.chunk(source, "doc-wide")

        self.assertGreater(len(snippets), 1)
        header_lines = 2
        for snippet in snippets:
            lines = snippet.summary_view.splitlines()[header_lines:]
            self.assertLessEqual(len(lines), 4)
            if len(lines) == 4:
                self.assertRegex(lines[-1], r"^\.\.\. \(\d+ more lines\)$")
            self.assertTrue(snippet.full_view.startswith("def wide() -> int:"))

    def test_overlong_single_function_is_split_into_paragraph_chunks(self):
        source = (
            '"""Huge module."""\n\n\n'
            "def huge() -> None:\n"
            '    """Do a lot of work in one very long function body."""\n'
            "    total = 0\n"
            + "".join(f"    total += value_{i} * ratio_{i} + offset_{i}\n" for i in range(60))
            + "    return total\n"
        )
        snippets = LongDocChunker(max_tokens=80).chunk(source, "doc-huge")

        self.assertGreater(len(snippets), 1)
        self.assertEqual({s.metadata.name for s in snippets}, {"<module>", "huge"})
        function_pieces = [s for s in snippets if s.metadata.name == "huge"]
        self.assertGreater(len(function_pieces), 1)
        for snippet in function_pieces:
            self.assertEqual(snippet.doc_id, "doc-huge")
            self.assertEqual(snippet.metadata.kind, "function")
            self.assertIn("def huge() -> None:", snippet.full_view)
        self.assertIn("return total", " ".join(s.full_view for s in function_pieces))

    def test_documents_without_definitions_fall_back_to_paragraphs(self):
        source = "\n\n".join(
            f'step_{i}_description = "Paragraph {i} describing step {i} of the workflow"'
            for i in range(40)
        )
        snippets = LongDocChunker(max_tokens=60).chunk(source, "doc-prose")

        self.assertGreater(len(snippets), 1)
        self.assertFalse(snippets[0].metadata.structural)
        self.assertEqual({s.doc_id for s in snippets}, {"doc-prose"})

    def test_chunk_corpus_preserves_document_ids(self):
        corpus = {"a": SHORT_DOC, "b": _make_long_doc(num_functions=6)}
        chunked = self.chunker.chunk_corpus(corpus)

        self.assertEqual(set(chunked), {"a", "b"})
        self.assertEqual({s.doc_id for s in chunked["a"]}, {"a"})
        self.assertEqual({s.doc_id for s in chunked["b"]}, {"b"})


class TestUnparseableDocument(unittest.TestCase):
    def test_unparseable_document_uses_raw_text_fallback(self):
        chunker = LongDocChunker(max_tokens=64)
        snippets = chunker.chunk(MALFORMED_DOC, "doc-broken")

        self.assertEqual(len(snippets), 1)
        snippet = snippets[0]
        self.assertTrue(snippet.metadata.parse_failed)
        self.assertFalse(snippet.metadata.structural)
        self.assertEqual(snippet.full_view, MALFORMED_DOC)
        self.assertEqual(snippet.summary_view, MALFORMED_DOC)
        self.assertEqual(snippet.metadata.has_imports, [])
        self.assertEqual(snippet.id, compute_content_hash(MALFORMED_DOC))

    def test_long_unparseable_document_is_not_dropped(self):
        junk = "\n".join(f"not valid python line {i}" for i in range(200))
        snippets = LongDocChunker(max_tokens=64).chunk(junk, "doc-junk")

        self.assertEqual(len(snippets), 1)
        self.assertTrue(snippets[0].metadata.parse_failed)
        self.assertEqual(snippets[0].full_view, junk)
        self.assertLess(
            len(snippets[0].summary_view.splitlines()), len(junk.splitlines())
        )

    def test_schema_round_trips_through_json(self):
        snippet = LongDocChunker().chunk(SHORT_DOC, "doc-short")[0]
        restored = SnippetSchema.model_validate_json(snippet.model_dump_json())

        self.assertEqual(restored.id, snippet.id)
        self.assertEqual(restored.doc_id, "doc-short")
        self.assertEqual(restored.metadata.has_imports, snippet.metadata.has_imports)
        self.assertEqual(restored.metadata.chunk_index, 0)


class _CountingChunker(LongDocChunker):
    """Chunker that records how often `chunk` actually recomputes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def chunk(self, source, doc_id):
        self.calls += 1
        return super().chunk(source, doc_id)


class TestChunkCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "nested", "chunker_cache.sqlite3")

    def tearDown(self):
        self._tmp.cleanup()

    def test_cache_skips_reprocessing_unchanged_snippets(self):
        chunker = _CountingChunker(max_tokens=120)
        with ChunkCache(self.db_path, chunker=chunker) as cache:
            first = cache.get_or_compute(SHORT_DOC, "doc-short")
            second = cache.get_or_compute(SHORT_DOC, "doc-short")

            self.assertEqual(chunker.calls, 1)
            self.assertEqual(cache.stats(), {"hits": 1, "misses": 1, "entries": 1})
            self.assertEqual([s.id for s in first], [s.id for s in second])
            self.assertEqual([s.summary_view for s in first], [s.summary_view for s in second])

    def test_cache_persists_across_instances_and_keys_on_content(self):
        chunker = _CountingChunker(max_tokens=120)
        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(_make_long_doc(num_functions=4), "doc-long")

        with ChunkCache(self.db_path, chunker=chunker) as reopened:
            snippets = reopened.get_or_compute(_make_long_doc(num_functions=4), "doc-long")
            self.assertGreater(len(snippets), 1)
            self.assertEqual(chunker.calls, 1)
            self.assertEqual(reopened.stats()["hits"], 1)

            reopened.get_or_compute(SHORT_DOC, "doc-short")
            self.assertEqual(chunker.calls, 2)

    def test_cache_ignores_pure_formatting_changes(self):
        chunker = _CountingChunker(max_tokens=256)
        reformatted = "\n\n".join(
            line.rstrip() for line in SHORT_DOC.splitlines() if line.strip()
        )

        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(SHORT_DOC, "doc-a")
            cache.get_or_compute(reformatted, "doc-b")

        self.assertEqual(chunker.calls, 1)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_comment_only_edits_miss_the_cache(self):
        chunker = _CountingChunker(max_tokens=256)
        commented = "# a new leading comment\n" + SHORT_DOC

        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(SHORT_DOC, "doc-a")
            cache.get_or_compute(commented, "doc-a")

        self.assertEqual(chunker.calls, 2)
        self.assertEqual(cache.stats()["entries"], 2)

    def test_cache_recomputes_when_budget_changes(self):
        chunker = _CountingChunker(max_tokens=256)
        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(_make_long_doc(num_functions=6), "doc-long")
            chunker.max_tokens = 64
            cache.get_or_compute(_make_long_doc(num_functions=6), "doc-long")

        self.assertEqual(chunker.calls, 2)

    def test_cached_snippets_are_restamped_with_requested_doc_id(self):
        chunker = LongDocChunker(max_tokens=120)
        source = _make_long_doc(num_functions=4)

        with ChunkCache(self.db_path, chunker=chunker) as cache:
            first = cache.get_or_compute(source, "doc-one")
            second = cache.get_or_compute(source, "doc-two")

        self.assertEqual({s.doc_id for s in first}, {"doc-one"})
        self.assertEqual({s.doc_id for s in second}, {"doc-two"})
        self.assertEqual([s.id for s in first], [s.id for s in second])

    def test_in_memory_cache_is_functional(self):
        with ChunkCache(":memory:") as cache:
            first = cache.get_or_compute(SHORT_DOC, "doc-short")
            second = cache.get_or_compute(SHORT_DOC, "doc-short")

            cache.clear()
            self.assertEqual(cache.stats()["entries"], 0)

        self.assertEqual(len(first), len(second))

    def test_clear_forces_recomputation(self):
        chunker = _CountingChunker(max_tokens=256)
        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(SHORT_DOC, "doc-short")
            cache.clear()
            cache.get_or_compute(SHORT_DOC, "doc-short")

        self.assertEqual(chunker.calls, 2)

    def test_cache_hits_avoid_parsing_the_document(self):
        chunker = LongDocChunker(max_tokens=120)
        source = _make_long_doc(num_functions=6)
        parsed: List[str] = []
        extractor = chunker.extractor.extract

        def counting_extract(text):
            parsed.append(text)
            return extractor(text)

        chunker.extractor.extract = counting_extract
        with ChunkCache(self.db_path, chunker=chunker) as cache:
            cache.get_or_compute(source, "doc-long")
            cache.get_or_compute(source, "doc-long")

        self.assertEqual(len(parsed), 1, "cache hits must not re-parse the document")


class TestAggregateScores(unittest.TestCase):
    def test_max_pools_chunk_scores_onto_documents(self):
        chunk_scores = {
            "chunk_a0": 0.10,
            "chunk_a1": 0.85,
            "chunk_a2": 0.42,
            "chunk_b0": 0.05,
            "chunk_b1": 0.31,
        }
        chunk_to_doc = {
            "chunk_a0": "doc-a",
            "chunk_a1": "doc-a",
            "chunk_a2": "doc-a",
            "chunk_b0": "doc-b",
            "chunk_b1": "doc-b",
        }

        doc_scores = aggregate_scores(chunk_scores, chunk_to_doc)

        self.assertEqual(doc_scores, {"doc-a": 0.85, "doc-b": 0.31})

    def test_orders_documents_by_first_seen_chunk(self):
        doc_scores = aggregate_scores(
            {"c1": 0.2, "c2": 0.9, "c3": 0.4}, {"c1": "doc-b", "c2": "doc-a", "c3": "doc-b"}
        )

        self.assertEqual(list(doc_scores), ["doc-b", "doc-a"])

    def test_ignores_unmapped_chunks_and_empty_inputs(self):
        self.assertEqual(aggregate_scores({"c1": 0.5}, {}), {})
        self.assertEqual(aggregate_scores({}, {"c1": "doc-a"}), {})
        self.assertEqual(
            aggregate_scores({"c1": 0.5, "orphan": 0.99}, {"c1": "doc-a"}), {"doc-a": 0.5}
        )

    def test_agrees_with_chunking_pipeline_output(self):
        corpus = {"doc-a": SHORT_DOC, "doc-b": _make_long_doc(num_functions=8)}
        snippets = []
        for doc_id, source in corpus.items():
            snippets.extend(LongDocChunker(max_tokens=120).chunk(source, doc_id))

        chunk_to_doc = build_chunk_to_doc(snippets)
        self.assertEqual(len(chunk_to_doc), len(snippets))

        chunk_scores = {snippet.id: index / 100.0 for index, snippet in enumerate(snippets)}
        doc_scores = aggregate_scores(chunk_scores, chunk_to_doc)

        self.assertEqual(set(doc_scores), set(corpus))
        chunks_per_doc = {doc_id: sum(1 for s in snippets if s.doc_id == doc_id) for doc_id in corpus}
        self.assertGreater(chunks_per_doc["doc-b"], 1)
        for doc_id in corpus:
            self.assertEqual(
                doc_scores[doc_id],
                max(score for cid, score in chunk_scores.items() if chunk_to_doc[cid] == doc_id),
            )

    def test_handles_single_chunk_documents(self):
        self.assertEqual(aggregate_scores({"c1": 0.7}, {"c1": "doc-a"}), {"doc-a": 0.7})
        self.assertEqual(
            aggregate_scores({"c1": -1.0, "c2": -0.5}, {"c1": "doc-a", "c2": "doc-a"}),
            {"doc-a": -0.5},
        )

    def test_identical_chunks_in_two_documents_do_not_collide(self):
        snippets = [
            LongDocChunker().chunk(SHORT_DOC, "doc-a")[0],
            LongDocChunker().chunk(SHORT_DOC, "doc-b")[0],
        ]
        self.assertEqual(snippets[0].id, snippets[1].id)

        mapping = build_chunk_to_doc(snippets)
        self.assertEqual(len(mapping), 2, "identical chunks must not share a score key")
        keys = list(mapping)
        doc_scores = aggregate_scores({keys[0]: 0.9, keys[1]: 0.4}, mapping)

        self.assertEqual(doc_scores, {"doc-a": 0.9, "doc-b": 0.4})


if __name__ == "__main__":
    unittest.main()
