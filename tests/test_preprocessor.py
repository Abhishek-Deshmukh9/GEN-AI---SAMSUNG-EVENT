"""Unit tests for QueryParser, Intent Classifier, and Identifier Expander."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import IdentifierExpander, QueryParser, StructuredQuery



import unittest


class TestPreprocessor(unittest.TestCase):
    def test_intent_classification(self):
        parser = QueryParser()

        # Behavior lookup (from PDF example)
        q1 = "How is the input preprocessed before going to the main function?"
        res1 = parser.parse(q1)
        self.assertEqual(res1.intent, "behavior-lookup")
        self.assertGreaterEqual(res1.intent_confidence, 0.7)
        self.assertIsInstance(res1, StructuredQuery)

        # Definition lookup
        q2 = "Where is the definition of normalize function?"
        res2 = parser.parse(q2)
        self.assertEqual(res2.intent, "definition-lookup")
        self.assertGreaterEqual(res2.intent_confidence, 0.9)

        # Usage lookup
        q3 = "How to call check function with parameter s?"
        res3 = parser.parse(q3)
        self.assertEqual(res3.intent, "usage-lookup")
        self.assertGreaterEqual(res3.intent_confidence, 0.9)

    def test_identifier_expansion(self):
        expander = IdentifierExpander()
        _, expanded = expander.extract_and_expand("input preprocessed")

        # Verify camelCase, PascalCase, and snake_case variants exist
        self.assertIn("inputPreprocessed", expanded)
        self.assertIn("InputPreprocessed", expanded)
        self.assertIn("input_preprocessed", expanded)

    def test_pydantic_schema_serialization(self):
        parser = QueryParser()
        res = parser.parse("find class PrePostPipelineEncoder")
        self.assertEqual(res.intent, "definition-lookup")
        json_data = res.model_dump()
        self.assertIn("raw_query", json_data)
        self.assertIn("intent", json_data)
        self.assertIn("expanded_identifiers", json_data)
        self.assertIn("processed_query", json_data)


if __name__ == "__main__":
    unittest.main()
