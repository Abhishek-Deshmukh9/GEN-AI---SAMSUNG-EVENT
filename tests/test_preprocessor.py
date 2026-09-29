"""Unit tests for QueryParser, Intent Classifier, and Identifier Expander."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import IdentifierExpander, QueryParser, StructuredQuery



def test_intent_classification():
    parser = QueryParser()

    # Behavior lookup (from PDF example)
    q1 = "How is the input preprocessed before going to the main function?"
    res1 = parser.parse(q1)
    assert res1.intent == "behavior-lookup"
    assert res1.intent_confidence >= 0.7
    assert isinstance(res1, StructuredQuery)

    # Definition lookup
    q2 = "Where is the definition of normalize function?"
    res2 = parser.parse(q2)
    assert res2.intent == "definition-lookup"
    assert res2.intent_confidence >= 0.9

    # Usage lookup
    q3 = "How to call check function with parameter s?"
    res3 = parser.parse(q3)
    assert res3.intent == "usage-lookup"
    assert res3.intent_confidence >= 0.9


def test_identifier_expansion():
    expander = IdentifierExpander()
    _, expanded = expander.extract_and_expand("input preprocessed")

    # Verify camelCase, PascalCase, and snake_case variants exist
    assert "inputPreprocessed" in expanded
    assert "InputPreprocessed" in expanded
    assert "input_preprocessed" in expanded


def test_pydantic_schema_serialization():
    parser = QueryParser()
    res = parser.parse("find class PrePostPipelineEncoder")
    assert res.intent == "definition-lookup"
    json_data = res.model_dump()
    assert "raw_query" in json_data
    assert "intent" in json_data
    assert "expanded_identifiers" in json_data
    assert "processed_query" in json_data


if __name__ == "__main__":
    test_intent_classification()
    test_identifier_expansion()
    test_pydantic_schema_serialization()
    print("All preprocessor tests passed successfully!")
