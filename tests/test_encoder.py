"""Unit tests for PrePostPipelineEncoder conforming to MTEB AbsEncoder."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from src.encoder import PrePostPipelineEncoder


def test_encoder_initialization_and_inheritance():
    encoder = PrePostPipelineEncoder(device="cpu")
    assert isinstance(encoder, AbsEncoder)
    assert hasattr(encoder, "mteb_model_meta")
    assert encoder.mteb_model_meta.name == "PrePostPipelineEncoder-all-MiniLM-L6-v2"
    assert encoder.mteb_model_meta.embed_dim == 384


def test_encoder_query_and_document_encoding():
    encoder = PrePostPipelineEncoder(device="cpu")

    # Test query encoding (triggers query preprocessing)
    queries = ["How is the input preprocessed before going to the main function?"]
    query_emb = encoder.encode(queries, prompt_type=PromptType.query)
    assert isinstance(query_emb, np.ndarray)
    assert query_emb.shape == (1, 384)

    # Test document encoding (triggers raw code encoding)
    docs = [
        "function normalize(str) { const str2 = str.trim(); return forward(str2); }",
        "function check(s) { var pre = s.slice(0, 6); return pre === 'en-US'; }",
    ]
    doc_emb = encoder.encode(docs, prompt_type=PromptType.document)
    assert isinstance(doc_emb, np.ndarray)
    assert doc_emb.shape == (2, 384)

    # Test similarity
    sims = encoder.similarity(query_emb, doc_emb)
    assert sims.shape == (1, 2)
    # Cosine similarity between normalized vectors should be bounded [-1, 1]
    assert np.all(sims.numpy() >= -1.0) and np.all(sims.numpy() <= 1.0)


if __name__ == "__main__":
    test_encoder_initialization_and_inheritance()
    test_encoder_query_and_document_encoding()
    print("All encoder tests passed successfully!")
