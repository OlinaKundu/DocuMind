"""Unit tests for DocuMind's embedding generation module (Phase 3).

By default, all tests use a deterministic in-memory fake model so that the test suite
remains 100% offline-safe, fast, and does not require downloading model weights from Hugging Face.
An optional real-model smoke test is explicitly marked with @pytest.mark.real_model.
"""

from typing import Sequence
import numpy as np
import pytest
from unittest.mock import patch

from src.chunker import TextChunk
from src.embeddings import (
    ChunkEmbeddings,
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EmbeddingError,
    ModelLoadingError,
    embed_chunks,
    embed_query,
    embed_texts,
    get_embedding_model,
)


class MockSentenceTransformer:
    """Deterministic, offline-safe mock that produces valid (N, 384) float32 unit vectors."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def encode(
        self,
        sentences: Sequence[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = []
        for s in sentences:
            # Deterministic pseudo-random seed based on string content
            seed = (abs(hash(s)) % 100000) + 1
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimension, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def mock_model():
    """Provides a fresh MockSentenceTransformer instance for test isolation."""
    return MockSentenceTransformer(dimension=EMBEDDING_DIMENSION)


def test_embed_texts_shape_and_dtype(mock_model):
    """Verify embed_texts returns a 2D float32 array with (N, 384) dimensions."""
    texts = [
        "First technical paragraph about neural networks.",
        "Second paragraph describing backpropagation algorithms.",
        "Third paragraph evaluating experimental metrics.",
    ]
    vectors = embed_texts(texts, model=mock_model)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32


def test_embed_texts_unit_normalization(mock_model):
    """Every generated vector must be unit-normalized (L2 norm approximately 1.0)."""
    texts = ["Sample passage for norm verification.", "Another sentence."]
    vectors = embed_texts(texts, model=mock_model)

    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-5), f"Expected norm 1.0, got {norm}"


def test_embed_texts_ordering_preserved(mock_model):
    """Verify that output vectors map 1:1 to the input text sequence order."""
    text_a = "Topic Alpha"
    text_b = "Topic Beta"

    # Encode individually
    vec_a = embed_texts([text_a], model=mock_model)[0]
    vec_b = embed_texts([text_b], model=mock_model)[0]

    # Encode together
    batch_vecs = embed_texts([text_a, text_b], model=mock_model)

    assert np.allclose(batch_vecs[0], vec_a, atol=1e-6)
    assert np.allclose(batch_vecs[1], vec_b, atol=1e-6)
    assert not np.allclose(vec_a, vec_b)  # Different texts yield distinct vectors


def test_embed_texts_empty_sequence(mock_model):
    """Empty text sequence must return an empty (0, 384) float32 array without calling encode."""
    empty_result = embed_texts([], model=mock_model)

    assert isinstance(empty_result, np.ndarray)
    assert empty_result.shape == (0, 384)
    assert empty_result.dtype == np.float32


def test_embed_texts_invalid_inputs(mock_model):
    """Invalid input types should be rejected with clear, descriptive exceptions."""
    # None input
    with pytest.raises(ValueError, match="cannot be None"):
        embed_texts(None, model=mock_model)

    # Passing a single string instead of a sequence
    with pytest.raises(TypeError, match="Expected a sequence of strings"):
        embed_texts("A single naked string", model=mock_model)

    # Sequence containing non-string items
    with pytest.raises(TypeError, match="All elements in 'texts' must be strings"):
        embed_texts(["valid text", 12345, "another text"], model=mock_model)


def test_embed_chunks_preserves_metadata_and_alignment(mock_model):
    """ChunkEmbeddings must strictly preserve chunk order, metadata, and 1:1 vector alignment."""
    chunks = [
        TextChunk(
            chunk_id="p1_c0",
            page_number=1,
            text="Chunk 0 on page 1",
            start_char=0,
            end_char=17,
        ),
        TextChunk(
            chunk_id="p1_c1",
            page_number=1,
            text="Chunk 1 on page 1",
            start_char=10,
            end_char=27,
        ),
        TextChunk(
            chunk_id="p2_c0",
            page_number=2,
            text="Chunk 0 on page 2",
            start_char=0,
            end_char=17,
        ),
    ]

    result = embed_chunks(chunks, model=mock_model)

    assert isinstance(result, ChunkEmbeddings)
    assert result.count == 3
    assert result.dimension == 384
    assert result.embeddings.shape == (3, 384)
    assert result.embeddings.dtype == np.float32

    # Check metadata integrity
    assert result.chunks[0].chunk_id == "p1_c0"
    assert result.chunks[0].page_number == 1
    assert result.chunks[2].chunk_id == "p2_c0"
    assert result.chunks[2].page_number == 2

    # Test tuple unpacking: chunks, vectors = embed_chunks(...)
    unpacked_chunks, unpacked_vectors = result
    assert unpacked_chunks == chunks
    assert np.array_equal(unpacked_vectors, result.embeddings)


def test_embed_chunks_empty_sequence(mock_model):
    """Embedding empty chunk list should return a ChunkEmbeddings with (0, 384) array."""
    result = embed_chunks([], model=mock_model)

    assert result.count == 0
    assert result.chunks == []
    assert result.embeddings.shape == (0, 384)
    assert result.embeddings.dtype == np.float32


def test_embed_chunks_invalid_items(mock_model):
    """Passing non-TextChunk objects inside the sequence must raise a TypeError."""
    with pytest.raises(TypeError, match="must be TextChunk instances"):
        embed_chunks(["not a TextChunk"], model=mock_model)

    with pytest.raises(ValueError, match="cannot be None"):
        embed_chunks(None, model=mock_model)


def test_embed_query_vector_properties(mock_model):
    """embed_query should return a 1D unit vector of shape (384,) and float32 dtype."""
    query = "What is the computational complexity of transformer attention?"
    vec = embed_query(query, model=mock_model)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_embed_query_invalid_inputs(mock_model):
    """Invalid query strings (empty, whitespace, non-string) must raise informative errors."""
    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query("", model=mock_model)

    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query("     \n\t   ", model=mock_model)

    with pytest.raises(TypeError, match="query must be a string"):
        embed_query(12345, model=mock_model)


def test_model_loading_error_reported_clearly():
    """Verify that a model loading failure raises ModelLoadingError with helpful instructions."""
    with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Connection refused")):
        # Clear the lru_cache first to ensure get_embedding_model attempts to reload
        get_embedding_model.cache_clear()
        with pytest.raises(ModelLoadingError, match="active internet access is required"):
            get_embedding_model("nonexistent-test-model-xyz")


@pytest.mark.real_model
def test_real_model_smoke_test():
    """Optional smoke test that verifies real SentenceTransformer model execution.

    Requires local model weights or internet access.
    Can be run explicitly with: pytest -m real_model
    """
    model = get_embedding_model(DEFAULT_MODEL_NAME)
    sample_text = "DocuMind offline-first RAG embedding verification."
    vec = embed_query(sample_text, model=model)

    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
