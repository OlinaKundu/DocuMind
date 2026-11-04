"""Unit tests for DocuMind's FAISS retrieval module (Phase 4).

All tests run 100% offline using deterministic in-memory vectors and mock models.
No real Sentence Transformer weights or network calls are required.
"""

from typing import List, Sequence
import numpy as np
import pytest

# Verify FAISS installation
try:
    import faiss
except ImportError:
    pytest.fail("FAISS (faiss-cpu) is required but not installed in the test environment.")

from src.chunker import TextChunk
from src.embeddings import (
    EMBEDDING_DIMENSION,
    ChunkEmbeddings,
)
from src.retriever import (
    IndexBuildingError,
    RetrievalError,
    SearchError,
    SearchResult,
    VectorStore,
    build_index,
    retrieve,
    search_index,
)


def _make_unit_vector(dim: int = EMBEDDING_DIMENSION, active_idx: int = 0) -> np.ndarray:
    """Creates a deterministic unit-normalized 1D vector with 1.0 at active_idx."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[active_idx % dim] = 1.0
    return vec


def _make_custom_unit_vector(components: List[float], dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    """Creates a deterministic unit-normalized vector with specific initial components."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, val in enumerate(components):
        vec[i] = val
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


def _make_chunk(chunk_id: str, page_number: int, text: str) -> TextChunk:
    """Helper to create a TextChunk instance."""
    return TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )


class MockEmbeddingModel:
    """Deterministic offline mock model returning predictable unit vectors."""

    def __init__(self, target_vector: np.ndarray):
        self.target_vector = target_vector

    def encode(
        self,
        sentences: Sequence[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(self.target_vector)), dtype=np.float32)
        # Return copies of the target vector for each sentence
        return np.tile(self.target_vector, (len(sentences), 1)).astype(np.float32)


# ==============================================================================
# 1. Index Construction Tests
# ==============================================================================

def test_build_index_correct_dimension_and_count():
    """Verify that build_index creates a FAISS IndexFlatIP with correct dimensions and count."""
    chunks = [
        _make_chunk("p1_c0", 1, "First chunk text."),
        _make_chunk("p1_c1", 1, "Second chunk text."),
        _make_chunk("p2_c0", 2, "Third chunk text on page 2."),
    ]
    vecs = np.stack([_make_unit_vector(EMBEDDING_DIMENSION, i) for i in range(3)])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=vecs)

    vs = build_index(ce, dimension=EMBEDDING_DIMENSION)

    assert isinstance(vs, VectorStore)
    assert vs.dimension == EMBEDDING_DIMENSION
    assert vs.count == 3
    assert not vs.is_empty
    assert vs.index.d == EMBEDDING_DIMENSION
    assert vs.index.ntotal == 3
    assert len(vs.chunks) == 3


def test_build_index_preserves_vector_chunk_ordering():
    """Verify that chunk ordering in VectorStore strictly matches input ChunkEmbeddings."""
    chunks = [
        _make_chunk("chunk_alpha", 1, "Alpha content"),
        _make_chunk("chunk_beta", 2, "Beta content"),
        _make_chunk("chunk_gamma", 3, "Gamma content"),
    ]
    vecs = np.stack([_make_unit_vector(EMBEDDING_DIMENSION, i) for i in range(3)])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=vecs)

    vs = build_index(ce)

    assert [c.chunk_id for c in vs.chunks] == ["chunk_alpha", "chunk_beta", "chunk_gamma"]
    assert [c.page_number for c in vs.chunks] == [1, 2, 3]


def test_build_index_empty_collection():
    """Verify that building an index from an empty collection succeeds gracefully."""
    empty_ce = ChunkEmbeddings(
        chunks=[],
        embeddings=np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32),
    )

    vs = build_index(empty_ce, dimension=EMBEDDING_DIMENSION)

    assert isinstance(vs, VectorStore)
    assert vs.count == 0
    assert vs.is_empty
    assert vs.index.ntotal == 0
    assert vs.chunks == []


def test_build_index_invalid_embedding_shapes():
    """Verify that 1D, 3D, and count-mismatched embedding shapes raise IndexBuildingError."""
    chunk = _make_chunk("p1_c0", 1, "Sample chunk.")

    # 1D array instead of 2D
    with pytest.raises(IndexBuildingError, match="must be 2-dimensional"):
        build_index(
            ChunkEmbeddings(
                chunks=[chunk],
                embeddings=np.ones(EMBEDDING_DIMENSION, dtype=np.float32),
            )
        )

    # 3D array instead of 2D
    with pytest.raises(IndexBuildingError, match="must be 2-dimensional"):
        build_index(
            ChunkEmbeddings(
                chunks=[chunk],
                embeddings=np.ones((1, 1, EMBEDDING_DIMENSION), dtype=np.float32),
            )
        )

    # Chunk count does not match row count of embeddings
    with pytest.raises(IndexBuildingError, match="Chunk count .* does not match"):
        build_index(
            ChunkEmbeddings(
                chunks=[chunk],  # 1 chunk
                embeddings=np.stack([  # 2 vectors
                    _make_unit_vector(EMBEDDING_DIMENSION, 0),
                    _make_unit_vector(EMBEDDING_DIMENSION, 1),
                ]),
            )
        )


def test_build_index_incorrect_dimensions():
    """Verify that embedding matrices with incorrect dimensions are rejected."""
    chunk = _make_chunk("p1_c0", 1, "Sample chunk.")
    wrong_dim = 512
    vec = _make_unit_vector(wrong_dim, 0).reshape(1, -1)

    with pytest.raises(IndexBuildingError, match="Embedding dimension mismatch"):
        build_index(
            ChunkEmbeddings(chunks=[chunk], embeddings=vec),
            dimension=EMBEDDING_DIMENSION,  # 384
        )


def test_build_index_nan_and_infinity_rejection():
    """Verify that NaN, +Inf, and -Inf values in embeddings raise IndexBuildingError."""
    chunk = _make_chunk("p1_c0", 1, "Sample chunk.")

    # Embeddings with NaN
    nan_vec = _make_unit_vector(EMBEDDING_DIMENSION, 0).reshape(1, -1)
    nan_vec[0, 5] = np.nan
    with pytest.raises(IndexBuildingError, match="non-finite values"):
        build_index(ChunkEmbeddings(chunks=[chunk], embeddings=nan_vec))

    # Embeddings with +Inf
    inf_vec = _make_unit_vector(EMBEDDING_DIMENSION, 0).reshape(1, -1)
    inf_vec[0, 5] = np.inf
    with pytest.raises(IndexBuildingError, match="non-finite values"):
        build_index(ChunkEmbeddings(chunks=[chunk], embeddings=inf_vec))

    # Embeddings with -Inf
    neginf_vec = _make_unit_vector(EMBEDDING_DIMENSION, 0).reshape(1, -1)
    neginf_vec[0, 5] = -np.inf
    with pytest.raises(IndexBuildingError, match="non-finite values"):
        build_index(ChunkEmbeddings(chunks=[chunk], embeddings=neginf_vec))


def test_build_index_non_normalized_vector_handling():
    """Verify that unnormalized embedding vectors are caught and rejected."""
    chunk = _make_chunk("p1_c0", 1, "Sample chunk.")

    # Vector with norm = 2.0 (not unit normalized)
    unnormalized_vec = (_make_unit_vector(EMBEDDING_DIMENSION, 0) * 2.0).reshape(1, -1)

    with pytest.raises(IndexBuildingError, match="must be unit-normalized"):
        build_index(ChunkEmbeddings(chunks=[chunk], embeddings=unnormalized_vec))


def test_build_index_invalid_input_types():
    """Verify that None or non-ChunkEmbeddings objects raise ValueError or TypeError."""
    with pytest.raises(ValueError, match="cannot be None"):
        build_index(None)

    with pytest.raises(TypeError, match="Expected a ChunkEmbeddings instance"):
        build_index("not a ChunkEmbeddings object")

    with pytest.raises(TypeError, match="Expected a ChunkEmbeddings instance"):
        build_index([_make_chunk("p1_c0", 1, "test")])


# ==============================================================================
# 2. Search Tests
# ==============================================================================

@pytest.fixture
def sample_vector_store():
    """Creates a predictable 3-vector store with known cosine similarities.

    v0 = [1.0, 0.0, 0.0, ...]
    v1 = [0.8, 0.6, 0.0, ...] (norm = 1.0)
    v2 = [0.0, 1.0, 0.0, ...]
    """
    v0 = _make_custom_unit_vector([1.0, 0.0, 0.0])
    v1 = _make_custom_unit_vector([0.8, 0.6, 0.0])
    v2 = _make_custom_unit_vector([0.0, 1.0, 0.0])

    chunks = [
        _make_chunk("p1_c0", 1, "Chunk zero (x-axis aligned)"),
        _make_chunk("p1_c1", 1, "Chunk one (intermediate angle)"),
        _make_chunk("p2_c0", 2, "Chunk two (y-axis aligned)"),
    ]
    ce = ChunkEmbeddings(chunks=chunks, embeddings=np.stack([v0, v1, v2]))
    return build_index(ce)


def test_search_correct_nearest_neighbor_ranking(sample_vector_store):
    """Verify that search ranks candidates in descending order of inner product / cosine similarity."""
    # Query q = [0.96, 0.28, 0.0] (norm = 1.0)
    # Expected dot products:
    # q . v0 = 0.960
    # q . v1 = 0.96 * 0.8 + 0.28 * 0.6 = 0.768 + 0.168 = 0.936
    # q . v2 = 0.280
    # Expected rank order: 0, 1, 2
    query = _make_custom_unit_vector([0.96, 0.28, 0.0])

    scores, indices = search_index(sample_vector_store, query, top_k=3)

    assert len(scores) == 3
    assert len(indices) == 3
    assert list(indices) == [0, 1, 2]
    assert np.isclose(scores[0], 0.960, atol=1e-3)
    assert np.isclose(scores[1], 0.936, atol=1e-3)
    assert np.isclose(scores[2], 0.280, atol=1e-3)


def test_search_similarity_scores_in_descending_order(sample_vector_store):
    """Verify that similarity scores are strictly monotonically descending."""
    query = _make_custom_unit_vector([0.6, 0.8, 0.0])

    scores, _ = search_index(sample_vector_store, query, top_k=3)

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score {scores[i]} is smaller than next score {scores[i+1]}"


def test_search_top_k_equals_one(sample_vector_store):
    """Verify top_k = 1 returns exactly one nearest neighbor."""
    query = _make_custom_unit_vector([0.0, 1.0, 0.0])  # identical to v2

    scores, indices = search_index(sample_vector_store, query, top_k=1)

    assert len(scores) == 1
    assert len(indices) == 1
    assert indices[0] == 2
    assert np.isclose(scores[0], 1.0, atol=1e-3)


def test_search_top_k_greater_than_indexed_vectors(sample_vector_store):
    """Verify that requesting top_k larger than total indexed vectors clamps results without padding."""
    query = _make_custom_unit_vector([1.0, 0.0, 0.0])

    # Index has 3 vectors, request top_k=10
    scores, indices = search_index(sample_vector_store, query, top_k=10)

    assert len(scores) == 3
    assert len(indices) == 3
    # No FAISS -1 sentinel indices
    assert all(idx >= 0 for idx in indices)
    assert set(indices) == {0, 1, 2}


def test_search_invalid_top_k_values(sample_vector_store):
    """Verify invalid top_k values (0, negative, bools, floats, strings) raise ValueError."""
    query = _make_unit_vector(EMBEDDING_DIMENSION, 0)

    # Zero
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k=0)

    # Negative
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k=-3)

    # Boolean True (in Python, isinstance(True, int) is True unless guarded)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k=True)

    # Boolean False
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k=False)

    # Float
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k=3.5)

    # String
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_index(sample_vector_store, query, top_k="5")


def test_search_invalid_query_dimensions(sample_vector_store):
    """Verify mismatched or multidimensional query vectors raise SearchError."""
    # Wrong 1D dimension (128 instead of 384)
    short_query = _make_unit_vector(128, 0)
    with pytest.raises(SearchError, match="does not match index dimension"):
        search_index(sample_vector_store, short_query)

    # 2D query with multiple rows (batch query rejected by single-query API)
    batch_query = np.stack([
        _make_unit_vector(EMBEDDING_DIMENSION, 0),
        _make_unit_vector(EMBEDDING_DIMENSION, 1),
    ])
    with pytest.raises(SearchError, match="Expected a single query vector"):
        search_index(sample_vector_store, batch_query)

    # 3D query
    cube_query = np.ones((1, 1, EMBEDDING_DIMENSION), dtype=np.float32)
    with pytest.raises(SearchError, match="Query vector must be 1D or 2D"):
        search_index(sample_vector_store, cube_query)


def test_search_non_finite_query_vectors(sample_vector_store):
    """Verify queries containing NaN or Inf raise SearchError."""
    nan_query = _make_unit_vector(EMBEDDING_DIMENSION, 0)
    nan_query[0] = np.nan
    with pytest.raises(SearchError, match="non-finite values"):
        search_index(sample_vector_store, nan_query)

    inf_query = _make_unit_vector(EMBEDDING_DIMENSION, 0)
    inf_query[0] = np.inf
    with pytest.raises(SearchError, match="non-finite values"):
        search_index(sample_vector_store, inf_query)


def test_search_zero_length_query_vector(sample_vector_store):
    """Verify empty query array or all-zeros query vector raises SearchError."""
    # Zero-length array
    empty_vec = np.array([], dtype=np.float32)
    with pytest.raises(SearchError, match="cannot be empty"):
        search_index(sample_vector_store, empty_vec)

    # All-zeros vector (zero L2 norm)
    zero_vec = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    with pytest.raises(SearchError, match="zero or near-zero L2 norm"):
        search_index(sample_vector_store, zero_vec)


def test_search_unnormalized_query_vector(sample_vector_store):
    """Verify unnormalized query vectors raise SearchError."""
    unnorm_query = _make_unit_vector(EMBEDDING_DIMENSION, 0) * 3.0
    with pytest.raises(SearchError, match="must be unit-normalized"):
        search_index(sample_vector_store, unnorm_query)


def test_search_empty_index():
    """Verify searching an empty index returns empty score and index arrays gracefully."""
    empty_ce = ChunkEmbeddings(
        chunks=[],
        embeddings=np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32),
    )
    vs = build_index(empty_ce)
    query = _make_unit_vector(EMBEDDING_DIMENSION, 0)

    scores, indices = search_index(vs, query, top_k=5)

    assert isinstance(scores, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert len(scores) == 0
    assert len(indices) == 0


def test_search_accepts_raw_faiss_index(sample_vector_store):
    """Verify search_index accepts a raw faiss.IndexFlatIP object as well as a VectorStore."""
    query = _make_custom_unit_vector([1.0, 0.0, 0.0])

    scores, indices = search_index(sample_vector_store.index, query, top_k=2)

    assert len(scores) == 2
    assert len(indices) == 2
    assert indices[0] == 0


# ==============================================================================
# 3. Metadata Preservation & High-Level Retrieval Tests
# ==============================================================================

def test_retrieve_maps_faiss_index_to_correct_text_chunk():
    """Verify that each retrieved FAISS index maps to the exact source TextChunk and metadata."""
    chunks = [
        _make_chunk("p1_c0", 1, "Introduction to neural network optimization."),
        _make_chunk("p1_c1", 1, "Loss landscape and gradient descent dynamics."),
        _make_chunk("p2_c0", 2, "Empirical evaluation on benchmark datasets."),
        _make_chunk("p3_c0", 3, "Conclusion and future research directions."),
    ]
    # Orthogonal unit vectors for complete distinctness
    vecs = np.stack([_make_unit_vector(EMBEDDING_DIMENSION, i) for i in range(4)])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=vecs)
    vs = build_index(ce)

    # Query matching chunk index 2 (Page 2)
    query_vec = _make_unit_vector(EMBEDDING_DIMENSION, 2)
    results = retrieve(query_vec, vs, top_k=1)

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, SearchResult)
    assert res.index == 2
    assert res.chunk == chunks[2]
    assert res.chunk_id == "p2_c0"
    assert res.page_number == 2
    assert res.text == "Empirical evaluation on benchmark datasets."
    assert np.isclose(res.score, 1.0, atol=1e-3)


def test_retrieve_preserves_score_to_chunk_correspondence():
    """Verify that similarity scores strictly correspond to the correct chunks."""
    chunks = [
        _make_chunk("p1_c0", 1, "Chunk A"),
        _make_chunk("p2_c0", 2, "Chunk B"),
        _make_chunk("p3_c0", 3, "Chunk C"),
    ]
    v0 = _make_custom_unit_vector([1.0, 0.0, 0.0])
    v1 = _make_custom_unit_vector([0.8, 0.6, 0.0])
    v2 = _make_custom_unit_vector([0.0, 1.0, 0.0])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=np.stack([v0, v1, v2]))
    vs = build_index(ce)

    query_vec = _make_custom_unit_vector([0.96, 0.28, 0.0])
    results = retrieve(query_vec, vs, top_k=3)

    assert len(results) == 3
    # Check 1:1 correspondence
    assert results[0].index == 0
    assert results[0].chunk.chunk_id == "p1_c0"
    assert np.isclose(results[0].score, 0.960, atol=1e-3)

    assert results[1].index == 1
    assert results[1].chunk.chunk_id == "p2_c0"
    assert np.isclose(results[1].score, 0.936, atol=1e-3)

    assert results[2].index == 2
    assert results[2].chunk.chunk_id == "p3_c0"
    assert np.isclose(results[2].score, 0.280, atol=1e-3)


def test_retrieve_empty_index_returns_empty_list():
    """Verify retrieve on an empty VectorStore returns an empty list."""
    empty_ce = ChunkEmbeddings(
        chunks=[],
        embeddings=np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32),
    )
    vs = build_index(empty_ce)
    query_vec = _make_unit_vector(EMBEDDING_DIMENSION, 0)

    results = retrieve(query_vec, vs, top_k=5)

    assert isinstance(results, list)
    assert len(results) == 0


def test_retrieve_top_k_greater_than_available_chunks():
    """Verify retrieve with top_k > available chunks returns all available chunks without errors."""
    chunks = [
        _make_chunk("p1_c0", 1, "Chunk 1"),
        _make_chunk("p1_c1", 1, "Chunk 2"),
    ]
    vecs = np.stack([_make_unit_vector(EMBEDDING_DIMENSION, i) for i in range(2)])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=vecs)
    vs = build_index(ce)

    results = retrieve(_make_unit_vector(EMBEDDING_DIMENSION, 0), vs, top_k=10)

    assert len(results) == 2
    assert results[0].chunk_id == "p1_c0"
    assert results[1].chunk_id == "p1_c1"


def test_retrieve_invalid_vector_store_type():
    """Verify retrieve rejects invalid vector_store parameter."""
    with pytest.raises(TypeError, match="Expected 'vector_store' to be a VectorStore"):
        retrieve("search query", "invalid_store", top_k=3)


# ==============================================================================
# 4. Integration Tests (End-to-End Flow)
# ==============================================================================

def test_integration_chunk_embeddings_to_retrieval_flow():
    """Verify full end-to-end flow from ChunkEmbeddings through build_index, query embed, and retrieve."""
    chunks = [
        _make_chunk("doc_p1_c0", 1, "Attention mechanisms enable modeling long-range token dependencies."),
        _make_chunk("doc_p2_c0", 2, "Convolutional layers process local spatial features effectively."),
        _make_chunk("doc_p3_c0", 3, "Recurrent architectures suffer from vanishing gradient limitations."),
    ]
    # Synthetic embeddings for chunks
    v_transformer = _make_custom_unit_vector([0.9, 0.4358899, 0.0])
    v_cnn = _make_custom_unit_vector([0.0, 1.0, 0.0])
    v_rnn = _make_custom_unit_vector([0.0, 0.0, 1.0])

    ce = ChunkEmbeddings(chunks=chunks, embeddings=np.stack([v_transformer, v_cnn, v_rnn]))
    vector_store = build_index(ce)

    # Query specifically targeted at attention mechanisms
    # Create mock model that generates a query vector highly aligned with v_transformer
    mock_query_vec = _make_custom_unit_vector([0.95, 0.3122499, 0.0])
    mock_model = MockEmbeddingModel(target_vector=mock_query_vec)

    results = retrieve(
        query="How do attention layers handle token dependencies?",
        vector_store=vector_store,
        top_k=2,
        model=mock_model,
    )

    assert len(results) == 2
    top_result = results[0]

    # Verify complete source tracking
    assert top_result.chunk_id == "doc_p1_c0"
    assert top_result.page_number == 1
    assert "Attention mechanisms enable" in top_result.text
    assert top_result.score > 0.90
    assert top_result.index == 0


def test_integration_retrieve_with_direct_query_vector():
    """Verify retrieve works seamlessly with direct pre-computed numpy query vectors."""
    chunks = [
        _make_chunk("p1_c0", 1, "Alpha"),
        _make_chunk("p2_c0", 2, "Beta"),
    ]
    ce = ChunkEmbeddings(
        chunks=chunks,
        embeddings=np.stack([
            _make_unit_vector(EMBEDDING_DIMENSION, 0),
            _make_unit_vector(EMBEDDING_DIMENSION, 1),
        ]),
    )
    vs = build_index(ce)

    # Directly pass a 1D unit vector
    direct_vec = _make_unit_vector(EMBEDDING_DIMENSION, 1)
    results = retrieve(direct_vec, vs, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "p2_c0"
    assert results[0].page_number == 2
    assert results[0].text == "Beta"
    assert np.isclose(results[0].score, 1.0, atol=1e-3)

