"""Vector similarity retrieval module for DocuMind.

Phase 4: FAISS-based vector similarity search and metadata retrieval.
Uses exact inner-product search (IndexFlatIP) over unit-normalized dense embeddings,
making inner product mathematically equivalent to cosine similarity.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "FAISS is not installed. Please install 'faiss-cpu' via pip to use the retriever module."
    ) from exc

from src.chunker import TextChunk
from src.embeddings import (
    EMBEDDING_DIMENSION,
    ChunkEmbeddings,
    embed_query,
)


class RetrievalError(Exception):
    """Base exception for all retrieval and vector indexing errors."""
    pass


class IndexBuildingError(RetrievalError, ValueError):
    """Raised when FAISS index construction fails due to invalid embeddings or parameters."""
    pass


class SearchError(RetrievalError, ValueError):
    """Raised when query vector validation fails or search execution encounters an error."""
    pass


@dataclass(frozen=True)
class SearchResult:
    """Structured result returned from similarity search retrieval.

    Attributes:
        chunk: The original TextChunk mapped from the retrieved index.
        score: Cosine similarity score (inner product of unit-normalized vectors).
        index: The 0-based integer position of the vector within the FAISS index.
    """

    chunk: TextChunk
    score: float
    index: int

    @property
    def chunk_id(self) -> str:
        """Convenience accessor for the chunk identifier."""
        return self.chunk.chunk_id

    @property
    def page_number(self) -> int:
        """Convenience accessor for the source page number."""
        return self.chunk.page_number

    @property
    def text(self) -> str:
        """Convenience accessor for the chunk text."""
        return self.chunk.text


@dataclass(frozen=True)
class VectorStore:
    """Immutable container pairing a FAISS index with its source TextChunk records.

    Maintains strict 1:1 positional alignment: vector at FAISS index `i` corresponds
    to `chunks[i]`.
    """

    index: Any  # faiss.IndexFlatIP
    chunks: List[TextChunk]
    dimension: int = EMBEDDING_DIMENSION

    @property
    def count(self) -> int:
        """Returns the number of indexed vectors and chunks."""
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """Returns True if the index contains zero vectors."""
        return len(self.chunks) == 0


def build_index(
    chunk_embeddings: ChunkEmbeddings,
    dimension: int = EMBEDDING_DIMENSION,
) -> VectorStore:
    """Constructs a FAISS IndexFlatIP vector store from ChunkEmbeddings.

    Flat inner product search on unit-normalized vectors is mathematically equivalent
    to cosine similarity search. Original chunk ordering and metadata are strictly preserved.

    Args:
        chunk_embeddings: ChunkEmbeddings instance containing document TextChunks
            and their corresponding 2D embedding matrix.
        dimension: Expected vector dimensionality (default: 384).

    Returns:
        VectorStore: Container bundling the FAISS index and original chunks.

    Raises:
        ValueError: If chunk_embeddings is None.
        TypeError: If chunk_embeddings is not an instance of ChunkEmbeddings.
        IndexBuildingError: If embedding dimensions, shapes, values (NaN/Inf), or
            normalization checks fail.
    """
    if chunk_embeddings is None:
        raise ValueError("Input 'chunk_embeddings' cannot be None.")

    if not isinstance(chunk_embeddings, ChunkEmbeddings):
        raise TypeError(
            f"Expected a ChunkEmbeddings instance, got {type(chunk_embeddings).__name__}."
        )

    chunks = list(chunk_embeddings.chunks)
    embeddings = chunk_embeddings.embeddings

    if not isinstance(embeddings, np.ndarray):
        raise TypeError(
            f"Expected embeddings to be a numpy.ndarray, got {type(embeddings).__name__}."
        )

    # 1. Dimensionality and shape validation
    if embeddings.ndim != 2:
        raise IndexBuildingError(
            f"Embeddings array must be 2-dimensional (N, {dimension}), "
            f"got ndim={embeddings.ndim} with shape {embeddings.shape}."
        )

    if embeddings.shape[1] != dimension:
        raise IndexBuildingError(
            f"Embedding dimension mismatch: expected {dimension}, got {embeddings.shape[1]}."
        )

    if len(chunks) != embeddings.shape[0]:
        raise IndexBuildingError(
            f"Chunk count ({len(chunks)}) does not match embedding vector count ({embeddings.shape[0]})."
        )

    # 2. Finite value validation (reject NaN and Inf)
    if not np.all(np.isfinite(embeddings)):
        raise IndexBuildingError(
            "Embeddings contain non-finite values (NaN or Inf). All vector components must be finite numbers."
        )

    # 3. Unit normalization check on non-zero vectors
    if embeddings.shape[0] > 0:
        norms = np.linalg.norm(embeddings, axis=1)
        nonzero_mask = norms > 1e-6
        if np.any(nonzero_mask):
            if not np.all(np.isclose(norms[nonzero_mask], 1.0, atol=1e-3)):
                raise IndexBuildingError(
                    "Embedding vectors must be unit-normalized (L2 norm ~ 1.0) "
                    "for inner product search to equal cosine similarity."
                )

    # 4. Safely ensure contiguous float32 memory layout for FAISS C++ engine
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

    # 5. Build FAISS IndexFlatIP (exact nearest-neighbor inner-product index)
    index = faiss.IndexFlatIP(dimension)

    # 6. Add document vectors if collection is non-empty
    if len(chunks) > 0:
        index.add(vectors)

    return VectorStore(index=index, chunks=chunks, dimension=dimension)


def search_index(
    index: Union[VectorStore, Any],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Searches a FAISS index with a single query vector using inner product similarity.

    Args:
        index: VectorStore or raw faiss.IndexFlatIP instance.
        query_embedding: 1D array of shape (dimension,) or 2D array of shape (1, dimension).
        top_k: Maximum number of nearest neighbors to retrieve (default: 5).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - scores: 1D float32 array of similarity scores sorted descending.
            - indices: 1D int64 array of vector positions matching the scores.
            Returns empty arrays if the index has 0 vectors.

    Raises:
        ValueError: If top_k is not a positive integer (or is a boolean), or query_embedding is None.
        TypeError: If index is neither a VectorStore nor a faiss.Index.
        SearchError: If query dimensionality is wrong, values are non-finite (NaN/Inf),
            or the vector is zero-length/unnormalized.
    """
    # 1. Resolve index properties
    if isinstance(index, VectorStore):
        faiss_index = index.index
        dimension = index.dimension
        ntotal = index.count
    elif hasattr(index, "search") and hasattr(index, "ntotal") and hasattr(index, "d"):
        faiss_index = index
        dimension = index.d
        ntotal = index.ntotal
    else:
        raise TypeError(
            f"Expected 'index' to be a VectorStore or faiss.Index instance, got {type(index).__name__}."
        )

    # 2. Validate top_k (positive integer, reject booleans)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            f"top_k must be a positive integer, got: {top_k!r} (type: {type(top_k).__name__})"
        )

    # 3. Validate query_embedding presence and type
    if query_embedding is None:
        raise ValueError("query_embedding cannot be None.")

    if not isinstance(query_embedding, np.ndarray):
        try:
            query_vec = np.asarray(query_embedding, dtype=np.float32)
        except Exception as exc:
            raise SearchError(f"Cannot convert query_embedding to numpy array: {exc}") from exc
    else:
        query_vec = query_embedding

    # 4. Validate query vector shape and dimension
    if query_vec.size == 0:
        raise SearchError("query_embedding cannot be empty (0 elements).")

    if query_vec.ndim == 1:
        if query_vec.shape[0] != dimension:
            raise SearchError(
                f"Query vector dimension ({query_vec.shape[0]}) does not match index dimension ({dimension})."
            )
        query_2d = query_vec.reshape(1, -1)
    elif query_vec.ndim == 2:
        if query_vec.shape[0] != 1:
            raise SearchError(
                f"Expected a single query vector (1, {dimension}), got shape {query_vec.shape}."
            )
        if query_vec.shape[1] != dimension:
            raise SearchError(
                f"Query vector dimension ({query_vec.shape[1]}) does not match index dimension ({dimension})."
            )
        query_2d = query_vec
    else:
        raise SearchError(
            f"Query vector must be 1D or 2D (shape ({dimension},) or (1, {dimension})), "
            f"got {query_vec.ndim}D array with shape {query_vec.shape}."
        )

    # 5. Check finite numbers (reject NaN and Inf)
    if not np.all(np.isfinite(query_2d)):
        raise SearchError("Query vector contains non-finite values (NaN or Inf).")

    # 6. Check nonzero and unit normalization
    norm = float(np.linalg.norm(query_2d))
    if norm < 1e-6:
        raise SearchError(
            f"Query vector has zero or near-zero L2 norm ({norm:.2e}). "
            "Expected a non-zero, unit-normalized query vector."
        )

    if not np.isclose(norm, 1.0, atol=1e-3):
        raise SearchError(
            f"Query vector must be unit-normalized (L2 norm ~ 1.0). Found norm: {norm:.4f}."
        )

    # 7. Graceful empty index handling and limit requested k to indexed count
    effective_k = min(top_k, ntotal)
    if effective_k == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    # 8. Perform FAISS search
    query_contiguous = np.ascontiguousarray(query_2d, dtype=np.float32)
    distances, indices = faiss_index.search(query_contiguous, effective_k)

    scores = distances[0].astype(np.float32)
    retrieved_indices = indices[0].astype(np.int64)

    # 9. Ensure results are strictly sorted descending by similarity score
    if len(scores) > 1:
        sort_order = np.argsort(-scores)
        scores = scores[sort_order]
        retrieved_indices = retrieved_indices[sort_order]

    return scores, retrieved_indices


def retrieve(
    query: Union[str, np.ndarray],
    vector_store: VectorStore,
    top_k: int = 5,
    model: Optional[Any] = None,
) -> List[SearchResult]:
    """Retrieves top-k most semantically similar text chunks for a query.

    Coordinates query vector generation, FAISS similarity search, and 1:1 metadata mapping
    back to the original TextChunk records.

    Args:
        query: User search query string or pre-computed 1D unit query vector.
        vector_store: VectorStore instance containing the FAISS index and TextChunk records.
        top_k: Maximum number of relevant chunks to retrieve (default: 5).
        model: Optional SentenceTransformer model or mock. If query is a string and model
            is None, the default cached model is used via `embed_query`.

    Returns:
        List[SearchResult]: Ordered list of retrieval results sorted from highest
        similarity to lowest. Returns an empty list if the vector store is empty.

    Raises:
        ValueError: If query is invalid, or top_k <= 0.
        TypeError: If vector_store is not a VectorStore instance.
        RetrievalError: If search or indexing operations fail.
    """
    if not isinstance(vector_store, VectorStore):
        raise TypeError(
            f"Expected 'vector_store' to be a VectorStore instance, got {type(vector_store).__name__}."
        )

    # 1. Process query into a vector representation
    if isinstance(query, str):
        query_vector = embed_query(query, model=model, dimension=vector_store.dimension)
    elif isinstance(query, np.ndarray):
        query_vector = query
    elif hasattr(query, "__iter__"):
        query_vector = np.asarray(query, dtype=np.float32)
    else:
        raise TypeError(
            f"query must be a string or a numpy.ndarray vector, got {type(query).__name__}."
        )

    # 2. Search FAISS index
    scores, indices = search_index(vector_store, query_vector, top_k=top_k)

    # 3. Map returned vector indices back to original TextChunk records
    results: List[SearchResult] = []
    for score, idx in zip(scores, indices):
        chunk = vector_store.chunks[idx]
        results.append(
            SearchResult(
                chunk=chunk,
                score=float(score),
                index=int(idx),
            )
        )

    return results
