"""Embedding generation module for DocuMind.

Responsible for generating dense, unit-normalized vector representations
of document chunks and user search queries using Sentence Transformers.
"""

from dataclasses import dataclass
import functools
from typing import Any, List, Optional, Sequence
import numpy as np
from src.chunker import TextChunk

# Centralized configuration
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384


class EmbeddingError(Exception):
    """Base exception for all embedding generation and model loading errors."""
    pass


class ModelLoadingError(EmbeddingError):
    """Exception raised when the Sentence Transformer model cannot be loaded or downloaded."""
    pass


@dataclass(frozen=True)
class ChunkEmbeddings:
    """Pairs document text chunks with their corresponding dense vector embeddings.

    Maintains strict 1:1 index alignment: embeddings[i] is the vector for chunks[i].
    Supports tuple unpacking:
        chunks, vectors = embed_chunks(...)
    """

    chunks: List[TextChunk]
    embeddings: np.ndarray  # Shape: (N, EMBEDDING_DIMENSION), dtype: float32

    def __iter__(self):
        """Allows unpacking as (chunks, embeddings)."""
        return iter((self.chunks, self.embeddings))

    @property
    def count(self) -> int:
        """Returns the number of chunk-embedding pairs."""
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        if self.embeddings.ndim == 2:
            return self.embeddings.shape[1]
        return EMBEDDING_DIMENSION


@functools.lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> Any:
    """Loads and caches the Sentence Transformer model.

    Uses LRU caching to guarantee that the model is loaded lazily and once in memory,
    rather than instantiated on every function call.

    Args:
        model_name: The Hugging Face model identifier or local directory path.
            Default: "sentence-transformers/all-MiniLM-L6-v2".
        device: Device to load the model on ("cpu", "cuda", etc.). Defaults to None (auto).

    Returns:
        SentenceTransformer: The loaded model instance.

    Raises:
        ModelLoadingError: If the model cannot be found or downloaded.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise ModelLoadingError(
            f"Failed to load embedding model '{model_name}'. "
            "Note: If this is the first time running this model, active internet access is required "
            f"to download model weights from Hugging Face Hub. Underlying error: {exc}"
        ) from exc


def embed_texts(
    texts: Sequence[str],
    model: Optional[Any] = None,
    dimension: int = EMBEDDING_DIMENSION,
) -> np.ndarray:
    """Encodes a sequence of text strings into unit-normalized dense vectors.

    Args:
        texts: A sequence of strings to embed.
        model: An optional loaded SentenceTransformer model or mock. If None,
            `get_embedding_model()` is used.
        dimension: Expected vector dimension (default: 384).

    Returns:
        np.ndarray: A 2D array of shape (N, dimension) with dtype float32,
        where each row vector has unit L2 norm (||v||_2 = 1.0).

    Raises:
        ValueError: If `texts` is None.
        TypeError: If `texts` is a single string or contains non-string items.
        EmbeddingError: If model encoding fails.
    """
    if texts is None:
        raise ValueError("Input 'texts' cannot be None. Expected a sequence of strings.")

    # Guard against passing a single string instead of a sequence
    if isinstance(texts, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of strings (e.g. list or tuple), got a single {type(texts).__name__}. "
            "Pass [text] instead."
        )

    # Validate that every element is a string
    text_list = list(texts)
    for idx, item in enumerate(text_list):
        if not isinstance(item, str):
            raise TypeError(
                f"All elements in 'texts' must be strings. Found {type(item).__name__} at index {idx}."
            )

    # Handle empty input safely: return correctly shaped (0, dimension) array
    if len(text_list) == 0:
        return np.zeros((0, dimension), dtype=np.float32)

    # Resolve model
    active_model = model if model is not None else get_embedding_model()

    try:
        if hasattr(active_model, "encode"):
            raw_embeddings = active_model.encode(
                text_list,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        elif callable(active_model):
            raw_embeddings = active_model(text_list)
        else:
            raise TypeError(
                f"Model object of type '{type(active_model).__name__}' does not provide an 'encode' method or callable interface."
            )
    except Exception as exc:
        if isinstance(exc, (TypeError, ValueError, EmbeddingError)):
            raise
        raise EmbeddingError(f"Error during text embedding generation: {exc}") from exc

    # Convert to float32 NumPy array
    vectors = np.asarray(raw_embeddings, dtype=np.float32)

    # Validate output dimensions
    if vectors.ndim == 1 and len(text_list) == 1:
        vectors = vectors.reshape(1, -1)
    elif vectors.ndim != 2:
        raise EmbeddingError(
            f"Expected 2D embedding array from model, got shape {vectors.shape}."
        )

    if vectors.shape[0] != len(text_list):
        raise EmbeddingError(
            f"Output vector count ({vectors.shape[0]}) does not match input text count ({len(text_list)})."
        )

    # Enforce unit L2 normalization: ||v|| = 1.0 (vital for cosine similarity search in FAISS)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms[norms == 0] = 1.0
    normalized_vectors = (vectors / norms).astype(np.float32)

    return normalized_vectors


def embed_chunks(
    chunks: Sequence[TextChunk],
    model: Optional[Any] = None,
    dimension: int = EMBEDDING_DIMENSION,
) -> ChunkEmbeddings:
    """Generates dense vector embeddings for an ordered sequence of TextChunk objects.

    Maintains exact 1:1 alignment between chunks and output vectors without altering
    chunk metadata or reordering records.

    Args:
        chunks: Sequence of TextChunk objects (from `chunker.py`).
        model: Optional embedding model or mock. If None, `get_embedding_model()` is used.
        dimension: Expected vector dimension (default: 384).

    Returns:
        ChunkEmbeddings: A container pairing the input chunks with their (N, dimension)
        float32 embedding matrix.

    Raises:
        ValueError: If `chunks` is None.
        TypeError: If any item in `chunks` is not a TextChunk instance.
    """
    if chunks is None:
        raise ValueError("Input 'chunks' cannot be None. Expected a sequence of TextChunk objects.")

    chunk_list = list(chunks)

    # Handle empty input safely
    if len(chunk_list) == 0:
        return ChunkEmbeddings(
            chunks=[],
            embeddings=np.zeros((0, dimension), dtype=np.float32),
        )

    # Validate that every element is a TextChunk
    for idx, item in enumerate(chunk_list):
        if not isinstance(item, TextChunk):
            raise TypeError(
                f"All elements in 'chunks' must be TextChunk instances. Found {type(item).__name__} at index {idx}."
            )

    # Extract text from chunks preserving strict sequential ordering
    chunk_texts = [c.text for c in chunk_list]

    # Generate embeddings
    embeddings = embed_texts(chunk_texts, model=model, dimension=dimension)

    return ChunkEmbeddings(chunks=chunk_list, embeddings=embeddings)


def embed_query(
    query: str,
    model: Optional[Any] = None,
    dimension: int = EMBEDDING_DIMENSION,
) -> np.ndarray:
    """Generates a 1D unit-normalized vector for a user search query.

    Uses the exact same embedding model, dimension, and normalization rules as document
    chunks, ensuring query and document vectors reside in the identical semantic vector space.

    Args:
        query: The user query string.
        model: Optional embedding model or mock.
        dimension: Expected vector dimension (default: 384).

    Returns:
        np.ndarray: A 1D array of shape (dimension,) with dtype float32 and unit norm.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty or contains only whitespace.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query cannot be empty or contain only whitespace.")

    # Embed as single-item sequence, then flatten to 1D
    vector_2d = embed_texts([cleaned_query], model=model, dimension=dimension)
    return vector_2d[0]
