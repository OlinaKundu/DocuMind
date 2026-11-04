"""Unit and integration tests for DocuMind's RAG pipeline module (Phase 5.3).

Verifies the end-to-end integration of vector retrieval, prompt building, and LLM generation.
All default tests run 100% offline using in-memory mocked models and vector stores.
"""

import hashlib
import os
from typing import Any, List, Optional, Sequence
import httpx
import numpy as np
import pytest
import groq

from src.chunker import TextChunk
from src.embeddings import EMBEDDING_DIMENSION, ChunkEmbeddings
from src.retriever import SearchResult, VectorStore, build_index
from src.prompt_builder import NO_EVIDENCE_MESSAGE
from src.llm import (
    DEFAULT_MODEL,
    LLMAPIError,
    LLMAuthenticationError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    generate_response,
)
from src.rag_pipeline import RAGResponse, answer_question


# ==============================================================================
# Test Helpers and Mocks
# ==============================================================================

class MockMessage:
    def __init__(self, content: Optional[str]):
        self.content = content


class MockChoice:
    def __init__(self, message: Optional[MockMessage]):
        self.message = message


class MockChatCompletion:
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices


class MockCompletionsResource:
    def __init__(self, response: Any = None, side_effect: Optional[Exception] = None):
        self.response = response
        self.side_effect = side_effect
        self.last_call_params: dict[str, Any] = {}

    def create(self, **kwargs) -> Any:
        self.last_call_params = kwargs
        if self.side_effect is not None:
            raise self.side_effect
        return self.response


class MockGroqClient:
    def __init__(self, response: Any = None, side_effect: Optional[Exception] = None):
        self.completions = MockCompletionsResource(response=response, side_effect=side_effect)
        self.chat = self


class MockEmbeddingModel:
    """Deterministic offline mock model returning predictable unit vectors."""

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
            s_lower = s.lower()
            if "dataset" in s_lower or "cifar" in s_lower:
                vec = _make_unit_vector(0, self.dimension)
            elif "optimizer" in s_lower or "adamw" in s_lower:
                vec = _make_unit_vector(1, self.dimension)
            elif "accuracy" in s_lower:
                vec = _make_unit_vector(2, self.dimension)
            else:
                seed = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 100000 + 1
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.dimension, dtype=np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def _make_unit_vector(active_idx: int = 0, dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[active_idx % dim] = 1.0
    return vec


@pytest.fixture
def sample_vector_store() -> VectorStore:
    """Creates a deterministic 3-chunk vector store for pipeline testing."""
    chunks = [
        TextChunk(
            chunk_id="p1_c0",
            page_number=1,
            text="The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes.",
            start_char=0,
            end_char=73,
        ),
        TextChunk(
            chunk_id="p2_c0",
            page_number=2,
            text="The model was trained using the AdamW optimizer with learning rate 0.001.",
            start_char=0,
            end_char=74,
        ),
        TextChunk(
            chunk_id="p3_c0",
            page_number=3,
            text="Final top-1 accuracy achieved on the test partition was 94.2%.",
            start_char=0,
            end_char=63,
        ),
    ]
    vectors = np.stack([_make_unit_vector(i) for i in range(3)])
    ce = ChunkEmbeddings(chunks=chunks, embeddings=vectors)
    return build_index(ce)


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    return MockEmbeddingModel()


# ==============================================================================
# 1. Pipeline Execution & RAGResponse Tests
# ==============================================================================

def test_successful_end_to_end_rag_execution(sample_vector_store, mock_embedding_model):
    """Verify answer_question coordinates retrieval, prompt building, and LLM generation."""
    expected_answer = "The model was trained with AdamW [Page 2]."
    mock_llm_response = MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content=expected_answer))]
    )
    mock_llm = MockGroqClient(response=mock_llm_response)

    response = answer_question(
        question="What optimizer was used?",
        vector_store=sample_vector_store,
        top_k=2,
        llm_client=mock_llm,
        embedding_model=mock_embedding_model,
    )

    assert isinstance(response, RAGResponse)
    assert response.answer == expected_answer
    assert response.question == "What optimizer was used?"
    assert response.model == DEFAULT_MODEL
    assert len(response.sources) == 2
    assert response.source_count == 2
    assert response.has_sources is True


def test_retrieval_results_and_messages_passed_to_llm(sample_vector_store, mock_embedding_model):
    """Verify retrieved passages are properly formatted into chat messages sent to Groq."""
    mock_llm_response = MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content="Response text."))]
    )
    mock_llm = MockGroqClient(response=mock_llm_response)

    question = "Describe the dataset."
    answer_question(
        question=question,
        vector_store=sample_vector_store,
        top_k=1,
        llm_client=mock_llm,
        embedding_model=mock_embedding_model,
    )

    call_params = mock_llm.completions.last_call_params
    assert call_params["model"] == DEFAULT_MODEL
    assert call_params["temperature"] == 0.0

    messages = call_params["messages"]
    assert len(messages) == 2

    # Check system message
    assert messages[0]["role"] == "system"
    assert "GROUNDING RULES" in messages[0]["content"]
    assert "DOCUMENT SECURITY RULES" in messages[0]["content"]

    # Check user message
    assert messages[1]["role"] == "user"
    assert f"QUESTION:\n{question}" in messages[1]["content"]
    assert "DOCUMENT CONTEXT:" in messages[1]["content"]
    assert "[Source 1 | Page 1 | Chunk p1_c0]" in messages[1]["content"]
    assert "CIFAR-10" in messages[1]["content"]


def test_source_metadata_preservation_on_rag_response(sample_vector_store, mock_embedding_model):
    """Verify page numbers, chunk IDs, text, and scores remain accessible on response.sources."""
    mock_llm = MockGroqClient(
        response=MockChatCompletion(choices=[MockChoice(message=MockMessage(content="Grounded answer."))])
    )

    response = answer_question(
        question="What was the accuracy?",
        vector_store=sample_vector_store,
        top_k=3,
        llm_client=mock_llm,
        embedding_model=mock_embedding_model,
    )

    assert len(response.sources) == 3
    for s in response.sources:
        assert isinstance(s, SearchResult)
        assert isinstance(s.page_number, int)
        assert s.page_number in {1, 2, 3}
        assert isinstance(s.chunk_id, str)
        assert s.chunk_id.startswith("p")
        assert isinstance(s.text, str)
        assert isinstance(s.score, float)


def test_custom_model_configuration_preserved(sample_vector_store, mock_embedding_model):
    """Verify custom model argument is passed to LLM client and recorded in RAGResponse."""
    custom_model = "llama-3.3-70b-versatile"
    mock_llm = MockGroqClient(
        response=MockChatCompletion(choices=[MockChoice(message=MockMessage(content="Custom model answer."))])
    )

    response = answer_question(
        question="What was the test accuracy?",
        vector_store=sample_vector_store,
        model=custom_model,
        llm_client=mock_llm,
        embedding_model=mock_embedding_model,
    )

    assert response.model == custom_model
    assert mock_llm.completions.last_call_params["model"] == custom_model


# ==============================================================================
# 2. Empty Retrieval Handling
# ==============================================================================

def test_empty_retrieval_handling(mock_embedding_model):
    """Verify pipeline handles empty vector stores gracefully without fabricating context."""
    empty_ce = ChunkEmbeddings(
        chunks=[],
        embeddings=np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32),
    )
    empty_store = build_index(empty_ce)

    insufficient_answer = "The available document evidence is insufficient to answer the question."
    mock_llm = MockGroqClient(
        response=MockChatCompletion(choices=[MockChoice(message=MockMessage(content=insufficient_answer))])
    )

    response = answer_question(
        question="What is the learning rate?",
        vector_store=empty_store,
        llm_client=mock_llm,
        embedding_model=mock_embedding_model,
    )

    assert isinstance(response, RAGResponse)
    assert response.answer == insufficient_answer
    assert response.sources == []
    assert response.source_count == 0
    assert response.has_sources is False

    # Check that LLM received the empty evidence notice
    messages = mock_llm.completions.last_call_params["messages"]
    assert NO_EVIDENCE_MESSAGE in messages[1]["content"]


# ==============================================================================
# 3. Input Validation & Error Propagation
# ==============================================================================

def test_invalid_questions_rejected(sample_vector_store):
    """Verify invalid questions (None, non-string, empty, whitespace) raise TypeError or ValueError."""
    # None
    with pytest.raises(TypeError, match="question must be a string"):
        answer_question(None, sample_vector_store)

    # Integer
    with pytest.raises(TypeError, match="question must be a string"):
        answer_question(12345, sample_vector_store)

    # Empty string
    with pytest.raises(ValueError, match="question cannot be empty or contain only whitespace"):
        answer_question("", sample_vector_store)

    # Whitespace-only string
    with pytest.raises(ValueError, match="question cannot be empty or contain only whitespace"):
        answer_question("   \n\t  ", sample_vector_store)


def test_invalid_vector_store_rejected():
    """Verify invalid vector_store parameter raises TypeError."""
    with pytest.raises(TypeError, match="Expected 'vector_store' to be a VectorStore"):
        answer_question("Valid question?", None)

    with pytest.raises(TypeError, match="Expected 'vector_store' to be a VectorStore"):
        answer_question("Valid question?", "invalid_store_string")


def test_invalid_top_k_propagates_error(sample_vector_store):
    """Verify invalid top_k values (0, negative, bools) raise ValueError."""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question("Valid question?", sample_vector_store, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question("Valid question?", sample_vector_store, top_k=-2)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question("Valid question?", sample_vector_store, top_k=True)


def test_groq_authentication_error_propagates(sample_vector_store, mock_embedding_model):
    """Verify Groq authentication failure propagates as LLMAuthenticationError."""
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(401, request=req)
    mock_llm = MockGroqClient(
        side_effect=groq.AuthenticationError("Invalid API key", response=resp, body=None)
    )

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        answer_question(
            question="What is the loss?",
            vector_store=sample_vector_store,
            llm_client=mock_llm,
            embedding_model=mock_embedding_model,
        )


def test_groq_rate_limit_error_propagates(sample_vector_store, mock_embedding_model):
    """Verify Groq rate limit failure propagates as LLMRateLimitError."""
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(429, request=req)
    mock_llm = MockGroqClient(
        side_effect=groq.RateLimitError("Rate limit exceeded", response=resp, body=None)
    )

    with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
        answer_question(
            question="What is the loss?",
            vector_store=sample_vector_store,
            llm_client=mock_llm,
            embedding_model=mock_embedding_model,
        )


def test_groq_network_error_propagates(sample_vector_store, mock_embedding_model):
    """Verify Groq network connection error propagates as LLMAPIError."""
    req = httpx.Request("POST", "https://api.groq.com")
    mock_llm = MockGroqClient(side_effect=groq.APIConnectionError(request=req))

    with pytest.raises(LLMAPIError, match="Network error"):
        answer_question(
            question="What is the loss?",
            vector_store=sample_vector_store,
            llm_client=mock_llm,
            embedding_model=mock_embedding_model,
        )


def test_groq_empty_response_propagates(sample_vector_store, mock_embedding_model):
    """Verify empty LLM response propagates as LLMEmptyResponseError."""
    mock_llm = MockGroqClient(
        response=MockChatCompletion(choices=[MockChoice(message=MockMessage(content="   "))])
    )

    with pytest.raises(LLMEmptyResponseError, match="empty or whitespace-only"):
        answer_question(
            question="What is the loss?",
            vector_store=sample_vector_store,
            llm_client=mock_llm,
            embedding_model=mock_embedding_model,
        )


def test_existing_generate_response_compatibility():
    """Verify existing generate_response() function remains backward compatible."""
    mock_llm = MockGroqClient(
        response=MockChatCompletion(choices=[MockChoice(message=MockMessage(content="Single prompt answer."))])
    )
    ans = generate_response("Test prompt", client=mock_llm)
    assert ans == "Single prompt answer."


# ==============================================================================
# 4. Optional Real API Integration Smoke Test
# ==============================================================================

@pytest.mark.real_api
def test_real_rag_pipeline_smoke_test(sample_vector_store):
    """Smoke test running full pipeline against live Groq API.

    Requires an active GROQ_API_KEY. Runs only when explicitly invoked:
        pytest -v -m real_api tests/test_rag_pipeline.py
    """
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key or not key.strip():
        pytest.skip("GROQ_API_KEY not configured. Skipping live RAG pipeline test.")

    response = answer_question(
        question="What was the final test accuracy achieved?",
        vector_store=sample_vector_store,
        top_k=3,
    )

    assert isinstance(response, RAGResponse)
    assert len(response.answer.strip()) > 0
    assert len(response.sources) == 3
    assert "94.2" in response.answer or "Page 3" in response.answer
