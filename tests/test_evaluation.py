"""Unit and integration tests for DocuMind's retrieval evaluation module (Phase 6.2).

All tests run 100% offline using deterministic in-memory vector stores and mock models.
No real Sentence Transformer weights or network calls are required.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.chunker import TextChunk
from src.embeddings import EMBEDDING_DIMENSION, ChunkEmbeddings
from src.retriever import VectorStore, build_index
from src.evaluation import (
    EvaluationReport,
    QueryEvaluationResult,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scripts.evaluate_retrieval import load_dataset, main, parse_k_values


def _make_chunk(chunk_id: str, page_number: int, text: str) -> TextChunk:
    """Helper to create a TextChunk."""
    return TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )


class MockEmbeddingModel:
    """Deterministic offline mock model returning predictable unit vectors."""

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.default_vec = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        self.default_vec[0] = 1.0

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.mapping.get(texts, self.default_vec)
        vectors = []
        for t in texts:
            vectors.append(self.mapping.get(t, self.default_vec))
        return np.array(vectors, dtype=np.float32)


# ==============================================================================
# Precision@K Tests
# ==============================================================================


def test_precision_at_k_perfect_score():
    """All top-k retrieved items are relevant."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1", "c2", "c3"}
    score = precision_at_k(retrieved, relevant, k=3)
    assert score == 1.0
    assert isinstance(score, float)


def test_precision_at_k_zero_hits():
    """No retrieved items are in the relevant set."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c99", "c100"}
    score = precision_at_k(retrieved, relevant, k=3)
    assert score == 0.0


def test_precision_at_k_partial_hits():
    """2 of top 5 retrieved items are relevant."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4", "c9"}
    score = precision_at_k(retrieved, relevant, k=5)
    assert score == pytest.approx(2.0 / 5.0)


def test_precision_at_k_fixed_denominator_short_retrieval():
    """Missing result positions count as non-relevant (fixed-K denominator)."""
    retrieved = ["c1", "c2"]
    relevant = {"c1"}
    # Only 2 items retrieved, but k=5 -> denominator remains 5
    score = precision_at_k(retrieved, relevant, k=5)
    assert score == pytest.approx(1.0 / 5.0)


def test_precision_at_k_duplicate_retrieved_ids_not_double_counted():
    """Duplicate retrieved IDs within top-k must be counted at most once."""
    retrieved = ["c1", "c1", "c2"]
    relevant = {"c1"}
    score = precision_at_k(retrieved, relevant, k=2)
    # top 2 is ["c1", "c1"]. Only 1 unique relevant chunk retrieved.
    assert score == pytest.approx(1.0 / 2.0)


def test_precision_at_k_empty_inputs():
    """Empty retrieved_ids or relevant_ids return 0.0."""
    assert precision_at_k([], {"c1"}, k=5) == 0.0
    assert precision_at_k(["c1"], set(), k=5) == 0.0
    assert precision_at_k([], set(), k=5) == 0.0


def test_precision_at_k_invalid_k():
    """Invalid K values must raise TypeError or ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(["c1"], {"c1"}, k=0)

    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(["c1"], {"c1"}, k=-3)

    with pytest.raises(TypeError, match="must be an integer"):
        precision_at_k(["c1"], {"c1"}, k=True)

    with pytest.raises(TypeError, match="must be an integer"):
        precision_at_k(["c1"], {"c1"}, k=2.5)

    with pytest.raises(TypeError, match="must be an integer"):
        precision_at_k(["c1"], {"c1"}, k="3")


def test_precision_at_k_does_not_mutate_inputs():
    """Input collections must remain unmutated."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c2"]
    retrieved_copy = list(retrieved)
    relevant_copy = list(relevant)

    precision_at_k(retrieved, relevant, k=2)

    assert retrieved == retrieved_copy
    assert relevant == relevant_copy


# ==============================================================================
# Recall@K Tests
# ==============================================================================


def test_recall_at_k_perfect_score():
    """All known relevant chunks are retrieved in top-k."""
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3"}
    score = recall_at_k(retrieved, relevant, k=4)
    assert score == 1.0
    assert isinstance(score, float)


def test_recall_at_k_partial_score():
    """1 of 2 known relevant chunks is retrieved in top-k."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c2", "c9"}
    score = recall_at_k(retrieved, relevant, k=3)
    assert score == pytest.approx(1.0 / 2.0)


def test_recall_at_k_zero_hits():
    """No relevant chunks retrieved."""
    retrieved = ["c1", "c2"]
    relevant = {"c3", "c4"}
    score = recall_at_k(retrieved, relevant, k=2)
    assert score == 0.0


def test_recall_at_k_empty_ground_truth_raises_value_error():
    """Empty ground-truth sets must raise ValueError (cannot silently return misleading score)."""
    with pytest.raises(ValueError, match="relevant_ids cannot be empty"):
        recall_at_k(["c1", "c2"], set(), k=2)

    with pytest.raises(ValueError, match="relevant_ids cannot be empty"):
        recall_at_k(["c1", "c2"], [], k=2)


def test_recall_at_k_empty_retrieved_returns_zero():
    """Empty retrieved list with valid relevant set returns 0.0."""
    score = recall_at_k([], {"c1", "c2"}, k=5)
    assert score == 0.0


def test_recall_at_k_duplicate_relevant_ids_deduplicated():
    """Duplicate IDs in relevant_ids sequence do not artificially inflate the denominator."""
    retrieved = ["c1"]
    relevant = ["c1", "c1"]  # 1 unique chunk
    score = recall_at_k(retrieved, relevant, k=1)
    assert score == 1.0


def test_recall_at_k_invalid_k():
    """Invalid K values must raise TypeError or ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["c1"], {"c1"}, k=0)

    with pytest.raises(TypeError, match="must be an integer"):
        recall_at_k(["c1"], {"c1"}, k=False)


# ==============================================================================
# Reciprocal Rank (RR) Tests
# ==============================================================================


def test_reciprocal_rank_first_result():
    """First result is relevant -> RR = 1.0."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1"}
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_reciprocal_rank_second_result():
    """Second result is first relevant -> RR = 0.5."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c2"}
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_third_result():
    """Third result is first relevant -> RR = 1/3."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c3"}
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0 / 3.0)


def test_reciprocal_rank_no_relevant_found():
    """No relevant result found -> RR = 0.0."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c99"}
    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_reciprocal_rank_empty_inputs():
    """Empty retrieved or relevant list returns 0.0."""
    assert reciprocal_rank([], {"c1"}) == 0.0
    assert reciprocal_rank(["c1"], set()) == 0.0
    assert reciprocal_rank([], set()) == 0.0


def test_reciprocal_rank_duplicate_retrieved_handling():
    """Duplicate retrieved items before the first match reflect their 1-based index."""
    retrieved = ["c_irrelevant", "c_irrelevant", "c_relevant"]
    relevant = {"c_relevant"}
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0 / 3.0)


# ==============================================================================
# Mean Reciprocal Rank (MRR) Tests
# ==============================================================================


def test_mean_reciprocal_rank_computation():
    """MRR is the arithmetic mean of individual reciprocal ranks."""
    rankings = [
        ["c1", "c2"],  # RR = 1.0
        ["c2", "c1"],  # RR = 0.5
        ["c3", "c4"],  # RR = 0.0
    ]
    relevants = [
        {"c1"},
        {"c1"},
        {"c1"},
    ]
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    mrr = mean_reciprocal_rank(rankings, relevants)
    assert mrr == pytest.approx(0.5)


def test_mean_reciprocal_rank_empty_sequence():
    """Empty query sequence returns 0.0."""
    assert mean_reciprocal_rank([], []) == 0.0


def test_mean_reciprocal_rank_length_mismatch():
    """Mismatched rankings and relevant_ids lengths must raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        mean_reciprocal_rank([["c1"]], [{"c1"}, {"c2"}])


# ==============================================================================
# Evaluation Report & Dataclass Tests
# ==============================================================================


def test_query_evaluation_result_serialization():
    """QueryEvaluationResult correctly serializes to dictionary."""
    res = QueryEvaluationResult(
        query_id="q1",
        question="What is DocuMind?",
        retrieved_chunk_ids=["c1", "c2"],
        relevant_chunk_ids={"c1"},
        precision_at_k={1: 1.0, 3: 0.3333},
        recall_at_k={1: 1.0, 3: 1.0},
        reciprocal_rank=1.0,
        document_id="doc1",
        notes="sample query",
    )
    d = res.to_dict()
    assert d["query_id"] == "q1"
    assert d["question"] == "What is DocuMind?"
    assert d["relevant_chunk_ids"] == ["c1"]
    assert d["reciprocal_rank"] == 1.0
    assert d["precision_at_k"]["1"] == 1.0


def test_evaluation_report_format_text():
    """EvaluationReport produces readable ASCII text without unicode issues."""
    res = QueryEvaluationResult(
        query_id="q1",
        question="What is DocuMind?",
        retrieved_chunk_ids=["c1"],
        relevant_chunk_ids={"c1"},
        precision_at_k={1: 1.0},
        recall_at_k={1: 1.0},
        reciprocal_rank=1.0,
    )
    report = EvaluationReport(
        results=[res],
        mean_reciprocal_rank=1.0,
        mean_precision_at_k={1: 1.0},
        mean_recall_at_k={1: 1.0},
        k_values=[1],
        total_queries=1,
    )
    text = report.format_text()
    assert "DOCUMIND RETRIEVAL BENCHMARK REPORT" in text
    assert "Mean Reciprocal Rank (MRR): 1.0000" in text
    assert "Precision@1 = 1.0000" in text


# ==============================================================================
# End-to-End evaluate_retrieval Integration Tests
# ==============================================================================


def test_evaluate_retrieval_with_mock_vector_store():
    """End-to-end evaluation using an offline FAISS VectorStore and deterministic mock model."""
    chunks = [
        _make_chunk("c1", 1, "DocuMind optimizer is SGD with learning rate 0.01"),
        _make_chunk("c2", 2, "Hardware setup uses NVIDIA A100 GPUs"),
        _make_chunk("c3", 3, "Baseline datasets GLUE and SQuAD were evaluated"),
    ]

    # Assign distinct 1-hot vectors
    v1 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    v1[1] = 1.0
    v2 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    v2[2] = 1.0
    v3 = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    v3[3] = 1.0

    mock_model = MockEmbeddingModel({
        "DocuMind optimizer is SGD with learning rate 0.01": v1,
        "Hardware setup uses NVIDIA A100 GPUs": v2,
        "Baseline datasets GLUE and SQuAD were evaluated": v3,
        "What optimizer was used?": v1,  # Matches c1
        "What GPUs were used?": v2,      # Matches c2
    })

    chunk_embeddings = ChunkEmbeddings(
        chunks=chunks,
        embeddings=np.array([v1, v2, v3], dtype=np.float32),
    )
    vstore = build_index(chunk_embeddings)

    dataset = [
        {
            "id": "q1",
            "question": "What optimizer was used?",
            "relevant_chunk_ids": ["c1"],
        },
        {
            "id": "q2",
            "question": "What GPUs were used?",
            "relevant_chunk_ids": ["c2"],
        },
    ]

    report = evaluate_retrieval(
        vector_store=vstore,
        dataset=dataset,
        k_values=[1, 3],
        embedding_model=mock_model,
    )

    assert report.total_queries == 2
    assert report.mean_reciprocal_rank == 1.0
    assert report.mean_precision_at_k[1] == 1.0
    assert report.mean_recall_at_k[1] == 1.0


def test_evaluate_retrieval_empty_dataset():
    """evaluate_retrieval with an empty query dataset returns an empty report."""
    vstore = VectorStore(index=None, chunks=[])
    report = evaluate_retrieval(vstore, dataset=[], k_values=[1, 5])
    assert report.total_queries == 0
    assert report.mean_reciprocal_rank == 0.0
    assert report.mean_precision_at_k[1] == 0.0


def test_evaluate_retrieval_invalid_dataset_items():
    """Malformed dataset items raise descriptive errors."""
    vstore = VectorStore(index=None, chunks=[])

    with pytest.raises(TypeError, match="must be dictionaries"):
        evaluate_retrieval(vstore, dataset=["not a dict"])

    with pytest.raises(ValueError, match="missing a valid 'question'"):
        evaluate_retrieval(vstore, dataset=[{"id": "q1"}])

    with pytest.raises(ValueError, match="missing a 'relevant_chunk_ids'"):
        evaluate_retrieval(vstore, dataset=[{"question": "How?"}])


# ==============================================================================
# Dataset File & Schema Validation Tests
# ==============================================================================


def test_dataset_json_exists_and_conforms_to_schema():
    """evaluation/dataset.json exists, is valid JSON, and has all required fields."""
    dataset_path = Path("evaluation/dataset.json")
    assert dataset_path.exists(), "evaluation/dataset.json must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "queries" in data
    assert isinstance(data["queries"], list)
    assert len(data["queries"]) > 0

    for idx, q in enumerate(data["queries"]):
        assert "question" in q, f"Query at index {idx} missing 'question'"
        assert isinstance(q["question"], str) and q["question"].strip()
        assert "relevant_chunk_ids" in q, f"Query at index {idx} missing 'relevant_chunk_ids'"
        assert isinstance(q["relevant_chunk_ids"], list)
        assert len(q["relevant_chunk_ids"]) > 0


def test_dataset_schema_json_exists_and_is_valid():
    """evaluation/dataset.schema.json exists and is valid JSON."""
    schema_path = Path("evaluation/dataset.schema.json")
    assert schema_path.exists(), "evaluation/dataset.schema.json must exist"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema.get("type") == "object"
    assert "queries" in schema.get("properties", {})


# ==============================================================================
# Script Runner CLI Tests
# ==============================================================================


def test_parse_k_values_valid():
    """Valid comma-separated strings parse into sorted unique integer lists."""
    assert parse_k_values("1,3,5") == [1, 3, 5]
    assert parse_k_values(" 5, 1, 3 , 3 ") == [1, 3, 5]
    assert parse_k_values("10") == [10]


def test_parse_k_values_invalid():
    """Invalid K strings raise ValueError."""
    with pytest.raises(ValueError, match="Invalid --k"):
        parse_k_values("0,1,3")

    with pytest.raises(ValueError, match="Invalid --k"):
        parse_k_values("-5")

    with pytest.raises(ValueError, match="Invalid --k"):
        parse_k_values("abc")

    with pytest.raises(ValueError, match="Invalid --k"):
        parse_k_values("")


def test_load_dataset_missing_file():
    """load_dataset on non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("nonexistent/dataset.json"))


def test_load_dataset_invalid_format(tmp_path):
    """load_dataset on invalid JSON raises ValueError."""
    invalid_file = tmp_path / "bad.json"
    invalid_file.write_text('{"no_queries": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid dataset format"):
        load_dataset(invalid_file)


def test_runner_main_execution_mocked(tmp_path, monkeypatch):
    """Test CLI main() runner executes cleanly and exports JSON when requested."""
    from unittest.mock import MagicMock, patch

    # Mock sample vector store to avoid loading sentence-transformers
    mock_store = MagicMock(spec=VectorStore)
    mock_store.count = 3
    mock_store.dimension = EMBEDDING_DIMENSION

    # Create dummy dataset
    dummy_dataset_file = tmp_path / "test_ds.json"
    dummy_dataset_file.write_text(
        json.dumps({
            "metadata": {"is_placeholder_dataset": True},
            "queries": [
                {
                    "id": "q1",
                    "question": "Sample question?",
                    "relevant_chunk_ids": ["c1"],
                }
            ],
        }),
        encoding="utf-8",
    )

    output_json_file = tmp_path / "out.json"

    with patch("scripts.evaluate_retrieval.create_sample_vector_store", return_value=mock_store), \
         patch("scripts.evaluate_retrieval.evaluate_retrieval") as mock_eval:

        mock_eval.return_value = EvaluationReport(
            results=[
                QueryEvaluationResult(
                    query_id="q1",
                    question="Sample question?",
                    retrieved_chunk_ids=["c1"],
                    relevant_chunk_ids={"c1"},
                    precision_at_k={1: 1.0},
                    recall_at_k={1: 1.0},
                    reciprocal_rank=1.0,
                )
            ],
            mean_reciprocal_rank=1.0,
            mean_precision_at_k={1: 1.0},
            mean_recall_at_k={1: 1.0},
            k_values=[1],
            total_queries=1,
        )

        exit_code = main([
            "--dataset", str(dummy_dataset_file),
            "--k", "1",
            "--output-json", str(output_json_file),
        ])

        assert exit_code == 0
        assert output_json_file.exists()
        out_data = json.loads(output_json_file.read_text(encoding="utf-8"))
        assert out_data["summary"]["mean_reciprocal_rank"] == 1.0

