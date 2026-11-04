"""Retrieval evaluation and benchmarking module for DocuMind.

Phase 6.2: Measures vector retrieval quality independently of LLM generation using
standard Information Retrieval (IR) metrics:
- Precision@K (P@K)
- Recall@K (R@K)
- Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from src.retriever import SearchResult, VectorStore, retrieve


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculates Precision@K for a single query.

    Formula:
        Precision@K = (number of distinct relevant chunks in top K) / K

    Missing result positions (when len(retrieved_ids) < K) count as non-relevant,
    preserving the standard fixed-K denominator. Duplicate retrieved IDs are counted
    at most once.

    Args:
        retrieved_ids: Ordered sequence of retrieved chunk IDs from highest to lowest rank.
        relevant_ids: Set or sequence of ground-truth relevant chunk IDs.
        k: Positive integer cutoff rank.

    Returns:
        float: Precision@K in the range [0.0, 1.0].

    Raises:
        TypeError: If k is a boolean or not an integer, or inputs are not sequences/sets.
        ValueError: If k <= 0.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got: {k!r} ({type(k).__name__})")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got: {k}")

    if not isinstance(retrieved_ids, (list, tuple, Sequence)):
        raise TypeError(f"retrieved_ids must be a sequence, got: {type(retrieved_ids).__name__}")
    if not isinstance(relevant_ids, (set, frozenset, list, tuple, Sequence)):
        raise TypeError(f"relevant_ids must be a set or sequence, got: {type(relevant_ids).__name__}")

    if not retrieved_ids or not relevant_ids:
        return 0.0

    # Ensure no input mutation
    relevant_set: Set[str] = set(relevant_ids)
    top_k_ids = list(retrieved_ids[:k])

    # Count distinct relevant chunks retrieved within the top K
    relevant_retrieved = len(set(top_k_ids) & relevant_set)
    return float(relevant_retrieved) / float(k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculates Recall@K for a single query.

    Formula:
        Recall@K = (number of distinct relevant chunks in top K) / (total distinct relevant chunks)

    Args:
        retrieved_ids: Ordered sequence of retrieved chunk IDs from highest to lowest rank.
        relevant_ids: Set or sequence of ground-truth relevant chunk IDs. Must not be empty.
        k: Positive integer cutoff rank.

    Returns:
        float: Recall@K in the range [0.0, 1.0].

    Raises:
        TypeError: If k is a boolean or not an integer, or inputs are not sequences/sets.
        ValueError: If k <= 0, or if relevant_ids is empty (recall is undefined for empty ground truth).
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got: {k!r} ({type(k).__name__})")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got: {k}")

    if not isinstance(retrieved_ids, (list, tuple, Sequence)):
        raise TypeError(f"retrieved_ids must be a sequence, got: {type(retrieved_ids).__name__}")
    if not isinstance(relevant_ids, (set, frozenset, list, tuple, Sequence)):
        raise TypeError(f"relevant_ids must be a set or sequence, got: {type(relevant_ids).__name__}")

    relevant_set: Set[str] = set(relevant_ids)
    total_relevant = len(relevant_set)

    if total_relevant == 0:
        raise ValueError(
            "relevant_ids cannot be empty. Recall@K is undefined when total relevant chunks is 0."
        )

    if not retrieved_ids:
        return 0.0

    top_k_ids = list(retrieved_ids[:k])
    relevant_retrieved = len(set(top_k_ids) & relevant_set)
    return float(relevant_retrieved) / float(total_relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
) -> float:
    """Calculates Reciprocal Rank (RR) for a single query ranking.

    Formula:
        RR = 1.0 / rank of first relevant result (1-based index)
        Returns 0.0 if no relevant result is found.

    Args:
        retrieved_ids: Ordered sequence of retrieved chunk IDs from highest to lowest rank.
        relevant_ids: Set or sequence of ground-truth relevant chunk IDs.

    Returns:
        float: Reciprocal rank in the range [0.0, 1.0]. Returns 0.0 if retrieved_ids or
            relevant_ids is empty, or if no relevant chunk appears in retrieved_ids.

    Raises:
        TypeError: If inputs are not sequences/sets.
    """
    if not isinstance(retrieved_ids, (list, tuple, Sequence)):
        raise TypeError(f"retrieved_ids must be a sequence, got: {type(retrieved_ids).__name__}")
    if not isinstance(relevant_ids, (set, frozenset, list, tuple, Sequence)):
        raise TypeError(f"relevant_ids must be a set or sequence, got: {type(relevant_ids).__name__}")

    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set: Set[str] = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / float(rank)

    return 0.0


def mean_reciprocal_rank(
    all_rankings: Sequence[Sequence[str]],
    all_relevant_ids: Sequence[Union[Set[str], Sequence[str]]],
) -> float:
    """Calculates Mean Reciprocal Rank (MRR) across a collection of evaluation queries.

    Formula:
        MRR = (1 / |Q|) * sum_{q in Q} RR(q)

    Args:
        all_rankings: Sequence of retrieved chunk ID sequences, one per query.
        all_relevant_ids: Sequence of ground-truth relevant chunk ID sets/sequences.

    Returns:
        float: Mean reciprocal rank across all queries. Returns 0.0 if all_rankings is empty.

    Raises:
        ValueError: If len(all_rankings) != len(all_relevant_ids).
        TypeError: If inputs are not sequences.
    """
    if not isinstance(all_rankings, (list, tuple, Sequence)):
        raise TypeError(f"all_rankings must be a sequence, got: {type(all_rankings).__name__}")
    if not isinstance(all_relevant_ids, (list, tuple, Sequence)):
        raise TypeError(f"all_relevant_ids must be a sequence, got: {type(all_relevant_ids).__name__}")

    if len(all_rankings) != len(all_relevant_ids):
        raise ValueError(
            f"Length mismatch: all_rankings has {len(all_rankings)} items, but "
            f"all_relevant_ids has {len(all_relevant_ids)} items."
        )

    if not all_rankings:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_rankings, all_relevant_ids)
    ]
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Evaluation result for a single query.

    Attributes:
        query_id: Identifier for the evaluation query.
        question: The query string evaluated.
        retrieved_chunk_ids: Ordered list of chunk IDs retrieved.
        relevant_chunk_ids: Set of ground-truth relevant chunk IDs.
        precision_at_k: Mapping of K -> Precision@K score.
        recall_at_k: Mapping of K -> Recall@K score.
        reciprocal_rank: Reciprocal rank score.
        document_id: Optional document identifier.
        notes: Optional rationale or notes.
    """

    query_id: str
    question: str
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: Set[str]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float
    document_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a dictionary."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "relevant_chunk_ids": sorted(list(self.relevant_chunk_ids)),
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "document_id": self.document_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate evaluation report across all queries.

    Attributes:
        results: Detailed per-query evaluation results.
        mean_reciprocal_rank: Aggregate MRR score.
        mean_precision_at_k: Aggregate mean Precision@K for each evaluated cutoff K.
        mean_recall_at_k: Aggregate mean Recall@K for each evaluated cutoff K.
        k_values: List of K values evaluated.
        total_queries: Total number of evaluated queries.
    """

    results: List[QueryEvaluationResult]
    mean_reciprocal_rank: float
    mean_precision_at_k: Dict[int, float]
    mean_recall_at_k: Dict[int, float]
    k_values: List[int]
    total_queries: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete evaluation report to a dictionary."""
        return {
            "summary": {
                "total_queries": self.total_queries,
                "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
                "mean_precision_at_k": {
                    str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()
                },
                "mean_recall_at_k": {
                    str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()
                },
                "k_values": list(self.k_values),
            },
            "queries": [r.to_dict() for r in self.results],
        }

    def format_text(self) -> str:
        """Generates a human-readable text report table."""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("DOCUMIND RETRIEVAL BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Total Queries Evaluated: {self.total_queries}")
        lines.append(f"Mean Reciprocal Rank (MRR): {self.mean_reciprocal_rank:.4f}")
        lines.append("-" * 80)
        lines.append("AGGREGATE METRICS:")
        for k in self.k_values:
            p = self.mean_precision_at_k.get(k, 0.0)
            r = self.mean_recall_at_k.get(k, 0.0)
            lines.append(f"  - K={k:2d}: Precision@{k} = {p:.4f} | Recall@{k} = {r:.4f}")
        lines.append("-" * 80)
        lines.append("PER-QUERY BREAKDOWN:")
        lines.append(
            f"{'ID':<6} {'RR':<6} "
            + " ".join([f"P@{k:<2}" for k in self.k_values])
            + " "
            + " ".join([f"R@{k:<2}" for k in self.k_values])
            + "  Question"
        )
        lines.append("-" * 80)
        for r in self.results:
            p_strs = " ".join([f"{r.precision_at_k.get(k, 0.0):.2f}" for k in self.k_values])
            r_strs = " ".join([f"{r.recall_at_k.get(k, 0.0):.2f}" for k in self.k_values])
            q_snippet = (r.question[:40] + "...") if len(r.question) > 40 else r.question
            lines.append(f"{r.query_id:<6} {r.reciprocal_rank:<6.2f} {p_strs} {r_strs}  {q_snippet}")
        lines.append("=" * 80)
        return "\n".join(lines)


def evaluate_retrieval(
    vector_store: VectorStore,
    dataset: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5),
    embedding_model: Optional[Any] = None,
) -> EvaluationReport:
    """Evaluates the vector store against a retrieval benchmark dataset.

    Args:
        vector_store: VectorStore containing the FAISS index and TextChunk records.
        dataset: Sequence of dictionaries, each containing:
            - 'question': str
            - 'relevant_chunk_ids': list of str
            - optional 'id' or 'query_id': str
            - optional 'document_id': str
            - optional 'notes': str
        k_values: Cutoff values for Precision@K and Recall@K. Default: (1, 3, 5).
        embedding_model: Optional SentenceTransformer or mock embedding model.

    Returns:
        EvaluationReport containing per-query metrics and aggregate benchmarks.

    Raises:
        TypeError: If dataset is not a sequence, or k_values contains invalid types.
        ValueError: If k_values is empty or contains non-positive numbers.
    """
    if not isinstance(dataset, (list, tuple, Sequence)):
        raise TypeError(f"dataset must be a sequence, got {type(dataset).__name__}")
    if not isinstance(k_values, (list, tuple, Sequence)):
        raise TypeError(f"k_values must be a sequence of integers, got {type(k_values).__name__}")

    clean_k_values: List[int] = []
    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k values must be positive integers, got: {k!r}")
        clean_k_values.append(k)

    clean_k_values = sorted(list(set(clean_k_values)))
    if not clean_k_values:
        raise ValueError("k_values cannot be empty.")

    max_k = max(clean_k_values)
    results: List[QueryEvaluationResult] = []

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise TypeError(f"Dataset items must be dictionaries, got {type(item).__name__} at index {idx}")

        question = item.get("question")
        if not question or not isinstance(question, str):
            raise ValueError(f"Dataset item at index {idx} is missing a valid 'question' string.")

        relevant_chunk_ids = item.get("relevant_chunk_ids")
        if relevant_chunk_ids is None or not isinstance(relevant_chunk_ids, (list, set, tuple)):
            raise ValueError(
                f"Dataset item at index {idx} ('{question}') is missing a 'relevant_chunk_ids' list."
            )

        query_id = str(item.get("id") or item.get("query_id") or f"q{idx + 1}")
        document_id = item.get("document_id")
        notes = item.get("notes")

        # Execute retrieval via FAISS
        search_results: List[SearchResult] = retrieve(
            query=question,
            vector_store=vector_store,
            top_k=max_k,
            model=embedding_model,
        )
        retrieved_ids = [res.chunk_id for res in search_results]

        # Calculate metrics
        p_at_k: Dict[int, float] = {}
        r_at_k: Dict[int, float] = {}
        for k in clean_k_values:
            p_at_k[k] = precision_at_k(retrieved_ids, relevant_chunk_ids, k=k)
            # Handle empty ground-truth sets gracefully for dataset queries
            if relevant_chunk_ids:
                r_at_k[k] = recall_at_k(retrieved_ids, relevant_chunk_ids, k=k)
            else:
                r_at_k[k] = 0.0

        rr = reciprocal_rank(retrieved_ids, relevant_chunk_ids)

        results.append(
            QueryEvaluationResult(
                query_id=query_id,
                question=question,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=set(relevant_chunk_ids),
                precision_at_k=p_at_k,
                recall_at_k=r_at_k,
                reciprocal_rank=rr,
                document_id=document_id,
                notes=notes,
            )
        )

    # Calculate aggregate metrics
    total_q = len(results)
    if total_q == 0:
        return EvaluationReport(
            results=[],
            mean_reciprocal_rank=0.0,
            mean_precision_at_k={k: 0.0 for k in clean_k_values},
            mean_recall_at_k={k: 0.0 for k in clean_k_values},
            k_values=clean_k_values,
            total_queries=0,
        )

    mrr = float(sum(r.reciprocal_rank for r in results) / total_q)
    mean_p: Dict[int, float] = {}
    mean_r: Dict[int, float] = {}
    for k in clean_k_values:
        mean_p[k] = float(sum(r.precision_at_k[k] for r in results) / total_q)
        mean_r[k] = float(sum(r.recall_at_k[k] for r in results) / total_q)

    return EvaluationReport(
        results=results,
        mean_reciprocal_rank=mrr,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        k_values=clean_k_values,
        total_queries=total_q,
    )
