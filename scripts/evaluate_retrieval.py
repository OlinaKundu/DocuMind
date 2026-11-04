#!/usr/bin/env python3
"""Retrieval evaluation runner for DocuMind.

Phase 6.2: Measures vector retrieval quality using Precision@K, Recall@K,
and Mean Reciprocal Rank (MRR) independently of LLM answer generation.

Usage:
    # Run evaluation against the default sample dataset:
    python scripts/evaluate_retrieval.py

    # Run evaluation against a real PDF document and custom dataset:
    python scripts/evaluate_retrieval.py --pdf path/to/document.pdf --dataset path/to/dataset.json --k 1,3,5

    # Export results to JSON:
    python scripts/evaluate_retrieval.py --output-json report.json
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pdf_loader import PageRecord, extract_text_from_pdf
from src.chunker import TextChunk, chunk_pages
from src.embeddings import embed_chunks
from src.retriever import VectorStore, build_index
from src.evaluation import EvaluationReport, evaluate_retrieval


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Loads and validates an evaluation dataset JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(
            f"Invalid dataset format in {dataset_path}. Expected a JSON object with a 'queries' array."
        )

    return data


def create_sample_vector_store() -> VectorStore:
    """Constructs a deterministic sample VectorStore for demonstration and placeholder evaluation.

    Creates sample document pages with realistic scientific content whose chunk IDs
    match the placeholder queries in evaluation/dataset.json (e.g. doc1_p2_c0, doc1_p4_c0).
    """
    sample_pages = [
        PageRecord(
            page_number=1,
            text=(
                "DocuMind: Efficient Vector Retrieval and Grounded RAG Architecture. "
                "Abstract: We introduce DocuMind, an offline-first RAG pipeline for research documents."
            ),
        ),
        PageRecord(
            page_number=2,
            text=(
                "Model Architecture Comparison: Model A utilizes standard dense self-attention, "
                "whereas Model B adopts linear causal attention with flash-attention kernel dispatch. "
                "The primary architectural difference is Model B's sub-quadratic memory scaling."
            ),
        ),
        PageRecord(
            page_number=3,
            text=(
                "Data Preprocessing: Text streams were extracted directly from PDF records and normalized. "
                "Stopwords and whitespace were normalized using deterministic rules."
            ),
        ),
        PageRecord(
            page_number=4,
            text=(
                "Training Details and Hyperparameters: Model training was conducted using the SGD optimizer "
                "(Stochastic Gradient Descent) with an initial learning rate of 0.01 and momentum 0.9. "
                "A cosine annealing learning rate schedule was applied across 100 epochs."
            ),
        ),
        PageRecord(
            page_number=5,
            text=(
                "Hardware Specifications: All experimental evaluations were executed on a distributed cluster "
                "of 8 NVIDIA A100 Tensor Core GPUs (80GB VRAM each), interconnected via 400 Gbps InfiniBand."
            ),
        ),
        PageRecord(
            page_number=6,
            text=(
                "Evaluation Benchmark Datasets: To evaluate benchmark accuracy, we tested on GLUE, SuperGLUE, "
                "and SQuAD 2.0 question answering benchmarks, reporting exact match and F1 metrics."
            ),
        ),
        PageRecord(
            page_number=7,
            text=(
                "Experimental Results: Table 1 demonstrates that our retrieval architecture achieved a 12% "
                "relative latency reduction compared to baseline dense vector index configurations."
            ),
        ),
        PageRecord(
            page_number=8,
            text=(
                "Limitations and Future Directions: Key limitations include reliance on digital text extraction "
                "without OCR for scanned pages. Future directions include exploring multi-vector ColBERT indexing."
            ),
        ),
    ]

    # Chunk the sample pages using standard chunk_pages with doc_id='doc1'
    chunks = chunk_pages(sample_pages, chunk_size=500, chunk_overlap=50, doc_id="doc1")
    chunk_embeddings = embed_chunks(chunks)
    return build_index(chunk_embeddings)


def build_vector_store_from_pdf(pdf_path: Path, doc_id: str = "doc1") -> VectorStore:
    """Ingests a real PDF document and builds a FAISS VectorStore."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150, doc_id=doc_id)
    if not chunks:
        raise ValueError(f"No extractable text chunks found in PDF: {pdf_path}")

    chunk_embeddings = embed_chunks(chunks)
    return build_index(chunk_embeddings)


def parse_k_values(k_str: str) -> List[int]:
    """Parses a comma-separated string of K values (e.g., '1,3,5')."""
    try:
        k_values = [int(item.strip()) for item in k_str.split(",") if item.strip()]
        if not k_values:
            raise ValueError()
        for k in k_values:
            if k <= 0:
                raise ValueError()
        return sorted(list(set(k_values)))
    except Exception as exc:
        raise ValueError(
            f"Invalid --k parameter: {k_str!r}. Expected comma-separated positive integers (e.g. '1,3,5')."
        ) from exc


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="DocuMind Phase 6.2: Retrieval Quality Evaluation & Benchmarking"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "evaluation" / "dataset.json",
        help="Path to evaluation dataset JSON file (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional path to real PDF document to index for evaluation",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,3,5",
        help="Comma-separated K cutoff ranks (default: '1,3,5')",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write evaluation results as JSON",
    )

    parsed_args = parser.parse_args(args)

    try:
        k_values = parse_k_values(parsed_args.k)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    try:
        dataset_data = load_dataset(parsed_args.dataset)
    except Exception as err:
        print(f"Error loading dataset: {err}", file=sys.stderr)
        return 1

    queries = dataset_data.get("queries", [])
    metadata = dataset_data.get("metadata", {})
    is_placeholder = metadata.get("is_placeholder_dataset", False)

    print("=" * 80)
    print("DOCUMIND RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Dataset: {parsed_args.dataset}")
    print(f"Queries count: {len(queries)}")
    print(f"Evaluation cutoffs (K): {k_values}")

    # Build or load VectorStore
    if parsed_args.pdf:
        print(f"Indexing PDF document: {parsed_args.pdf}")
        try:
            vector_store = build_vector_store_from_pdf(parsed_args.pdf)
            print(f"Indexed {vector_store.count} chunks in FAISS IndexFlatIP.")
        except Exception as err:
            print(f"Error building vector store from PDF: {err}", file=sys.stderr)
            return 1
    else:
        print("\n" + "=" * 80)
        print("[!] NOTICE: DEMO / PLACEHOLDER EVALUATION RUN")
        print("No --pdf file was provided. Constructing an in-memory sample document VectorStore.")
        if is_placeholder:
            print("The evaluation dataset contains placeholder ground-truth labels.")
        print("To run an authoritative benchmark on a real PDF document:")
        print("    python scripts/evaluate_retrieval.py --pdf /path/to/paper.pdf --dataset evaluation/dataset.json")
        print("=" * 80 + "\n")
        vector_store = create_sample_vector_store()
        print(f"Constructed sample vector store with {vector_store.count} chunks.")

    # Run evaluation
    print("\nRunning retrieval evaluation...")
    report: EvaluationReport = evaluate_retrieval(
        vector_store=vector_store,
        dataset=queries,
        k_values=k_values,
    )

    # Print report
    print("\n" + report.format_text())

    # Optional JSON output
    if parsed_args.output_json:
        try:
            with open(parsed_args.output_json, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nReport written to: {parsed_args.output_json}")
        except Exception as err:
            print(f"Error saving JSON report: {err}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
