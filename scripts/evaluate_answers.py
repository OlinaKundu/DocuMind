#!/usr/bin/env python3
"""Answer Quality and Groundedness Evaluation Runner for DocuMind.

Phase 6.3: Standalone CLI to evaluate generated RAG answers using LLM-as-a-Judge
across Relevance, Faithfulness, and Correctness.

Usage:
    # Run against the sample/benchmark dataset using in-memory sample document:
    python scripts/evaluate_answers.py

    # Run against a specific PDF document and custom evaluation dataset:
    python scripts/evaluate_answers.py --pdf path/to/document.pdf --dataset evaluation/dataset.json --top-k 5

    # Export results to JSON:
    python scripts/evaluate_answers.py --output-json answer_eval_report.json
"""

import argparse
import json
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

from src.answer_evaluator import AnswerEvaluationReport, evaluate_dataset
from src.retriever import VectorStore
from scripts.evaluate_retrieval import (
    build_vector_store_from_pdf,
    create_sample_vector_store,
    load_dataset,
)


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running answer quality evaluation."""
    parser = argparse.ArgumentParser(
        description="DocuMind Phase 6.3: Answer Quality & Groundedness Evaluation (LLM-as-a-Judge)"
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
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question (default: 5)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional Groq model name for the LLM judge (default: configured Groq model)",
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default=None,
        help="Optional Groq model name for RAG generation (default: configured Groq model)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write evaluation results as JSON",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.top_k <= 0:
        print(f"Error: --top-k must be a positive integer, got {parsed_args.top_k}", file=sys.stderr)
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
    print("DOCUMIND ANSWER QUALITY & GROUNDEDNESS EVALUATION")
    print("=" * 80)
    print(f"Dataset:            {parsed_args.dataset}")
    print(f"Queries count:      {len(queries)}")
    print(f"Top-K Retrieval:    {parsed_args.top_k}")
    if parsed_args.judge_model:
        print(f"Judge Model:        {parsed_args.judge_model}")
    if parsed_args.gen_model:
        print(f"Generator Model:    {parsed_args.gen_model}")

    # Build or load VectorStore
    if parsed_args.pdf:
        print(f"\nIndexing PDF document: {parsed_args.pdf}")
        try:
            vector_store = build_vector_store_from_pdf(parsed_args.pdf)
            print(f"Indexed {vector_store.count} chunks in FAISS IndexFlatIP.")
        except Exception as err:
            print(f"Error building vector store from PDF: {err}", file=sys.stderr)
            return 1
    else:
        print("\n" + "=" * 80)
        print("[!] NOTICE: DEMO / PLACEHOLDER ANSWER EVALUATION RUN")
        print("No --pdf file was provided. Constructing an in-memory sample document VectorStore.")
        if is_placeholder:
            print("The evaluation dataset contains placeholder/sample reference answers.")
        print("To run an authoritative evaluation on a real PDF document:")
        print("    python scripts/evaluate_answers.py --pdf /path/to/paper.pdf --dataset evaluation/dataset.json")
        print("=" * 80 + "\n")
        vector_store = create_sample_vector_store()
        print(f"Constructed sample vector store with {vector_store.count} chunks.")

    # Run evaluation
    print("\nRunning RAG answer generation and LLM-as-a-Judge evaluation...")
    report: AnswerEvaluationReport = evaluate_dataset(
        vector_store=vector_store,
        dataset=queries,
        top_k=parsed_args.top_k,
        judge_model=parsed_args.judge_model,
        gen_model=parsed_args.gen_model,
    )

    # Print formatted report
    print("\n" + report.format_text())

    # Optional JSON export
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

