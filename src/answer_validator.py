"""Answer grounding and citation validation module for DocuMind.

Phase 5.4: Evaluates whether LLM-generated answers adhere to citation rules,
verifies cited page numbers against retrieved source metadata, and assesses
sentence-level citation coverage without performing LLM-as-judge calls.
"""

from dataclasses import dataclass
import re
from typing import Any, List, Optional
from src.citation_validator import (
    extract_citations,
    get_available_pages,
    has_citations,
    validate_page_citations,
)
from src.rag_pipeline import RAGResponse


# Standard phrases indicating the model acknowledged insufficient document evidence
INSUFFICIENT_EVIDENCE_PHRASES = (
    "insufficient",
    "not enough information",
    "does not provide enough information",
    "does not contain enough information",
    "not mentioned in the provided",
    "not found in the provided",
    "not stated in the provided",
    "cannot be determined from the",
    "no information is provided",
    "no relevant document passages",
    "does not mention",
    "does not specify",
    "does not contain",
    "available document evidence is insufficient",
)


@dataclass(frozen=True)
class ValidationReport:
    """Immutable report containing the deterministic validation results of a RAG answer.

    Attributes:
        is_valid: True if answer has valid citations, no hallucinations of page numbers,
            and conforms to grounding rules. False if invalid citations exist, answer is empty,
            or citations are completely missing when sources were available.
        cited_pages: Deduplicated list of 1-based page numbers cited in the answer.
        available_pages: Sorted list of unique page numbers present in retrieved sources.
        invalid_citations: Cited page numbers absent from the retrieved sources.
        uncited_sentence_count: Number of sentences in the answer lacking page citations.
        warnings: List of human-readable warnings explaining potential issues.
    """

    is_valid: bool
    cited_pages: List[int]
    available_pages: List[int]
    invalid_citations: List[int]
    uncited_sentence_count: int
    warnings: List[str]

    @property
    def has_warnings(self) -> bool:
        """Returns True if the report contains one or more warnings."""
        return len(self.warnings) > 0

    @property
    def has_invalid_citations(self) -> bool:
        """Returns True if any cited page is missing from retrieved sources."""
        return len(self.invalid_citations) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serializes the validation report to a standard dictionary."""
        return {
            "is_valid": self.is_valid,
            "cited_pages": self.cited_pages,
            "available_pages": self.available_pages,
            "invalid_citations": self.invalid_citations,
            "uncited_sentence_count": self.uncited_sentence_count,
            "warnings": self.warnings,
        }


def split_sentences(text: str) -> List[str]:
    """Conservatively segments text into candidate sentences.

    Keeps trailing [Page N] citations bound to their preceding sentence. Avoids
    splitting on decimal numbers (e.g. 94.2) or empty whitespace lines.

    Limitations:
        Uses a heuristic regular expression rather than a full syntactic NLP parser.
        May not handle arbitrary nested parentheses, rare abbreviations, or code snippets.

    Args:
        text: Input string to segment.

    Returns:
        List[str]: List of segmented sentence strings.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: List[str] = []

    # Pattern matches text up to a sentence terminator (. ! ?) optionally followed by [Page N],
    # or ending in [Page N], followed by space and a capitalized word or end of line.
    pattern = re.compile(
        r"(.+?(?:[.!?](?:\s*\[[Pp]ages?\s+[^\]]+\])*|[.!?]?\s*\[[Pp]ages?\s+[^\]]+\]))(?=\s+[A-Z\d]|$)",
        re.DOTALL,
    )

    for line in lines:
        pos = 0
        line_sentences: List[str] = []
        for match in pattern.finditer(line):
            s = match.group(1).strip()
            if s:
                line_sentences.append(s)
                pos = match.end()

        # Capture any remainder on the line
        if pos < len(line):
            remainder = line[pos:].strip()
            if remainder:
                line_sentences.append(remainder)

        if not line_sentences:
            line_sentences = [line]

        sentences.extend(line_sentences)

    return sentences


def validate_answer(response: RAGResponse) -> ValidationReport:
    """Deterministically validates citations and grounding heuristics for a RAGResponse.

    Checks:
    1. Rejects empty or whitespace-only answers.
    2. Identifies all [Page N] citations and verifies they exist in retrieved sources.
    3. Flags invalid/hallucinated page citations.
    4. Evaluates whether answers without citations declare insufficient evidence.
    5. Calculates sentence-level citation coverage.

    Args:
        response: A RAGResponse object containing the answer and retrieved sources.

    Returns:
        ValidationReport: Immutable validation report.

    Raises:
        TypeError: If response is None or not a RAGResponse instance.
    """
    if response is None:
        raise TypeError("response cannot be None. Expected a RAGResponse instance.")

    if not isinstance(response, RAGResponse):
        raise TypeError(
            f"Expected 'response' to be a RAGResponse instance, got {type(response).__name__}."
        )

    answer = response.answer
    sources = response.sources

    cleaned_answer = answer.strip() if answer else ""
    available_pages = get_available_pages(sources)
    cited_pages = extract_citations(cleaned_answer)
    _, invalid_citations = validate_page_citations(cited_pages, available_pages)

    warnings: List[str] = []

    # 1. Check for empty answer
    if not cleaned_answer:
        return ValidationReport(
            is_valid=False,
            cited_pages=[],
            available_pages=available_pages,
            invalid_citations=[],
            uncited_sentence_count=0,
            warnings=["Answer is empty or contains only whitespace."],
        )

    # 2. Segment sentences and measure citation coverage
    sentences = split_sentences(cleaned_answer)
    uncited_sentences = [s for s in sentences if not has_citations(s)]
    uncited_count = len(uncited_sentences)

    lower_ans = cleaned_answer.lower()
    is_refusal = any(phrase in lower_ans for phrase in INSUFFICIENT_EVIDENCE_PHRASES)

    is_valid = True

    # 3. Case A: No evidence sources were retrieved
    if not available_pages:
        warnings.append("No document evidence was retrieved for this query.")
        if cited_pages:
            is_valid = False
            warnings.append(
                f"Answer cited page(s) {cited_pages} but no evidence sources were retrieved."
            )
        elif is_refusal:
            is_valid = True
            warnings.append("Answer declares available evidence is insufficient to answer the question.")
        else:
            is_valid = False
            warnings.append(
                "No evidence was retrieved, but the answer appears to make substantive claims instead of declaring insufficient evidence."
            )
        return ValidationReport(
            is_valid=is_valid,
            cited_pages=cited_pages,
            available_pages=available_pages,
            invalid_citations=invalid_citations,
            uncited_sentence_count=uncited_count,
            warnings=warnings,
        )

    # 4. Case B: Sources were retrieved
    if invalid_citations:
        is_valid = False
        warnings.append(
            f"Answer cites page(s) {invalid_citations} not present in retrieved sources (available: {available_pages})."
        )

    if not cited_pages:
        if is_refusal:
            is_valid = True
            warnings.append("Answer declares available evidence is insufficient to answer the question.")
        else:
            is_valid = False
            warnings.append("Answer contains no page citations despite available retrieved sources.")

    if uncited_count > 0 and not is_refusal:
        warnings.append(f"{uncited_count} of {len(sentences)} sentence(s) lack page citations.")

    return ValidationReport(
        is_valid=is_valid,
        cited_pages=cited_pages,
        available_pages=available_pages,
        invalid_citations=invalid_citations,
        uncited_sentence_count=uncited_count,
        warnings=warnings,
    )
