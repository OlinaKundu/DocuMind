"""Unit tests for DocuMind's citation extraction and validation module (Phase 5.4).

Verifies regex citation extraction, multi-page parsing, deduplication,
source page harvesting, and page validity checking. All tests run offline.
"""

from typing import List
import pytest

from src.chunker import TextChunk
from src.retriever import SearchResult
from src.citation_validator import (
    extract_citations,
    get_available_pages,
    has_citations,
    validate_page_citations,
)


def _make_search_result(page_number: int, chunk_id: str = "c0") -> SearchResult:
    """Helper to build a SearchResult for citation validator tests."""
    chunk = TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text="Sample passage text.",
        start_char=0,
        end_char=20,
    )
    return SearchResult(chunk=chunk, score=0.9, index=0)


# ==============================================================================
# 1. Extraction Tests
# ==============================================================================

def test_extract_single_citation():
    """Verify extracting a single standard [Page N] citation."""
    text = "The CIFAR-10 dataset contains 60,000 images. [Page 3]"
    assert extract_citations(text) == [3]


def test_extract_multiple_citations_separate_brackets():
    """Verify extracting multiple distinct citations across sentences."""
    text = (
        "Neural networks require regularisation [Page 2]. "
        "Weight decay prevents overfitting effectively [Page 7]."
    )
    assert extract_citations(text) == [2, 7]


def test_extract_multiple_citations_single_bracket():
    """Verify comma-separated citations inside a single bracket are parsed."""
    text1 = "Both studies agree on the baseline metrics [Page 4, 8]."
    assert extract_citations(text1) == [4, 8]

    text2 = "Cross-referenced in literature [Page 1, Page 3, Page 9]."
    assert extract_citations(text2) == [1, 3, 9]

    text3 = "Combined findings [Pages 12 and 15]."
    assert extract_citations(text3) == [12, 15]


def test_extract_case_insensitivity_and_spacing():
    """Verify extraction handles varying cases and whitespace."""
    assert extract_citations("Refer to [page 5] for details.") == [5]
    assert extract_citations("Refer to [PAGE 10] for details.") == [10]
    assert extract_citations("Refer to [Page    42] for details.") == [42]


def test_extract_deduplicates_duplicate_citations():
    """Verify duplicate citations within an answer are deduplicated while preserving order."""
    text = "Initial discovery [Page 3]. Further confirmation [Page 7]. Concluding remarks [Page 3]."
    assert extract_citations(text) == [3, 7]


def test_extract_empty_or_non_string_inputs():
    """Verify empty text, whitespace, or invalid types return an empty list without crashing."""
    assert extract_citations("") == []
    assert extract_citations("   \n\t  ") == []
    assert extract_citations(None) == []  # type: ignore
    assert extract_citations(12345) == []  # type: ignore


def test_has_citations():
    """Verify has_citations returns True only when valid page citations exist."""
    assert has_citations("Contains citation [Page 4].") is True
    assert has_citations("No citations in this sentence.") is False
    assert has_citations("") is False


# ==============================================================================
# 2. Available Page Harvesting Tests
# ==============================================================================

def test_get_available_pages_normal():
    """Verify extracting unique sorted page numbers from search results."""
    sources = [
        _make_search_result(page_number=3),
        _make_search_result(page_number=1),
        _make_search_result(page_number=7),
    ]
    assert get_available_pages(sources) == [1, 3, 7]


def test_get_available_pages_handles_duplicates():
    """Verify multiple chunks originating from the same page are deduplicated."""
    sources = [
        _make_search_result(page_number=2, chunk_id="p2_c0"),
        _make_search_result(page_number=2, chunk_id="p2_c1"),
        _make_search_result(page_number=5, chunk_id="p5_c0"),
        _make_search_result(page_number=2, chunk_id="p2_c2"),
    ]
    assert get_available_pages(sources) == [2, 5]


def test_get_available_pages_empty_sources():
    """Verify empty source collection returns empty list."""
    assert get_available_pages([]) == []
    assert get_available_pages(None) == []  # type: ignore


# ==============================================================================
# 3. Citation Validity Tests
# ==============================================================================

def test_validate_page_citations_all_valid():
    """Verify when all cited pages exist in available sources."""
    cited = [2, 4]
    available = [1, 2, 3, 4, 5]
    valid, invalid = validate_page_citations(cited, available)

    assert valid == [2, 4]
    assert invalid == []


def test_validate_page_citations_some_invalid():
    """Verify when cited pages contain pages missing from available sources."""
    cited = [2, 99]
    available = [1, 2, 3]
    valid, invalid = validate_page_citations(cited, available)

    assert valid == [2]
    assert invalid == [99]


def test_validate_page_citations_deduplicates_invalid():
    """Verify invalid citations are reported without duplicate entries."""
    cited = [88, 88, 99]
    available = [1, 2]
    valid, invalid = validate_page_citations(cited, available)

    assert valid == []
    assert invalid == [88, 99]


def test_validate_page_citations_empty_sources():
    """Verify citations when no sources exist are all classified as invalid."""
    cited = [3, 5]
    available: List[int] = []
    valid, invalid = validate_page_citations(cited, available)

    assert valid == []
    assert invalid == [3, 5]


def test_validate_page_citations_empty_citations():
    """Verify validating empty citations list returns empty tuples."""
    valid, invalid = validate_page_citations([], [1, 2, 3])
    assert valid == []
    assert invalid == []

