"""Unit tests for DocuMind's answer validator module (Phase 5.4).

Verifies deterministic citation verification, heuristic sentence coverage checking,
empty evidence handling, and edge cases. All tests run offline without network.
"""

import pytest

from src.chunker import TextChunk
from src.retriever import SearchResult
from src.rag_pipeline import RAGResponse
from src.answer_validator import ValidationReport, split_sentences, validate_answer


def _make_search_result(page_number: int, chunk_id: str = "p_c0", text: str = "Evidence text.") -> SearchResult:
    """Helper to build a SearchResult for answer validator tests."""
    chunk = TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return SearchResult(chunk=chunk, score=0.88, index=0)


def _make_rag_response(answer: str, pages: list[int]) -> RAGResponse:
    """Helper to build a RAGResponse with mock sources on specified pages."""
    sources = [_make_search_result(page_number=p, chunk_id=f"p{p}_c0") for p in pages]
    return RAGResponse(
        answer=answer,
        sources=sources,
        model="openai/gpt-oss-120b",
        question="Test query?",
    )


# ==============================================================================
# 1. Specification Examples Tests (A, B, C, D)
# ==============================================================================

def test_example_a_valid_citation():
    """Example A: Answer cites valid available page [Page 3] (available: [3, 7])."""
    response = _make_rag_response(
        answer="Exercise improves cardiovascular health. [Page 3]",
        pages=[3, 7],
    )
    report = validate_answer(response)

    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert report.cited_pages == [3]
    assert report.available_pages == [3, 7]
    assert report.invalid_citations == []
    assert report.uncited_sentence_count == 0
    assert report.has_invalid_citations is False


def test_example_b_invalid_citation():
    """Example B: Answer cites invalid page [Page 12] not present in sources [3, 7]."""
    response = _make_rag_response(
        answer="Exercise improves cardiovascular health. [Page 12]",
        pages=[3, 7],
    )
    report = validate_answer(response)

    assert report.is_valid is False
    assert report.cited_pages == [12]
    assert report.available_pages == [3, 7]
    assert report.invalid_citations == [12]
    assert report.has_invalid_citations is True
    assert any("not present in retrieved sources" in w for w in report.warnings)


def test_example_c_missing_citation_with_sources():
    """Example C: Answer provides no citations despite available sources [3, 7]."""
    response = _make_rag_response(
        answer="Exercise improves cardiovascular health.",
        pages=[3, 7],
    )
    report = validate_answer(response)

    assert report.is_valid is False
    assert report.cited_pages == []
    assert report.available_pages == [3, 7]
    assert report.invalid_citations == []
    assert report.uncited_sentence_count == 1
    assert any("contains no page citations" in w for w in report.warnings)
    assert any("lack page citations" in w for w in report.warnings)


def test_example_d_no_retrieved_evidence_proper_refusal():
    """Example D: No sources retrieved, answer correctly acknowledges insufficient evidence."""
    response = _make_rag_response(
        answer="The document does not provide enough information to answer this question.",
        pages=[],
    )
    report = validate_answer(response)

    assert report.is_valid is True
    assert report.cited_pages == []
    assert report.available_pages == []
    assert report.invalid_citations == []
    assert any("No document evidence was retrieved" in w for w in report.warnings)
    assert any("evidence is insufficient" in w for w in report.warnings)


# ==============================================================================
# 2. Empty Evidence Edge Cases
# ==============================================================================

def test_no_retrieved_evidence_but_makes_substantive_claim():
    """Verify warning and invalid status if no evidence retrieved but answer makes ungrounded claim."""
    response = _make_rag_response(
        answer="Quantum computing provides exponential speedups for factoring algorithms.",
        pages=[],
    )
    report = validate_answer(response)

    assert report.is_valid is False
    assert any("appears to make substantive claims" in w for w in report.warnings)


def test_no_retrieved_evidence_but_cites_fabricated_page():
    """Verify fabricated citations when 0 sources were retrieved are marked invalid."""
    response = _make_rag_response(
        answer="Quantum computing is fast. [Page 5]",
        pages=[],
    )
    report = validate_answer(response)

    assert report.is_valid is False
    assert report.invalid_citations == [5]
    assert any("no evidence sources were retrieved" in w for w in report.warnings)


# ==============================================================================
# 3. Sentence Coverage & Multi-Sentence Tests
# ==============================================================================

def test_mixed_sentence_citation_coverage():
    """Verify sentence coverage accurately counts uncited sentences in a multi-sentence answer."""
    answer = (
        "The model uses the AdamW optimizer with learning rate 0.001. [Page 2] "
        "Training was executed over 50 epochs without early stopping. "
        "Test accuracy reached 94.2%. [Page 3]"
    )
    response = _make_rag_response(answer=answer, pages=[2, 3])
    report = validate_answer(response)

    assert report.is_valid is True
    assert report.cited_pages == [2, 3]
    assert report.available_pages == [2, 3]
    assert report.invalid_citations == []
    # Sentence 2 lacks citation, so uncited count is 1
    assert report.uncited_sentence_count == 1
    assert any("1 of 3 sentence(s) lack page citations" in w for w in report.warnings)


def test_empty_and_whitespace_answers():
    """Verify empty or whitespace-only answers are flagged as invalid."""
    res_empty = _make_rag_response(answer="", pages=[1, 2])
    rep_empty = validate_answer(res_empty)
    assert rep_empty.is_valid is False
    assert any("empty" in w.lower() for w in rep_empty.warnings)

    res_ws = _make_rag_response(answer="   \n\t  ", pages=[1, 2])
    rep_ws = validate_answer(res_ws)
    assert rep_ws.is_valid is False
    assert any("empty" in w.lower() for w in rep_ws.warnings)


def test_duplicate_citations_and_sources():
    """Verify duplicate citations and multiple chunks on same page are handled cleanly."""
    answer = "Point A [Page 1]. Point B [Page 1]."
    # Duplicate sources from Page 1 and Page 2
    sources = [
        _make_search_result(page_number=1, chunk_id="p1_c0"),
        _make_search_result(page_number=1, chunk_id="p1_c1"),
        _make_search_result(page_number=2, chunk_id="p2_c0"),
    ]
    response = RAGResponse(
        answer=answer,
        sources=sources,
        model="openai/gpt-oss-120b",
        question="Query?",
    )
    report = validate_answer(response)

    assert report.is_valid is True
    assert report.cited_pages == [1]
    assert report.available_pages == [1, 2]
    assert report.invalid_citations == []


# ==============================================================================
# 4. Sentence Splitter Behavior & Limitations
# ==============================================================================

def test_split_sentences_preserves_decimals_and_citations():
    """Verify sentence splitter does not break on decimal floats and keeps citations bound."""
    text = (
        "Accuracy reached 94.2% on benchmark tests. [Page 3] "
        "Loss reduced to 0.0012 over 100 iterations. [Page 4]"
    )
    sentences = split_sentences(text)

    assert len(sentences) == 2
    assert "94.2%" in sentences[0]
    assert "[Page 3]" in sentences[0]
    assert "0.0012" in sentences[1]
    assert "[Page 4]" in sentences[1]


def test_split_sentences_empty_text():
    """Verify sentence splitter returns empty list for empty string."""
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


# ==============================================================================
# 5. Type Validation & Immutability
# ==============================================================================

def test_validate_answer_input_type_checking():
    """Verify validate_answer rejects None and non-RAGResponse types."""
    with pytest.raises(TypeError, match="cannot be None"):
        validate_answer(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected 'response' to be a RAGResponse"):
        validate_answer("not a RAGResponse")  # type: ignore


def test_validation_report_to_dict():
    """Verify ValidationReport serialization method produces all expected keys."""
    report = ValidationReport(
        is_valid=True,
        cited_pages=[1, 2],
        available_pages=[1, 2, 3],
        invalid_citations=[],
        uncited_sentence_count=0,
        warnings=[],
    )
    d = report.to_dict()

    assert d["is_valid"] is True
    assert d["cited_pages"] == [1, 2]
    assert d["available_pages"] == [1, 2, 3]
    assert d["invalid_citations"] == []
    assert d["uncited_sentence_count"] == 0
    assert d["warnings"] == []

