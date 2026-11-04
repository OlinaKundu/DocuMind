"""Unit tests for DocuMind's context construction module (Phase 5.2).

Verifies correct passage attribution, ranking preservation, metadata formatting,
type validation, and immutability. All tests run offline.
"""

import pytest

from src.chunker import TextChunk
from src.retriever import SearchResult
from src.context_builder import PASSAGE_SEPARATOR, build_context


def _make_search_result(chunk_id: str, page_number: int, text: str, score: float = 0.85, index: int = 0) -> SearchResult:
    """Helper to create a deterministic SearchResult instance."""
    chunk = TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return SearchResult(chunk=chunk, score=score, index=index)


def test_build_context_single_chunk():
    """Verify build_context formats a single SearchResult correctly."""
    res = _make_search_result("p1_c0", 1, "The CIFAR-10 dataset contains 60,000 colour images.")
    context = build_context([res])

    expected = (
        "[Source 1 | Page 1 | Chunk p1_c0]\n"
        "The CIFAR-10 dataset contains 60,000 colour images."
    )
    assert context == expected


def test_build_context_multiple_chunks():
    """Verify multiple chunks are formatted with correct headers and standard separators."""
    r1 = _make_search_result("p3_c0", 3, "First passage about neural network loss landscapes.")
    r2 = _make_search_result("p4_c1", 4, "Second passage detailing learning rate schedules.")
    r3 = _make_search_result("p7_c2", 7, "Third passage reporting benchmark evaluation metrics.")

    context = build_context([r1, r2, r3])

    expected = (
        f"[Source 1 | Page 3 | Chunk p3_c0]\n{r1.text}"
        f"{PASSAGE_SEPARATOR}"
        f"[Source 2 | Page 4 | Chunk p4_c1]\n{r2.text}"
        f"{PASSAGE_SEPARATOR}"
        f"[Source 3 | Page 7 | Chunk p7_c2]\n{r3.text}"
    )
    assert context == expected


def test_build_context_preserves_retrieval_ranking_order():
    """Verify that retrieval rank order (Source 1, Source 2) is strictly maintained.

    Even when page numbers or chunk IDs are non-sequential, Source numbering must
    reflect the list ordering determined by the retriever.
    """
    # Result on page 9 is ranked #1 by retriever, result on page 2 is ranked #2
    r_top = _make_search_result("p9_c3", 9, "Top ranked finding.", score=0.95, index=5)
    r_second = _make_search_result("p2_c0", 2, "Second ranked background.", score=0.88, index=1)

    context = build_context([r_top, r_second])

    assert context.startswith("[Source 1 | Page 9 | Chunk p9_c3]")
    assert "[Source 2 | Page 2 | Chunk p2_c0]" in context
    # Source 1 must appear before Source 2 in the string
    assert context.index("[Source 1") < context.index("[Source 2")


def test_build_context_page_number_and_chunk_id_preservation():
    """Verify source page numbers and chunk IDs are strictly preserved without mutation."""
    res = _make_search_result("docA_p42_c5", 42, "Deep theoretical convergence proof.")
    context = build_context([res])

    assert "Page 42" in context
    assert "Chunk docA_p42_c5" in context
    assert "[Source 1 | Page 42 | Chunk docA_p42_c5]" in context


def test_build_context_empty_results():
    """Verify build_context returns an empty string when given empty collections."""
    assert build_context([]) == ""
    assert build_context(()) == ""


def test_build_context_invalid_inputs():
    """Verify invalid inputs raise clear TypeErrors."""
    # None input
    with pytest.raises(TypeError, match="cannot be None"):
        build_context(None)

    # String input
    with pytest.raises(TypeError, match="Expected a sequence of SearchResult objects"):
        build_context("invalid string")

    # Bytes input
    with pytest.raises(TypeError, match="Expected a sequence of SearchResult objects"):
        build_context(b"invalid bytes")

    # Non-iterable integer
    with pytest.raises(TypeError):
        build_context(12345)

    # List containing non-SearchResult objects
    with pytest.raises(TypeError, match="must be SearchResult instances"):
        build_context(["not a search result"])

    # List containing raw TextChunk instead of SearchResult
    raw_chunk = TextChunk("p1_c0", 1, "text", 0, 4)
    with pytest.raises(TypeError, match="must be SearchResult instances"):
        build_context([raw_chunk])


def test_build_context_original_objects_remain_unchanged():
    """Verify build_context does not modify or mutate SearchResult or TextChunk objects."""
    chunk = TextChunk("p2_c1", 2, "Immutable passage text.", 10, 33)
    res = SearchResult(chunk=chunk, score=0.91, index=3)

    # Capture state before
    chunk_id_before = res.chunk_id
    page_before = res.page_number
    text_before = res.text
    score_before = res.score
    index_before = res.index

    _ = build_context([res])

    # Assert state identical after
    assert res.chunk_id == chunk_id_before
    assert res.page_number == page_before
    assert res.text == text_before
    assert res.score == score_before
    assert res.index == index_before


def test_build_context_preserves_passage_text_verbatim():
    """Verify passage text is preserved exactly without alteration or stripping of internal formatting."""
    multiline_text = "Line 1: Special symbols: !@#$%^&*()\nLine 2: Tabbed\tcontent\nLine 3: End."
    res = _make_search_result("p1_c0", 1, multiline_text)

    context = build_context([res])
    assert multiline_text in context

