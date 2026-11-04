"""Unit tests for DocuMind's text chunking module (Phase 2).

All test inputs are small, predictable, and deterministic. No network calls or
external data files are used.
"""

import pytest
from src.pdf_loader import PageRecord
from src.chunker import TextChunk, chunk_pages


def test_short_page_produces_single_chunk():
    """A page with text shorter than chunk_size should produce exactly one chunk."""
    page = PageRecord(page_number=1, text="Short introductory paragraph.")
    chunks = chunk_pages([page], chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, TextChunk)
    assert chunk.chunk_id == "p1_c0"
    assert chunk.page_number == 1
    assert chunk.text == "Short introductory paragraph."
    assert chunk.start_char == 0
    assert chunk.end_char == len("Short introductory paragraph.")
    assert chunk.char_count == len("Short introductory paragraph.")


def test_long_page_produces_multiple_chunks():
    """A page longer than chunk_size should be sliced into multiple chunks."""
    # 25 characters total
    text = "ABCDEFGHIJKLMNOPQRSTUVWXY"
    page = PageRecord(page_number=1, text=text)
    # chunk_size=10, overlap=3 -> step=7
    # c0: 0..10 ("ABCDEFGHIJ"), step to 7
    # c1: 7..17 ("HIJKLMNOPQ"), step to 14
    # c2: 14..24 ("OPQRSTUVWX"), step to 21
    # c3: 21..25 ("VWXY"), end >= 25 -> break
    chunks = chunk_pages([page], chunk_size=10, chunk_overlap=3)

    assert len(chunks) == 4
    assert [c.chunk_id for c in chunks] == ["p1_c0", "p1_c1", "p1_c2", "p1_c3"]
    assert chunks[0].text == "ABCDEFGHIJ"
    assert chunks[1].text == "HIJKLMNOPQ"
    assert chunks[2].text == "OPQRSTUVWX"
    assert chunks[3].text == "VWXY"


def test_correct_overlap_between_consecutive_chunks():
    """Verify that adjacent chunks share the exact specified character overlap."""
    text = "0123456789ABCDEFGHIJ"  # 20 chars
    page = PageRecord(page_number=1, text=text)
    # chunk_size=10, overlap=4 -> step=6
    # c0: 0..10  -> "0123456789"
    # c1: 6..16  -> "6789ABCDEF"
    # Overlap between c0 and c1: "6789" (length 4)
    chunks = chunk_pages([page], chunk_size=10, chunk_overlap=4)

    assert len(chunks) >= 2
    c0 = chunks[0]
    c1 = chunks[1]

    overlap_from_c0 = c0.text[-4:]
    overlap_from_c1 = c1.text[:4]
    assert overlap_from_c0 == overlap_from_c1
    assert overlap_from_c0 == "6789"
    assert len(overlap_from_c0) == 4


def test_preservation_of_page_numbers():
    """Each chunk must carry the original 1-based page number of its source page."""
    pages = [
        PageRecord(page_number=5, text="Page five content."),
        PageRecord(page_number=12, text="Page twelve content."),
    ]
    chunks = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].page_number == 5
    assert chunks[1].page_number == 12


def test_deterministic_chunk_ids_with_and_without_doc_id():
    """Verify chunk ID naming format is stable, deterministic, and supports document prefixes."""
    pages = [
        PageRecord(page_number=1, text="Page one long text that needs splitting into two parts."),
    ]
    # Default without doc_id
    chunks_no_doc = chunk_pages(pages, chunk_size=30, chunk_overlap=5)
    assert chunks_no_doc[0].chunk_id == "p1_c0"
    assert chunks_no_doc[1].chunk_id == "p1_c1"

    # With doc_id prefix
    chunks_with_doc = chunk_pages(pages, chunk_size=30, chunk_overlap=5, doc_id="paperA")
    assert chunks_with_doc[0].chunk_id == "paperA_p1_c0"
    assert chunks_with_doc[1].chunk_id == "paperA_p1_c1"


def test_empty_input_handling():
    """Empty list or None input should return an empty list without raising errors."""
    assert chunk_pages([]) == []
    assert chunk_pages(None) == []


def test_whitespace_only_pages_are_ignored():
    """Pages containing only spaces, tabs, or newlines should not produce any chunks."""
    pages = [
        PageRecord(page_number=1, text="   \n\t   "),
        PageRecord(page_number=2, text="Valid content on page two."),
        PageRecord(page_number=3, text=""),
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].text == "Valid content on page two."


def test_multiple_pages_with_different_page_numbers():
    """Multi-page document produces chunks correctly partitioned per page."""
    pages = [
        PageRecord(page_number=1, text="Alpha Bravo Charlie"),
        PageRecord(page_number=2, text="Delta Echo Foxtrot"),
        PageRecord(page_number=3, text="Golf Hotel India"),
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 3
    assert [c.page_number for c in chunks] == [1, 2, 3]
    assert [c.chunk_id for c in chunks] == ["p1_c0", "p2_c0", "p3_c0"]


def test_invalid_chunk_sizes():
    """Invalid chunk sizes (zero, negative, non-integer, bool) must raise ValueError."""
    pages = [PageRecord(page_number=1, text="Sample text.")]

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_pages(pages, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_pages(pages, chunk_size=-50)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_pages(pages, chunk_size="1000")  # string

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_pages(pages, chunk_size=True)  # boolean should not pass as int


def test_invalid_overlap_values():
    """Negative overlap or non-integer overlap must raise ValueError."""
    pages = [PageRecord(page_number=1, text="Sample text.")]

    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_pages(pages, chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_pages(pages, chunk_size=100, chunk_overlap="20")

    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_pages(pages, chunk_size=100, chunk_overlap=False)


def test_overlap_equal_or_greater_than_chunk_size():
    """Overlap equal to or greater than chunk_size must raise ValueError."""
    pages = [PageRecord(page_number=1, text="Sample text.")]

    # Overlap equal to chunk_size
    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)

    # Overlap greater than chunk_size
    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        chunk_pages(pages, chunk_size=100, chunk_overlap=150)


def test_no_empty_chunks_emitted():
    """Ensure no chunks with empty text are ever emitted."""
    pages = [
        PageRecord(page_number=1, text="Hello        world"),
    ]
    # Small chunk size that hits whitespace gaps
    chunks = chunk_pages(pages, chunk_size=4, chunk_overlap=1)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.char_count > 0
        assert chunk.text.strip() == chunk.text


def test_to_dict_serialization():
    """TextChunk.to_dict should include all key metadata."""
    chunk = TextChunk(
        chunk_id="doc1_p3_c2",
        page_number=3,
        text="Sample passage.",
        start_char=50,
        end_char=65,
    )
    d = chunk.to_dict()
    assert d["chunk_id"] == "doc1_p3_c2"
    assert d["page_number"] == 3
    assert d["text"] == "Sample passage."
    assert d["start_char"] == 50
    assert d["end_char"] == 65
    assert d["char_count"] == len("Sample passage.")
