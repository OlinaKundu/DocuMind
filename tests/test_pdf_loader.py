"""Unit tests for DocuMind's PDF loader module.

All test PDFs are generated dynamically in memory using PyMuPDF to ensure tests
are fast, deterministic, self-contained, and require no external file downloads.
"""

import io
import pytest
import pymupdf

from src.pdf_loader import (
    PageRecord,
    PDFLoadingError,
    EmptyPDFError,
    InvalidPDFError,
    extract_text_from_pdf,
)


def _generate_test_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper utility to create an in-memory PDF with specified text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            # Insert text at standard margin position (x=50, y=72)
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_single_page():
    """Verify successful extraction from a single-page PDF."""
    sample_text = "DocuMind is an intelligent RAG system."
    pdf_bytes = _generate_test_pdf_bytes([sample_text])

    records = extract_text_from_pdf(pdf_bytes)

    assert len(records) == 1
    assert isinstance(records[0], PageRecord)
    assert records[0].page_number == 1
    assert records[0].text == sample_text
    assert records[0].char_count == len(sample_text)
    assert not records[0].is_empty


def test_one_based_page_numbering():
    """Verify that multi-page documents maintain sequential 1-based page numbers."""
    pages = [
        "First page content: Abstract and Introduction.",
        "Second page content: Methodology and System Architecture.",
        "Third page content: Experimental Evaluation and Results.",
    ]
    pdf_bytes = _generate_test_pdf_bytes(pages)

    records = extract_text_from_pdf(pdf_bytes)

    assert len(records) == 3
    for idx, record in enumerate(records, start=1):
        assert record.page_number == idx
        assert record.text == pages[idx - 1]


def test_empty_page_handling():
    """Verify that pages with no text (e.g. blank or scanned images) are preserved as empty strings."""
    pages = [
        "Page 1 has text content.",
        "",  # Blank page
        "Page 3 also has text content.",
    ]
    pdf_bytes = _generate_test_pdf_bytes(pages)

    records = extract_text_from_pdf(pdf_bytes)

    assert len(records) == 3
    assert records[0].page_number == 1
    assert records[0].text == "Page 1 has text content."
    assert not records[0].is_empty

    # Page 2 should have empty text and is_empty flag set to True
    assert records[1].page_number == 2
    assert records[1].text == ""
    assert records[1].char_count == 0
    assert records[1].is_empty

    assert records[2].page_number == 3
    assert records[2].text == "Page 3 also has text content."


def test_file_like_bytesio_input():
    """Verify that extract_text_from_pdf accepts an io.BytesIO stream (like Streamlit UploadedFile)."""
    sample_text = "Testing Streamlit-compatible file-like stream."
    pdf_bytes = _generate_test_pdf_bytes([sample_text])
    stream = io.BytesIO(pdf_bytes)

    records = extract_text_from_pdf(stream)

    assert len(records) == 1
    assert records[0].page_number == 1
    assert records[0].text == sample_text


def test_empty_bytes_raises_empty_pdf_error():
    """Verify that providing 0 bytes raises an EmptyPDFError."""
    with pytest.raises(EmptyPDFError, match="empty"):
        extract_text_from_pdf(b"")

    with pytest.raises(EmptyPDFError, match="empty"):
        extract_text_from_pdf(b"   \n\t  ")


def test_invalid_pdf_data_raises_invalid_pdf_error():
    """Verify that corrupt or non-PDF bytes raise an InvalidPDFError."""
    malformed_data = b"This is a random text file, definitely not a valid PDF!"

    with pytest.raises(InvalidPDFError, match="Failed to parse PDF document"):
        extract_text_from_pdf(malformed_data)


def test_none_input_raises_invalid_pdf_error():
    """Verify that None input is caught and raises an InvalidPDFError."""
    with pytest.raises(InvalidPDFError, match="No PDF input provided"):
        extract_text_from_pdf(None)


def test_unsupported_input_type_raises_invalid_pdf_error():
    """Verify that unexpected object types raise an InvalidPDFError."""
    with pytest.raises(InvalidPDFError, match="Unsupported input type"):
        extract_text_from_pdf(12345)
