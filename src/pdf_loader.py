"""PDF Loader Module for DocuMind.

Responsible for safely ingesting PDF documents, extracting textual content
page-by-page using PyMuPDF, preserving 1-based page metadata, and handling edge
cases such as corrupt, empty, or image-only scanned files.
"""

from dataclasses import dataclass
import io
from typing import Any, BinaryIO, List, Union
import pymupdf


class PDFLoadingError(Exception):
    """Base exception raised for general PDF loading or extraction errors."""
    pass


class EmptyPDFError(PDFLoadingError):
    """Exception raised when the input PDF file is empty or has zero pages."""
    pass


class InvalidPDFError(PDFLoadingError):
    """Exception raised when the input file is not a valid PDF or is corrupted."""
    pass


@dataclass(frozen=True)
class PageRecord:
    """Structured record representing extracted content from a single PDF page.

    Attributes:
        page_number: Human-readable 1-based page index (Page 1 = first page).
        text: The textual content extracted from the page (trimmed of outer whitespace).
    """

    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        """Returns the number of characters in the extracted text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Returns True if the page contains no extractable text (e.g., blank or scanned image)."""
        return len(self.text) == 0


def extract_text_from_pdf(
    file_input: Union[bytes, BinaryIO, io.BytesIO, Any]
) -> List[PageRecord]:
    """Extracts text page-by-page from a PDF byte stream or file-like object.

    Args:
        file_input: Raw bytes, an io.BytesIO stream, or a file-like object
            (such as Streamlit's UploadedFile).

    Returns:
        List[PageRecord]: A list of structured page records containing 1-based page
        numbers and extracted text.

    Raises:
        EmptyPDFError: If the input is empty (0 bytes) or has zero pages.
        InvalidPDFError: If the file is corrupt or cannot be parsed as a PDF.
        PDFLoadingError: If an unexpected error occurs during page extraction.

    Note on Scanned Documents (OCR):
        PyMuPDF extracts embedded digital text streams. If a PDF contains only
        scanned images (such as a photograph or scanned paper), `page.get_text()`
        will return an empty string for that page. Optical Character Recognition
        (OCR) is intentionally omitted in Phase 1 to keep dependencies lightweight
        and local.
    """
    if file_input is None:
        raise InvalidPDFError("No PDF input provided. Expected bytes or a file-like object.")

    # Extract raw bytes from various input formats without unnecessary file duplication
    if isinstance(file_input, bytes):
        pdf_bytes = file_input
    elif hasattr(file_input, "getvalue"):
        # Handles Streamlit UploadedFile and io.BytesIO
        pdf_bytes = file_input.getvalue()
    elif hasattr(file_input, "read"):
        # Handles standard file-like streams
        pdf_bytes = file_input.read()
    else:
        raise InvalidPDFError(
            f"Unsupported input type '{type(file_input).__name__}'. "
            "Expected bytes or a file-like object."
        )

    # Check for empty byte sequence or pure whitespace
    if not pdf_bytes or len(pdf_bytes.strip()) == 0:
        raise EmptyPDFError(
            "The provided PDF input is empty (0 bytes received). "
            "Please provide a valid, non-empty PDF file."
        )

    # Open PDF stream using PyMuPDF
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.EmptyFileError as exc:
        raise EmptyPDFError(f"Cannot open empty PDF stream: {exc}") from exc
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise InvalidPDFError(
            f"Failed to parse PDF document. The file may be corrupt or not a valid PDF: {exc}"
        ) from exc
    except Exception as exc:
        raise PDFLoadingError(f"Unexpected error opening PDF: {exc}") from exc

    # Context manager guarantees prompt closure of PyMuPDF C++ memory resources
    with doc:
        total_pages = len(doc)
        if total_pages == 0:
            raise EmptyPDFError("The PDF document contains 0 pages.")

        records: List[PageRecord] = []

        # Iterate through pages using human-readable 1-based indexing
        for page_index, page in enumerate(doc, start=1):
            try:
                # Extract text using PyMuPDF's layout-preserving plaintext extractor
                raw_text = page.get_text()
                cleaned_text = raw_text.strip()

                records.append(
                    PageRecord(
                        page_number=page_index,
                        text=cleaned_text,
                    )
                )
            except Exception as exc:
                raise PDFLoadingError(
                    f"Failed to extract text from page {page_index}: {exc}"
                ) from exc

        return records
