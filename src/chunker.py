"""Text chunking module for DocuMind.

Responsible for splitting extracted document page records into smaller,
manageable text segments (chunks) using a configurable character-based sliding
window, while strictly preserving 1-based source page metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from src.pdf_loader import PageRecord


@dataclass(frozen=True)
class TextChunk:
    """Structured record representing a text chunk derived from a document page.

    Attributes:
        chunk_id: Unique, deterministic identifier (e.g., 'p1_c0' or 'doc1_p1_c0').
        page_number: Human-readable 1-based source page number where the text originates.
        text: Extracted chunk text content (with outer whitespace trimmed).
        start_char: 0-based starting character offset within the source page text.
        end_char: 0-based ending character offset within the source page text.
    """

    chunk_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        """Returns the number of characters in the chunk text."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the chunk record to a standard dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "char_count": self.char_count,
        }


def chunk_pages(
    pages: Optional[Sequence[PageRecord]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    doc_id: Optional[str] = None,
) -> List[TextChunk]:
    """Splits a sequence of PageRecord objects into overlapping character-based chunks.

    Each page is chunked independently to guarantee unambiguous page-level citations.
    Pages containing only whitespace are skipped without generating empty chunks.

    Args:
        pages: A sequence of PageRecord instances (from `pdf_loader.py`).
        chunk_size: Maximum character count per chunk. Must be a positive integer.
            Default is 1000 characters.
        chunk_overlap: Number of characters to overlap between adjacent chunks.
            Must be a non-negative integer strictly less than chunk_size.
            Default is 150 characters.
        doc_id: Optional document identifier prefix for chunk IDs
            (e.g., 'doc1' produces 'doc1_p1_c0'). If None, chunk ID is 'p1_c0'.

    Returns:
        List[TextChunk]: An ordered list of structured TextChunk records.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    # 1. Parameter Validation
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got: {chunk_size!r}"
        )

    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap must be a non-negative integer, got: {chunk_overlap!r}"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than "
            f"chunk_size ({chunk_size}). Overlap cannot equal or exceed chunk size."
        )

    # 2. Handle empty input safely
    if not pages:
        return []

    chunks: List[TextChunk] = []
    step = chunk_size - chunk_overlap

    # 3. Process each page independently to preserve page boundaries
    for page in pages:
        raw_text = page.text
        # Skip pages with no text or only whitespace
        if not raw_text or not raw_text.strip():
            continue

        text_len = len(raw_text)
        chunk_index = 0
        start = 0

        # 4. Sliding window loop across the page text
        while start < text_len:
            end = min(start + chunk_size, text_len)
            slice_text = raw_text[start:end]
            cleaned_text = slice_text.strip()

            # Ensure we never emit empty or whitespace-only chunks
            if cleaned_text:
                if doc_id:
                    c_id = f"{doc_id}_p{page.page_number}_c{chunk_index}"
                else:
                    c_id = f"p{page.page_number}_c{chunk_index}"

                chunks.append(
                    TextChunk(
                        chunk_id=c_id,
                        page_number=page.page_number,
                        text=cleaned_text,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            # If the current window reached the end of the text, stop
            if end >= text_len:
                break

            start += step

    return chunks
