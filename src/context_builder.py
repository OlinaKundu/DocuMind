"""Context construction module for DocuMind.

Phase 5.2: Formats retrieved SearchResult passages into structured, page-attributed
evidence blocks for grounded LLM consumption while strictly preserving retrieval
ranking and source metadata.
"""

from typing import Optional, Sequence
from src.retriever import SearchResult

# Standard separator between distinct document passages
PASSAGE_SEPARATOR: str = "\n\n------------------------------\n\n"


def build_context(results: Optional[Sequence[SearchResult]]) -> str:
    """Builds a structured evidence context string from retrieved search results.

    Formats each passage with 1-based source index, source page number, and chunk ID,
    preserving original retrieval ranking order without mutating the input objects.

    Args:
        results: Sequence of SearchResult objects from the retriever.

    Returns:
        str: Formatted evidence string, or an empty string if results is empty.

    Raises:
        TypeError: If results is None, a string/bytes, not iterable, or contains
            elements that are not SearchResult instances.
    """
    if results is None:
        raise TypeError("Input 'results' cannot be None. Expected a sequence of SearchResult objects.")

    if isinstance(results, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of SearchResult objects, got {type(results).__name__}."
        )

    # Validate elements and build passage blocks preserving retrieval ranking
    passage_blocks: list[str] = []
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, SearchResult):
            raise TypeError(
                f"All elements in 'results' must be SearchResult instances. "
                f"Found {type(item).__name__} at index {idx - 1}."
            )

        header = f"[Source {idx} | Page {item.page_number} | Chunk {item.chunk_id}]"
        passage_blocks.append(f"{header}\n{item.text}")

    if not passage_blocks:
        return ""

    return PASSAGE_SEPARATOR.join(passage_blocks)

