"""Citation extraction and validation module for DocuMind.

Phase 5.4: Deterministically extracts [Page N] citations from generated text,
validates them against retrieved source pages, and identifies invalid or missing citations.
"""

import re
from typing import Any, List, Sequence, Set, Tuple


# Regex pattern to match bracketed page citations case-insensitively:
# Matches [Page 3], [page 12], [PAGE 42], [Page 3, 5], [Page 3, Page 5], [Pages 3 and 4]
CITATION_BRACKET_PATTERN = re.compile(r"\[pages?\s+([^\]]+)\]", re.IGNORECASE)
PAGE_DIGIT_PATTERN = re.compile(r"\b\d+\b")


def extract_citations(text: str) -> List[int]:
    """Extracts unique cited 1-based page numbers from text in order of appearance.

    Supports single citations ([Page 3]), multiple citations ([Page 3] [Page 4]),
    comma-separated citations ([Page 3, 5]), and case-insensitive variants ([page 3]).

    Args:
        text: Input string (e.g. an LLM-generated answer).

    Returns:
        List[int]: Deduplicated list of 1-based page numbers in order of first appearance.
        Returns an empty list if text is empty or contains no citations.
    """
    if not text or not isinstance(text, str):
        return []

    cited_pages: List[int] = []
    seen: Set[int] = set()

    for match in CITATION_BRACKET_PATTERN.finditer(text):
        bracket_content = match.group(1)
        # Extract all integer numbers within the citation bracket
        for num_str in PAGE_DIGIT_PATTERN.findall(bracket_content):
            page_num = int(num_str)
            if page_num > 0 and page_num not in seen:
                seen.add(page_num)
                cited_pages.append(page_num)

    return cited_pages


def has_citations(text: str) -> bool:
    """Returns True if the input text contains at least one valid [Page N] citation."""
    return len(extract_citations(text)) > 0


def get_available_pages(sources: Sequence[Any]) -> List[int]:
    """Extracts a sorted list of unique 1-based page numbers present in retrieved sources.

    Args:
        sources: Sequence of SearchResult objects (or objects with a page_number attribute).

    Returns:
        List[int]: Sorted list of unique page numbers. Returns [] if sources is empty.
    """
    if not sources:
        return []

    pages: Set[int] = set()
    for src in sources:
        page = getattr(src, "page_number", None)
        if page is not None and isinstance(page, int) and page > 0:
            pages.add(page)

    return sorted(pages)


def validate_page_citations(
    cited_pages: Sequence[int],
    available_pages: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Validates cited page numbers against available source page numbers.

    Deduplicates cited pages and distinguishes valid citations from invalid citations.

    Args:
        cited_pages: Sequence of page numbers cited in the text.
        available_pages: Sequence of page numbers available in the retrieved sources.

    Returns:
        Tuple[List[int], List[int]]:
            - valid_citations: Cited page numbers that exist in available_pages.
            - invalid_citations: Cited page numbers absent from available_pages.
    """
    available_set: Set[int] = set(available_pages) if available_pages else set()
    valid_citations: List[int] = []
    invalid_citations: List[int] = []
    seen: Set[int] = set()

    for p in cited_pages:
        if p not in seen:
            seen.add(p)
            if p in available_set:
                valid_citations.append(p)
            else:
                invalid_citations.append(p)

    return valid_citations, invalid_citations
