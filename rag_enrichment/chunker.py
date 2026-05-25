"""Split Gemini-extracted markdown into page-wise chunks."""

import re

PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
CHUNK_MARKER_RE = re.compile(r"<!--\s*chunk:\s*pages\s+\d+-\d+\s*-->", re.IGNORECASE)


def split_into_pages(markdown: str) -> list[dict]:
    """Split markdown on <!-- page N --> markers.

    Returns list of {"page_number": int, "text": str}.
    Skips empty pages and strips chunk markers.
    """
    cleaned = CHUNK_MARKER_RE.sub("", markdown)

    parts = PAGE_MARKER_RE.split(cleaned)

    # parts[0] = header text before first page marker (skip)
    # parts[1] = "1" (page number), parts[2] = page 1 text
    # parts[3] = "2" (page number), parts[4] = page 2 text, etc.

    pages = []
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            pages.append({"page_number": page_num, "text": text})

    return pages
