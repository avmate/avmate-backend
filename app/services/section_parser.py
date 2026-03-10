from __future__ import annotations

import re
from typing import Iterable


SECTION_PATTERN = re.compile(
    r"(?P<citation>(?:CASR|CAR|CAO|AIP|MOS|CAA)\s+(?:Part\s+)?[0-9A-Za-z.\-]+(?:\([0-9A-Za-z]+\))*)",
    re.IGNORECASE,
)


def split_into_sections(text: str) -> list[dict]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return []

    sections: list[dict] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        citation = " ".join(match.group("citation").split())
        title = section_text.splitlines()[0][:160]
        sections.append(
            {
                "regulation_id": citation,
                "citation": citation,
                "title": title,
                "text": section_text,
            }
        )
    return sections


def chunk_words(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

