from __future__ import annotations

import re
from typing import Iterable


SECTION_PATTERN = re.compile(
    r"(?P<citation>"
    r"(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?[0-9A-Za-z.\-]+(?:\([0-9A-Za-z]+\))*"
    r"|CAO\s+DOC\s+[0-9A-Za-z.\-]+"
    r"|AIP\s+Australia\s+\d+(?:\.\d+)+"
    r"|AIP\s+\d+(?:\.\d+)+"
    r")",
    re.IGNORECASE,
)
CITATION_QUERY_PATTERN = re.compile(
    r"\b("
    r"(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?[0-9A-Za-z.\-]+(?:\([0-9A-Za-z]+\))*"
    r"|CAO\s+DOC\s+[0-9A-Za-z.\-]+"
    r"|AIP\s+Australia\s+\d+(?:\.\d+)+"
    r"|AIP\s+\d+(?:\.\d+)+"
    r")",
    re.IGNORECASE,
)
TABLE_PATTERN = re.compile(r"\bTable\s+\d+(?:\.\d+)+\b", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"\b(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\d+\b", re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    citations: list[str] = []
    for match in CITATION_QUERY_PATTERN.finditer(text):
        citation = " ".join(match.group(0).split())
        if citation not in citations:
            citations.append(citation)
    return citations


def infer_part(citation: str) -> str:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", citation)
    return match.group(1) if match else ""


def normalize_citation(raw: str) -> str:
    citation = " ".join(raw.split())
    citation = re.sub(r"^AIP Australia\s+", "AIP ", citation, flags=re.IGNORECASE)
    citation = re.sub(r"\s+", " ", citation).strip()
    return citation


def derive_precise_citation(section_text: str, fallback: str) -> str:
    leading_text = " ".join(section_text.split())[:250]

    aip_match = re.search(r"\bAIP\s+Australia\s+(\d+(?:\.\d+)+)\b", leading_text, re.IGNORECASE)
    if aip_match:
        return f"AIP {aip_match.group(1)}"

    generic_aip_match = re.search(r"\bAIP\s+(\d+(?:\.\d+)+)\b", leading_text, re.IGNORECASE)
    if generic_aip_match:
        return f"AIP {generic_aip_match.group(1)}"

    cao_doc_match = re.search(r"\bCAO\s+DOC\s+([0-9A-Za-z.\-]+)\b", leading_text, re.IGNORECASE)
    if cao_doc_match:
        return f"CAO DOC {cao_doc_match.group(1)}"

    normalized = normalize_citation(fallback)
    if re.search(r"\d", normalized):
        return normalized

    page_match = PAGE_PATTERN.search(section_text[:500])
    table_match = TABLE_PATTERN.search(section_text[:500])
    if page_match and table_match:
        return f"{normalized} {page_match.group(0)} {table_match.group(0)}".strip()
    if page_match:
        return f"{normalized} {page_match.group(0)}".strip()
    if table_match:
        return f"{normalized} {table_match.group(0)}".strip()
    return normalized


def derive_precise_title(section_text: str, citation: str) -> str:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if not lines:
        return citation

    first_line = re.sub(r"\s+", " ", lines[0])
    if first_line.lower().startswith(citation.lower()):
        remainder = first_line[len(citation) :].strip(" .:-")
        if remainder:
            return f"{citation} {remainder}"[:160]
        return citation[:160]

    # Common AIP shape:
    # line 1 -> "AIP Australia"
    # line 2 -> "1.6.5 Heading text..."
    if len(lines) > 1 and first_line.lower().startswith("aip"):
        second_line = re.sub(r"\s+", " ", lines[1])
        match = re.match(r"^(?P<section>\d+(?:\.\d+)+)\s*(?P<heading>.*)$", second_line)
        if match:
            heading = match.group("heading").strip(" .:-")
            if heading:
                return f"AIP {match.group('section')} {heading}"[:160]
            return f"AIP {match.group('section')}"[:160]

    # Fallback: look for "citation heading" in the first two lines combined.
    combined = re.sub(r"\s+", " ", " ".join(lines[:2]))
    if combined.lower().startswith(citation.lower()):
        remainder = combined[len(citation) :].strip(" .:-")
        if remainder:
            return f"{citation} {remainder}"[:160]
        return citation[:160]

    return first_line[:160]


def extract_page_ref(section_text: str) -> str:
    match = PAGE_PATTERN.search(section_text[:1000])
    return match.group(0) if match else ""


def extract_table_ref(section_text: str) -> str:
    match = TABLE_PATTERN.search(section_text[:1000])
    return match.group(0) if match else ""


def split_into_sections(text: str) -> list[dict]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return []

    sections: list[dict] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        citation = derive_precise_citation(section_text, match.group("citation"))
        title = derive_precise_title(section_text, citation)
        sections.append(
            {
                "regulation_id": citation,
                "citation": citation,
                "title": title,
                "part": infer_part(citation),
                "section_label": citation,
                "page_ref": extract_page_ref(section_text),
                "table_ref": extract_table_ref(section_text),
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
