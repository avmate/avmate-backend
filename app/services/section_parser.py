from __future__ import annotations

import re
from typing import Iterable


SECTION_PATTERN = re.compile(
    r"(?P<citation>"
    r"(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?(?:[0-9]+[0-9A-Za-z.\-]*)(?:\([0-9A-Za-z]+\))*"
    r"|CAO\s+DOC\s+[0-9A-Za-z.\-]+"
    r"|AIP\s+Australia\s+\d+(?:\.\d+)+"
    r"|AIP\s+(?:(?:GEN|ENR|AD)\s+\d+(?:\.\d+)*(?:\s+\d+(?:\.\d+)*)?|\d+(?:\.\d+)+)"
    r")",
    re.IGNORECASE,
)
CITATION_QUERY_PATTERN = re.compile(
    r"\b("
    r"(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?(?:[0-9]+[0-9A-Za-z.\-]*)(?:\([0-9A-Za-z]+\))*"
    r"|CAO\s+DOC\s+[0-9A-Za-z.\-]+"
    r"|AIP\s+Australia\s+\d+(?:\.\d+)+"
    r"|AIP\s+(?:(?:GEN|ENR|AD)\s+\d+(?:\.\d+)*(?:\s+\d+(?:\.\d+)*)?|\d+(?:\.\d+)+)"
    r")",
    re.IGNORECASE,
)
ENR_SUBSECTION_QUERY_PATTERN = re.compile(
    r"\b(?:AIP\s+)?(?P<chapter>(?:GEN|ENR|AD)\s+\d+(?:\.\d+)?)(?:\s*-\s*\(?\d+\)?)?\s+"
    r"(?:subsection|section)\s+(?P<label>[1-9]\d?(?:\.\d+){1,4})\b",
    re.IGNORECASE,
)
TABLE_PATTERN = re.compile(r"\bTable\s+\d+(?:\.\d+)+\b", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"\b(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\(?\d+\)?\b", re.IGNORECASE)
AIP_SUBSECTION_PATTERN = re.compile(
    r"(?m)^(?P<label>\d+(?:\.\d+){1,4})\s+(?P<heading>[A-Za-z][^\n]{0,220})$"
)
LEGISLATION_HEADING_PATTERN = re.compile(
    r"(?m)^(?P<label>\d{1,3}(?:\.\d{1,3}){0,4}[A-Za-z]?(?:\([0-9A-Za-z]+\))?)\s+"
    r"(?P<heading>[A-Z][^\n]{2,220})$"
)
LEGISLATION_PREFIX_PATTERN = re.compile(r"\b(CASR|CAR|CAO|MOS|CAA)\b", re.IGNORECASE)
LEGISLATION_PAGE_NOISE_PATTERN = re.compile(
    r"(?:compilation\s+no\.|authorised\s+version|registered\s+\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)


def _normalize_ref(value: str) -> str:
    normalized = re.sub(r"\s*-\s*", " - ", value)
    normalized = re.sub(r"\(\s*(\d+)\s*\)", r"(\1)", normalized)
    return " ".join(normalized.split())


def _extract_aip_chapter(page_ref: str) -> tuple[str, str]:
    """Extract (chapter_type, chapter_num) from an AIP page marker.

    e.g. 'ENR 1.5 - 18' → ('ENR', '1.5')
         'GEN 3.4 - 1'  → ('GEN', '3.4')
         ''              → ('', '')
    """
    m = re.search(r"\b(GEN|ENR|AD)\s+(\d+(?:\.\d+)?)\s*-", page_ref, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2)
    return "", ""


def _build_aip_citation(label: str, page_ref: str) -> str:
    """Build a precise AIP citation using the chapter context from the nearest page marker.

    label='1.18', page_ref='ENR 1.5 - 18' → 'AIP ENR 1.5 1.18'
    label='3.4.1', page_ref='GEN 3.4 - 1' → 'AIP GEN 3.4.1'
    label='4.1',   page_ref='ENR 1.4 - 14'→ 'AIP ENR 1.4 4.1'
    label='1.18',  page_ref=''             → 'AIP 1.18' (fallback)
    """
    chapter_type, chapter_num = _extract_aip_chapter(page_ref)
    if not chapter_type:
        return f"AIP {label}"
    # If the label already starts with chapter_num (e.g. label='3.4.1', chapter='3.4'),
    # the chapter is embedded — just prefix the type: 'AIP GEN 3.4.1'
    if label.startswith(chapter_num) and (
        len(label) == len(chapter_num) or label[len(chapter_num)] == "."
    ):
        return f"AIP {chapter_type} {label}"
    # Otherwise the label is relative to the chapter: 'AIP ENR 1.5 1.18'
    return f"AIP {chapter_type} {chapter_num} {label}"


def extract_citations(text: str) -> list[str]:
    citations: list[str] = []
    for match in CITATION_QUERY_PATTERN.finditer(text):
        citation = " ".join(match.group(0).split())
        if citation not in citations:
            citations.append(citation)
    # Accept queries that specify AIP page blocks with explicit subsection labels,
    # e.g. "ENR 1.5 subsection 6.2" → "AIP ENR 1.5 6.2"
    for match in ENR_SUBSECTION_QUERY_PATTERN.finditer(text):
        chapter = " ".join(match.group("chapter").split())
        label = match.group("label")
        # Reuse _build_aip_citation by constructing a synthetic page_ref
        citation = _build_aip_citation(label, f"{chapter} - 0")
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

    fallback_prefix_match = re.match(r"^(CASR|CAR|CAO|MOS|CAA)\b", normalize_citation(fallback), re.IGNORECASE)
    if fallback_prefix_match:
        heading_match = re.search(
            r"(?m)^(?P<label>\d{1,3}(?:\.\d{1,3}){0,4}[A-Za-z]?(?:\([0-9A-Za-z]+\))?)\s+[A-Z]",
            section_text[:1600],
        )
        if heading_match:
            label = heading_match.group("label")
            if "." in label or len(re.sub(r"\D", "", label)) >= 2:
                return f"{fallback_prefix_match.group(1).upper()} {label}"

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


def extract_page_ref(section_text: str, full_text: str | None = None, section_start: int | None = None) -> str:
    if full_text is not None and section_start is not None:
        window_start = max(0, section_start - 12000)
        window_end = min(len(full_text), section_start + 4000)
        window_text = full_text[window_start:window_end]
        markers = list(PAGE_PATTERN.finditer(window_text))
        if markers:
            absolute_markers = [(window_start + m.start(), _normalize_ref(m.group(0))) for m in markers]
            preceding = [item for item in absolute_markers if item[0] <= section_start]
            if preceding:
                return preceding[-1][1]
            nearest = min(absolute_markers, key=lambda item: abs(item[0] - section_start))
            return nearest[1]

    match = PAGE_PATTERN.search(section_text[:12000])
    return _normalize_ref(match.group(0)) if match else ""


def extract_table_ref(section_text: str) -> str:
    match = TABLE_PATTERN.search(section_text[:12000])
    return _normalize_ref(match.group(0)) if match else ""


def split_into_sections(text: str, regulation_type: str = "") -> list[dict]:
    if regulation_type.upper() == "AIP":
        aip_sections = _split_aip_sections(text)
        if aip_sections:
            return aip_sections

    legislation_sections = _split_legislation_sections(text, regulation_type)
    if legislation_sections:
        return legislation_sections

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
                "page_ref": extract_page_ref(section_text, full_text=text, section_start=start),
                "table_ref": extract_table_ref(section_text),
                "text": section_text,
            }
        )
    return sections


def _split_aip_sections(text: str) -> list[dict]:
    matches = list(AIP_SUBSECTION_PATTERN.finditer(text))
    if not matches:
        return []

    # Pre-build label → section_text map for child aggregation.
    # Only include real sections (not TOC entries or TOC blocks).
    label_to_text: dict[str, str] = {}
    for index, match in enumerate(matches):
        if "....." in match.group("heading"):
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not _looks_like_toc_block(section_text):
            label_to_text[match.group("label")] = section_text

    raw: list[dict] = []
    for index, match in enumerate(matches):
        # Skip TOC entries: headings with dot-leaders like "1.18 Speed Restrictions...ENR 1.5-18"
        if "....." in match.group("heading"):
            continue

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if _looks_like_toc_block(section_text):
            continue

        label = match.group("label")
        if len(section_text) < 200:
            # Parent stub: synthesize by aggregating immediate children's text.
            child_prefix = label + "."
            child_texts = [t for lbl, t in label_to_text.items() if lbl.startswith(child_prefix)]
            if not child_texts:
                continue
            section_text = section_text + "\n\n" + "\n\n".join(child_texts)

        heading = " ".join(match.group("heading").split()).strip(" .:-")
        page_ref = extract_page_ref(section_text, full_text=text, section_start=start)
        citation = _build_aip_citation(label, page_ref)
        title = f"{citation} {heading}".strip()[:160] if heading else citation
        raw.append(
            {
                "regulation_id": citation,
                "citation": citation,
                "title": title,
                "part": infer_part(citation),
                "section_label": label,
                "page_ref": page_ref,
                "table_ref": extract_table_ref(section_text),
                "text": section_text,
            }
        )

    # Deduplicate by citation: keep the section with the most content (real section > TOC pointer)
    best: dict[str, dict] = {}
    order: list[str] = []
    for section in raw:
        citation = section["citation"]
        if citation not in best:
            best[citation] = section
            order.append(citation)
        elif len(section["text"]) > len(best[citation]["text"]):
            best[citation] = section

    return [best[c] for c in order]


def _infer_legislation_prefix(text: str, regulation_type: str) -> str:
    candidate = (regulation_type or "").strip().upper()
    if candidate in {"CASR", "CAR", "CAO", "MOS", "CAA"}:
        return candidate
    match = LEGISLATION_PREFIX_PATTERN.search(text[:2000])
    return match.group(1).upper() if match else ""


def _looks_like_legislation_heading(label: str, heading: str) -> bool:
    compact_heading = " ".join((heading or "").split())
    if not compact_heading or not re.search(r"[A-Za-z]{2,}", compact_heading):
        return False
    if "." not in label and len(re.sub(r"\D", "", label)) < 2:
        return False
    heading_lower = compact_heading.lower()
    noisy_prefixes = (
        "civil aviation safety regulations",
        "table of contents",
        "compilation no",
        "authorised version",
        "part ",
        "subpart ",
        "regulation ",
    )
    if heading_lower.startswith(noisy_prefixes):
        return False
    if "....." in compact_heading:
        return False
    return True


def _looks_like_toc_block(section_text: str) -> bool:
    lower = section_text[:900].lower()
    if "table of contents" in lower:
        return True

    lines = [line.strip() for line in section_text.splitlines()[:28] if line.strip()]
    if not lines:
        return False
    dot_leader_lines = sum(1 for line in lines if "....." in line)
    toc_like = sum(
        1
        for line in lines
        if re.match(r"^\d{1,3}(?:\.\d{1,3}){0,4}[A-Za-z]?(?:\([0-9A-Za-z]+\))?\s+\S", line)
        and not re.search(r"[.!?]\s", line)
    )
    numbered_paragraphs = sum(1 for line in lines if re.search(r"\(\d+\)", line))
    prose_like = sum(
        1
        for line in lines
        if len(line) >= 90 and re.search(r"[a-z]{3,}", line) and re.search(r"[.!?]", line)
    )
    if dot_leader_lines >= 2 and prose_like == 0:
        return True
    if toc_like >= 4 and prose_like <= 1 and numbered_paragraphs == 0:
        return True
    if LEGISLATION_PAGE_NOISE_PATTERN.search(lower) and toc_like >= 3 and prose_like == 0 and numbered_paragraphs == 0:
        return True
    return False


def _score_legislation_section_candidate(section: dict) -> float:
    text = " ".join((section.get("text", "") or "").split())
    title = " ".join((section.get("title", "") or "").split())
    if not text:
        return -9999.0

    score = float(len(text))
    if re.search(r"\(\d+\)", text):
        score += 600.0
    if re.search(r"\b(?:must|must not|authorised|contravenes|penalty|offence|requirement|required)\b", text, re.IGNORECASE):
        score += 140.0
    if re.search(r"\b(?:hours|aeronautical experience|flight time|take-offs|landings)\b", text, re.IGNORECASE):
        score += 80.0
    if re.search(r"\.{5,}", text):
        score -= 950.0
    if "table of contents" in text.lower():
        score -= 1200.0
    if LEGISLATION_PAGE_NOISE_PATTERN.search(text[:260]):
        score -= 120.0
    if len(text) < 180:
        score -= 220.0
    if title and "....." in title:
        score -= 300.0
    return score


def _dedupe_legislation_sections(sections: list[dict]) -> list[dict]:
    if not sections:
        return []

    best_by_citation: dict[str, tuple[float, dict]] = {}
    order: list[str] = []
    for section in sections:
        citation = section["citation"]
        candidate_score = _score_legislation_section_candidate(section)
        current = best_by_citation.get(citation)
        if current is None:
            best_by_citation[citation] = (candidate_score, section)
            order.append(citation)
            continue
        if candidate_score > current[0]:
            best_by_citation[citation] = (candidate_score, section)

    return [best_by_citation[citation][1] for citation in order if citation in best_by_citation]


def _extract_single_doc_identifier(text: str, prefix: str) -> str:
    """Return the document-level number for a single-document regulation (e.g. '48.1' for CAO 48.1).

    Some regulations (like individual CAOs) are a single document whose internal sections are
    numbered starting from 1.  Without a document-level qualifier the citations become ambiguous
    (e.g. 'CAO 4' could come from any CAO).  This helper extracts the qualifier from the
    document header so citations can be formed as 'CAO 48.1.4'.
    """
    if prefix != "CAO":
        return ""
    # Look for "CAO 48.1" or "Civil Aviation Order 48.1" near the top of the document.
    window = text[:4000]
    for pattern in (
        rf"\b{re.escape(prefix)}\s+(\d+(?:\.\d+)+)\b",
        r"\bCivil Aviation Order\s+(\d+(?:\.\d+)+)\b",
    ):
        m = re.search(pattern, window, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def _split_legislation_sections(text: str, regulation_type: str) -> list[dict]:
    prefix = _infer_legislation_prefix(text, regulation_type)
    if not prefix:
        return []

    matches = list(LEGISLATION_HEADING_PATTERN.finditer(text))
    if len(matches) < 3:
        return []

    # For single-document CAOs, prepend the document number (e.g. "48.1") so that
    # "CAO 4" becomes "CAO 48.1.4" — unambiguous and searchable.
    doc_number = _extract_single_doc_identifier(text, prefix)

    sections: list[dict] = []
    for index, match in enumerate(matches):
        label = match.group("label")
        heading = " ".join(match.group("heading").split()).strip(" .:-")
        if not _looks_like_legislation_heading(label, heading):
            continue

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) < 120:
            continue
        if _looks_like_toc_block(section_text):
            continue

        # Build citation: if doc_number is known and label is a short relative number
        # (not already containing the doc_number), prepend it.
        if doc_number and not label.startswith(doc_number) and re.match(r"^\d{1,3}(?:\.\d{1,3}){0,3}$", label):
            citation = f"{prefix} {doc_number}.{label}"
        else:
            citation = f"{prefix} {label}"
        title = f"{citation} {heading}".strip()[:160]
        sections.append(
            {
                "regulation_id": citation,
                "citation": citation,
                "title": title,
                "part": infer_part(citation),
                "section_label": label,
                "page_ref": extract_page_ref(section_text, full_text=text, section_start=start),
                "table_ref": extract_table_ref(section_text),
                "text": section_text,
            }
        )

    sections = _dedupe_legislation_sections(sections)

    if len(sections) < 5:
        return []
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


# Ordered separators for legal/regulatory text: prefer structural boundaries
# before falling back to arbitrary character splits.
_LEGAL_SEPARATORS = ["\n\n", "\n", "; ", " "]


def chunk_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks that respect legal paragraph boundaries.

    Tries \\n\\n, \\n, '; ', then ' ' as split points in that order.
    After chunking, each chunk (except the first) is prefixed with the tail
    of the prior chunk so context carries across boundaries (overlap).
    """
    if not text or not text.strip():
        return []

    def _split(txt: str, separators: list[str]) -> list[str]:
        if len(txt) <= chunk_size:
            stripped = txt.strip()
            return [stripped] if stripped else []

        for i, sep in enumerate(separators):
            if sep not in txt:
                continue
            pieces = txt.split(sep)
            chunks: list[str] = []
            current = ""
            for piece in pieces:
                if not piece.strip():
                    continue
                candidate = (current + sep + piece) if current else piece
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(piece) > chunk_size:
                        chunks.extend(_split(piece, separators[i + 1 :]))
                        current = ""
                    else:
                        current = piece
            if current.strip():
                chunks.append(current.strip())
            if chunks:
                return chunks

        # No separator worked — hard split
        step = max(chunk_size - overlap, 1)
        return [
            txt[j : j + chunk_size].strip()
            for j in range(0, len(txt), step)
            if txt[j : j + chunk_size].strip()
        ]

    raw = _split(text, _LEGAL_SEPARATORS)
    if not raw or overlap <= 0:
        return raw

    # Prefix each chunk after the first with the tail of the previous chunk
    result = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-overlap:].strip()
        result.append((tail + " " + raw[i]).strip() if tail else raw[i])
    return result
