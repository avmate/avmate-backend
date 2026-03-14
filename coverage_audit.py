from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pdfplumber
import requests

from app.config import get_settings
from app.services.r2_catalog import RegulationCatalog
from app.services.section_parser import chunk_chars, split_into_sections


AIP_PAGE_MARKER_RE = re.compile(r"\b(GEN|ENR|AD)\s+(\d+(?:\.\d+)?)\s*-\s*\(?\d+\)?", re.IGNORECASE)
CASR_PART_HEADER_RE = re.compile(r"(?m)^\s*Part\s+(\d+[A-Z]?)\b")
CASR_CITATION_RE = re.compile(r"^CASR\s+(\d+)")


@dataclass
class DocumentAudit:
    path: str
    doc_type: str
    title: str
    pages: int
    sections: int
    chunks: int


def _download_pdf(url: str, timeout: int) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _extract_pdf_text(raw_bytes: bytes) -> tuple[int, str]:
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return page_count, "\n".join(text_parts)


def _count_chunks(text: str, chunk_size_words: int, overlap_words: int) -> int:
    chunk_size = chunk_size_words * 6
    overlap = overlap_words * 6
    return len(list(chunk_chars(text, chunk_size=chunk_size, overlap=overlap)))


def _audit_aip_chapters(text: str, sections: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    page_markers: dict[str, set[str]] = defaultdict(set)
    for match in AIP_PAGE_MARKER_RE.finditer(text):
        chapter = f"AIP {match.group(1).upper()} {match.group(2)}"
        page_markers[chapter].add(match.group(0))

    parsed_counts: Counter[str] = Counter()
    for section in sections:
        citation = section.get("citation", "")
        match = re.match(r"^(AIP\s+(?:GEN|ENR|AD)\s+\d+(?:\.\d+)?)\b", citation, re.IGNORECASE)
        if match:
            parsed_counts[match.group(1).upper()] += 1

    return (
        {chapter: len(markers) for chapter, markers in page_markers.items()},
        dict(parsed_counts),
    )


def _audit_casr_parts(text: str, sections: list[dict]) -> tuple[set[str], Counter[str]]:
    expected_parts = {match.group(1) for match in CASR_PART_HEADER_RE.finditer(text)}
    parsed_parts: Counter[str] = Counter()
    for section in sections:
        citation = section.get("citation", "")
        match = CASR_CITATION_RE.match(citation)
        if match:
            parsed_parts[match.group(1)] += 1
    return expected_parts, parsed_parts


def main() -> int:
    settings = get_settings()
    catalog = RegulationCatalog(
        base_url=settings.r2_base_url,
        manifest_path=settings.local_manifest_path,
        manifest_url=settings.r2_manifest_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    documents = catalog.load_documents()

    audits: list[DocumentAudit] = []
    chunks_by_type: Counter[str] = Counter()
    sections_by_type: Counter[str] = Counter()
    aip_page_counts: dict[str, int] = {}
    aip_parsed_counts: dict[str, int] = {}
    casr_expected_parts: set[str] = set()
    casr_parsed_parts: Counter[str] = Counter()
    failures: list[str] = []

    print(f"Coverage audit starting for {len(documents)} manifest documents")
    print(f"Manifest source: {catalog.source_label()}")
    print()

    for doc in documents:
        print(f"[scan] {doc['type']:5s} {doc['path']}")
        try:
            raw_bytes = _download_pdf(doc["source_url"], settings.request_timeout_seconds)
            pages, text = _extract_pdf_text(raw_bytes)
            sections = split_into_sections(text, regulation_type=doc.get("type", ""))
            chunks = sum(
                _count_chunks(
                    section["text"],
                    settings.chunk_size_words,
                    settings.chunk_overlap_words,
                )
                for section in sections
            )
            audits.append(
                DocumentAudit(
                    path=doc["path"],
                    doc_type=doc["type"],
                    title=doc["title"],
                    pages=pages,
                    sections=len(sections),
                    chunks=chunks,
                )
            )
            sections_by_type[doc["type"]] += len(sections)
            chunks_by_type[doc["type"]] += chunks

            if doc["type"] == "AIP":
                aip_page_counts, aip_parsed_counts = _audit_aip_chapters(text, sections)
            elif doc["type"] == "CASR":
                expected, parsed = _audit_casr_parts(text, sections)
                casr_expected_parts |= expected
                casr_parsed_parts.update(parsed)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{doc['path']}: {exc}")
            print(f"  failed: {exc}")

    print()
    print("=== Manifest Coverage ===")
    print(f"Documents scanned: {len(audits)}/{len(documents)}")
    print(f"Failures: {len(failures)}")
    for audit in audits:
        print(
            f"  {audit.doc_type:7s} pages={audit.pages:4d} sections={audit.sections:5d} "
            f"chunks={audit.chunks:5d}  {audit.path}"
        )

    print()
    print("=== Totals By Type ===")
    for doc_type in sorted(sections_by_type):
        print(
            f"  {doc_type:7s} sections={sections_by_type[doc_type]:5d} "
            f"chunks={chunks_by_type[doc_type]:5d}"
        )

    if aip_page_counts:
        print()
        print("=== AIP Chapter Coverage ===")
        zero_parsed = []
        low_density = []
        for chapter in sorted(aip_page_counts):
            page_markers = aip_page_counts[chapter]
            parsed = aip_parsed_counts.get(chapter, 0)
            print(f"  {chapter:25s} page_markers={page_markers:3d} parsed_sections={parsed:4d}")
            if parsed == 0:
                zero_parsed.append(chapter)
            elif parsed < max(1, int(page_markers * 0.3)):
                low_density.append((chapter, page_markers, parsed))

        print()
        print(f"AIP chapters with zero parsed sections: {len(zero_parsed)}")
        for chapter in zero_parsed:
            print(f"  ZERO {chapter}")

        print(f"AIP chapters with low parsed density: {len(low_density)}")
        for chapter, pages, parsed in low_density:
            print(f"  LOW  {chapter:25s} page_markers={pages:3d} parsed_sections={parsed:4d}")

    if casr_expected_parts:
        print()
        print("=== CASR Part Coverage ===")
        missing_parts = []
        for part in sorted(casr_expected_parts, key=lambda value: (int(re.match(r"\d+", value).group(0)), value)):
            parsed = casr_parsed_parts.get(part, 0)
            print(f"  Part {part:4s} parsed_sections={parsed:4d}")
            if parsed == 0:
                missing_parts.append(part)
        print()
        print(f"CASR parts with zero parsed sections: {len(missing_parts)}")
        for part in missing_parts:
            print(f"  ZERO Part {part}")

    if failures:
        print()
        print("=== Failures ===")
        for failure in failures:
            print(f"  {failure}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
