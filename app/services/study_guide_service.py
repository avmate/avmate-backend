from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass
from urllib.parse import unquote, urljoin

import pdfplumber
import requests

from app.schemas import StudyGuideItem, StudyGuideResponse
from app.services.search_service import SearchService


DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
CASA_DOMAIN = "https://www.casa.gov.au"
CASA_SEARCH_URL = "https://www.casa.gov.au/search"


FORM_OVERRIDES: dict[str, str] = {
    "instrument rating - aeroplane": "https://www.casa.gov.au/resources-and-education/publications/flight-test-checklist-guidance-instrument-rating-aeroplane",
}


CRITERION_PATTERNS = [
    re.compile(r"^(?P<ref>[A-Z]{2,8}\.\d+[A-Z]?)\s*[-:]\s*(?P<text>.+)$"),
    re.compile(r"^(?P<ref>[A-Z]{2,8}\.\d+[A-Z]?)\s+(?P<text>.+)$"),
    re.compile(r"^(?P<ref>\d+(?:\.\d+){0,2}[A-Z]?)\s+(?P<text>.+)$"),
    re.compile(r"^(?P<ref>[A-Z]\d{1,2})\s+(?P<text>.+)$"),
    re.compile(r"^(?P<ref>\d+)\.\s+(?P<text>.+)$"),
]


@dataclass(frozen=True)
class FormSource:
    title: str
    page_url: str
    download_url: str


class StudyGuideService:
    def __init__(self, search_service: SearchService, timeout_seconds: int = 45) -> None:
        self._search_service = search_service
        self._timeout = timeout_seconds
        self._http = requests.Session()
        self._http.headers.update(
            {
                "User-Agent": "AvMateStudyBot/1.0 (+https://beta.avmate.com.au)",
                "Accept-Language": "en-AU,en;q=0.9",
            }
        )

    def build_study_guide(self, test_name: str, max_items: int) -> StudyGuideResponse:
        form_source = self._resolve_form_source(test_name)
        raw_text = self._extract_form_text(form_source.download_url)
        criteria = self._extract_criteria(raw_text, max_items=max_items)
        if not criteria:
            raise ValueError("Could not extract checklist criteria from the selected CASA form.")

        chronological_items: list[StudyGuideItem] = []
        for idx, item in enumerate(criteria, start=1):
            regulation_reference, confidence = self._map_regulation_reference(test_name, item["criterion"])
            chronological_items.append(
                StudyGuideItem(
                    order=idx,
                    section=item["section"],
                    form_reference=item["form_reference"],
                    criterion=item["criterion"],
                    regulation_reference=regulation_reference,
                    regulation_confidence=confidence,
                )
            )

        notes = [
            "Criteria are extracted in the form's visible order and mapped to indexed regulations.",
            "Regulation references with low confidence may require manual review against current CASA material.",
            "Always verify against the latest CASA form and legislation before operational use.",
        ]

        return StudyGuideResponse(
            test_name=test_name,
            form_title=form_source.title,
            form_page_url=form_source.page_url,
            form_download_url=form_source.download_url,
            chronological_items=chronological_items,
            notes=notes,
        )

    def _resolve_form_source(self, test_name: str) -> FormSource:
        normalized = self._normalize_label(test_name)
        override_url = FORM_OVERRIDES.get(normalized)
        if override_url:
            title, download_url = self._resolve_download_link(override_url)
            return FormSource(title=title or test_name, page_url=override_url, download_url=download_url)

        candidate_links = self._discover_casa_links(test_name)
        if not candidate_links:
            raise ValueError(
                f"Could not locate a CASA form page for '{test_name}'. "
                "Use the exact CASA test name or add a direct URL in FORM_OVERRIDES."
            )

        best_url = candidate_links[0]
        title, download_url = self._resolve_download_link(best_url)
        return FormSource(title=title or test_name, page_url=best_url, download_url=download_url)

    def _discover_casa_links(self, test_name: str) -> list[str]:
        discovered: list[str] = []
        for finder in (self._discover_via_duckduckgo, self._discover_via_casa_site_search):
            try:
                for url in finder(test_name):
                    if url not in discovered:
                        discovered.append(url)
            except Exception:
                continue

        scored = sorted(discovered, key=lambda url: self._link_score(url, test_name), reverse=True)
        return scored[:8]

    def _discover_via_duckduckgo(self, test_name: str) -> list[str]:
        query = f"site:casa.gov.au {test_name} flight test checklist"
        response = self._http.get(DUCKDUCKGO_HTML_URL, params={"q": query}, timeout=self._timeout)
        response.raise_for_status()
        html_text = response.text

        links: list[str] = []
        for encoded in re.findall(r"uddg=([^&\"'>]+)", html_text):
            url = unquote(encoded)
            if "casa.gov.au" not in url.lower():
                continue
            if url not in links:
                links.append(url)
        return links

    def _discover_via_casa_site_search(self, test_name: str) -> list[str]:
        query = f"{test_name} flight test checklist"
        response = self._http.get(CASA_SEARCH_URL, params={"keys": query}, timeout=self._timeout)
        response.raise_for_status()
        page_html = response.text

        links: list[str] = []
        for href in re.findall(r'href="([^"]+)"', page_html, flags=re.IGNORECASE):
            raw = html.unescape(href.strip())
            if not raw or raw.startswith("#") or raw.startswith("mailto:"):
                continue
            absolute = urljoin(CASA_DOMAIN, raw)
            lower = absolute.lower()
            if "casa.gov.au" not in lower:
                continue
            if ".pdf" not in lower and "checklist" not in lower and "flight-test" not in lower and "flight test" not in lower:
                continue
            if absolute not in links:
                links.append(absolute)
        return links

    def _link_score(self, url: str, test_name: str) -> float:
        label = self._normalize_label(test_name)
        url_lower = url.lower()
        score = 0.0
        for token in label.split():
            if token and token in url_lower:
                score += 1.0
        if "checklist" in url_lower:
            score += 2.0
        if "flight-test" in url_lower or "flight-test-checklist" in url_lower:
            score += 2.0
        if "resources-and-education/publications" in url_lower:
            score += 1.5
        if url_lower.endswith(".pdf"):
            score += 1.0
        return score

    def _resolve_download_link(self, page_url: str) -> tuple[str, str]:
        if page_url.lower().endswith(".pdf"):
            title = self._guess_title_from_url(page_url)
            return title, page_url

        response = self._http.get(page_url, timeout=self._timeout)
        response.raise_for_status()
        page_html = response.text

        title_match = re.search(r"<title>([^<]+)</title>", page_html, re.IGNORECASE)
        title = html.unescape(title_match.group(1)).strip() if title_match else self._guess_title_from_url(page_url)

        pdf_links = re.findall(r'href="([^"]+\.pdf(?:\?[^"]*)?)"', page_html, flags=re.IGNORECASE)
        if not pdf_links:
            pdf_links = re.findall(r"href='([^']+\.pdf(?:\?[^']*)?)'", page_html, flags=re.IGNORECASE)
        if not pdf_links:
            raise ValueError(f"Found CASA page but no downloadable PDF form link: {page_url}")

        download_candidates = [urljoin(page_url, html.unescape(link)) for link in pdf_links]
        download_url = max(download_candidates, key=self._pdf_link_score)
        return title, download_url

    def _extract_form_text(self, download_url: str) -> str:
        response = self._http.get(download_url, timeout=self._timeout)
        response.raise_for_status()
        content = response.content
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            page_chunks: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    page_chunks.append(text)
        return "\n".join(page_chunks)

    def _extract_criteria(self, text: str, max_items: int) -> list[dict]:
        lines = [self._normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        section = "General"
        current: dict | None = None
        criteria: list[dict] = []
        for line in lines:
            if self._is_noise(line):
                continue
            if self._is_section_heading(line):
                section = line
                current = None
                continue

            parsed = self._parse_criterion_line(line)
            if parsed:
                current = {
                    "section": section,
                    "form_reference": parsed[0],
                    "criterion": parsed[1],
                }
                criteria.append(current)
                continue

            if current and self._is_continuation_line(line):
                current["criterion"] = f"{current['criterion']} {line}".strip()
            else:
                current = None

            if len(criteria) >= max_items:
                break

        deduped: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for item in criteria:
            key = (item["form_reference"], item["criterion"][:120])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_items:
                break
        return deduped

    def _map_regulation_reference(self, test_name: str, criterion: str) -> tuple[str, int]:
        query = f"{test_name} {criterion[:220]}"
        try:
            result = self._search_service.search(query=query, top_k=1)
        except Exception:
            return "No reliable regulation match", 0
        if not result.references:
            return "No reliable regulation match", 0
        top = result.references[0]
        return top.citation, result.confidence

    def _normalize_label(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.lower()).strip()

    def _normalize_line(self, line: str) -> str:
        cleaned = html.unescape(line or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_noise(self, line: str) -> bool:
        lower = line.lower()
        if re.fullmatch(r"\d{1,3}", line):
            return True
        if "civil aviation safety authority" in lower:
            return True
        if "version" in lower and "page" in lower:
            return True
        if line.startswith("http://") or line.startswith("https://"):
            return True
        return False

    def _is_section_heading(self, line: str) -> bool:
        if len(line) < 4 or len(line) > 120:
            return False
        if re.match(r"^(AREA|SECTION|PHASE|PART)\s+[A-Z0-9]", line, re.IGNORECASE):
            return True
        alpha = sum(1 for ch in line if ch.isalpha())
        uppercase = sum(1 for ch in line if ch.isupper())
        return alpha > 0 and (uppercase / alpha) > 0.8 and not any(ch.isdigit() for ch in line[:4])

    def _parse_criterion_line(self, line: str) -> tuple[str, str] | None:
        for pattern in CRITERION_PATTERNS:
            match = pattern.match(line)
            if match:
                ref = match.group("ref").strip()
                text = match.group("text").strip(" -:")
                if len(text) < 5:
                    return None
                return ref, text
        return None

    def _is_continuation_line(self, line: str) -> bool:
        if len(line) < 4:
            return False
        if self._is_section_heading(line):
            return False
        if self._parse_criterion_line(line):
            return False
        return True

    def _guess_title_from_url(self, url: str) -> str:
        slug = url.rstrip("/").split("/")[-1]
        slug = slug.replace("-", " ").replace("_", " ")
        return slug.title() or "CASA Test Form"

    def _pdf_link_score(self, url: str) -> float:
        lower = url.lower()
        score = 0.0
        if "checklist" in lower:
            score += 2.0
        if "flight-test" in lower or "flight test" in lower:
            score += 2.0
        if "guidance" in lower:
            score += 1.0
        if lower.endswith(".pdf"):
            score += 1.0
        return score
