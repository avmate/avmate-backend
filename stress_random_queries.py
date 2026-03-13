from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests
from sqlalchemy import select

from app.config import get_settings, load_manifest_file
from app.db import session_scope
from app.models import RegulationSection


BASE_URL = os.getenv("AVMATE_BASE_URL", "https://avmate-backend-production.up.railway.app").rstrip("/")
QUERY_COUNT = int(os.getenv("AVMATE_RANDOM_QUERY_COUNT", "1200"))
TARGET_ACCURACY = float(os.getenv("AVMATE_TARGET_ACCURACY", "0.95"))
MAX_ROUNDS = int(os.getenv("AVMATE_MAX_ROUNDS", "10"))
REQUEST_TIMEOUT = int(os.getenv("AVMATE_REQUEST_TIMEOUT_SECONDS", "90"))
RETRY_ATTEMPTS = int(os.getenv("AVMATE_CASE_RETRY_ATTEMPTS", "5"))
RETRY_DELAY_SECONDS = float(os.getenv("AVMATE_CASE_RETRY_DELAY_SECONDS", "8"))
QUERY_DELAY_SECONDS = float(os.getenv("AVMATE_QUERY_DELAY_SECONDS", "0.55"))
TOP_K = int(os.getenv("AVMATE_RANDOM_QUERY_TOP_K", "5"))
REPORT_PATH = Path(os.getenv("AVMATE_RANDOM_QUERY_REPORT", "stress_random_queries_report.json"))
MIN_QUERIES_PER_MANUAL = int(os.getenv("AVMATE_MIN_QUERIES_PER_MANUAL", "1"))
QUERY_BANK_PATHS = os.getenv(
    "AVMATE_QUERY_BANK_PATHS",
    "data/query_coverage_bank.csv,data/query_coverage_bank_law.csv",
)

NON_AIP_CITATION_PATTERN = re.compile(
    r"^(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?[0-9]+[0-9A-Za-z.\-]*(?:\([0-9A-Za-z]+\))*$|^CAO\s+DOC\s+[0-9A-Za-z.\-]+$",
    re.IGNORECASE,
)
AIP_OUTPUT_CITATION_PATTERN = re.compile(
    r"^AIP\s+(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\(?\d+\)?\s+subsection\s+[0-9]+(?:\.[0-9]+){1,4}(?:\.\s*Table\s+\d+(?:\.\d+)*)?$",
    re.IGNORECASE,
)
SUBSECTION_PATTERN = re.compile(r"\bAIP\s+([0-9]+(?:\.[0-9]+){1,4})\b", re.IGNORECASE)
REFERENCE_CITATION_HINT_PATTERN = re.compile(
    r"\b("
    r"(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?[0-9]+[0-9A-Za-z.\-]*(?:\([0-9A-Za-z]+\))*"
    r"|CAO\s+DOC\s+[0-9A-Za-z.\-]+"
    r"|AIP\s+(?:GEN|ENR|AD)\s+\d+(?:\.\d+)?(?:\s*-\s*\(?\d+\)?)?(?:\s+subsection\s+[0-9]+(?:\.[0-9]+){1,4})?"
    r"|AIP\s+[0-9]+(?:\.[0-9]+){1,4}"
    r")\b",
    re.IGNORECASE,
)
PART_HINT_PATTERN = re.compile(r"\bpart\s+(\d+)\b", re.IGNORECASE)
OUT_OF_SCOPE_HINT_PATTERNS = (
    re.compile(r"\b(?:hr|behavioral|operational|general practice)\b", re.IGNORECASE),
    re.compile(r"\b(?:casa workbook|syllabus|guide|advisory circular)\b", re.IGNORECASE),
    re.compile(r"\b(?:bo?m|meteorology|human factors)\b", re.IGNORECASE),
    re.compile(r"\b(?:airservices|naips|atsb|transport safety investigation act)\b", re.IGNORECASE),
)


def _load_available_catalog_tokens() -> set[str]:
    settings = get_settings()
    if not settings.local_manifest_path.exists():
        return set()

    tokens: set[str] = set()
    try:
        manifest = load_manifest_file(settings.local_manifest_path)
    except Exception:
        return set()

    for item in manifest:
        for value in (item.get("path", ""), item.get("type", ""), item.get("title", "")):
            text = " ".join(str(value).split()).lower()
            if not text:
                continue
            tokens.add(text)
            for token in re.findall(r"[a-z0-9./_-]+", text):
                if len(token) >= 3:
                    tokens.add(token)
    return tokens


AVAILABLE_CATALOG_TOKENS = _load_available_catalog_tokens()
CATALOG_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "mos": (" mos", "manual of standards", "part 91 mos", "part 61 mos", "part 121 mos"),
    "vfrg": ("vfrg",),
    "ersa": ("ersa",),
    "dap": ("dap",),
    "airservices": ("airservices", "naips"),
    "atsb": ("atsb", "transport safety investigation act", "asir"),
    "bom": ("bom", "graphical area forecast", "gfa"),
    "workbook": ("workbook",),
    "syllabus": ("syllabus",),
    "advisory": ("advisory circular",),
    "website": ("website", "casa website"),
}


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_citation: str
    regulation_type: str
    source_file: str
    expected_hint: str = ""
    category: str = ""
    is_bank_case: bool = False
    out_of_scope: bool = False


@dataclass
class QueryResult:
    ok: bool
    reason: str
    query: str
    expected_citation: str
    citations: list[str]
    status_code: int
    is_bank_case: bool = False
    out_of_scope: bool = False
    hint_match: bool = False


def normalize_spaces(value: str) -> str:
    return " ".join((value or "").split())


def is_precise_citation(citation: str, regulation_type_hint: str = "") -> bool:
    normalized = normalize_spaces(citation)
    if not normalized:
        return False
    if " subsection " in normalized.lower():
        return bool(AIP_OUTPUT_CITATION_PATTERN.match(normalized))
    if regulation_type_hint.upper() == "AIP":
        return False
    return bool(NON_AIP_CITATION_PATTERN.match(normalized))


def expected_output_citation(row: RegulationSection) -> str:
    citation = normalize_spaces(row.citation)
    regulation_type = normalize_spaces(row.regulation_type).upper()
    if regulation_type == "AIP":
        subsection_match = SUBSECTION_PATTERN.search(citation)
        label = subsection_match.group(1) if subsection_match else ""
        page_ref = normalize_spaces(row.page_ref)
        if not (label and page_ref):
            return ""
        out = f"AIP {page_ref} subsection {label}"
        table_ref = normalize_spaces(row.table_ref)
        if table_ref:
            out = f"{out}. {table_ref}"
        return out
    return citation


def build_query_case(row: RegulationSection) -> QueryCase | None:
    expected = expected_output_citation(row)
    if not expected:
        return None
    regulation_type = normalize_spaces(row.regulation_type).upper()
    citation = normalize_spaces(row.citation)

    if regulation_type == "AIP":
        subsection = ""
        page_ref = normalize_spaces(row.page_ref)
        match = SUBSECTION_PATTERN.search(citation)
        if match:
            subsection = match.group(1)
        templates = [
            f"What does {expected} say?",
            f"Summarise {expected}",
            f"Explain {expected} in plain english.",
        ]
        if subsection:
            templates.append(f"What does AIP {subsection} require?")
        if subsection and page_ref:
            templates.append(f"What does {page_ref} subsection {subsection} say?")
    else:
        templates = [
            f"What does {expected} require?",
            f"Summarise {expected}",
            f"Explain {expected} in plain english.",
        ]

    return QueryCase(
        query=random.choice(templates),
        expected_citation=expected,
        regulation_type=regulation_type,
        source_file=normalize_spaces(row.source_file),
    )


def _sample_manual_rows(rows: list[RegulationSection], n: int) -> list[RegulationSection]:
    if not rows or n <= 0:
        return []
    if len(rows) >= n:
        return random.sample(rows, n)
    sampled = list(rows)
    while len(sampled) < n:
        sampled.append(random.choice(rows))
    return sampled


def infer_expected_citation_from_hint(hint: str) -> str:
    normalized_hint = normalize_spaces(hint)
    if not normalized_hint:
        return ""
    match = REFERENCE_CITATION_HINT_PATTERN.search(normalized_hint)
    if not match:
        return ""
    candidate = normalize_spaces(match.group(0))
    if candidate.upper().startswith("AIP "):
        return candidate
    if candidate.lower().startswith("part "):
        return ""
    return candidate


def _hint_matches_citations(hint: str, citations: list[str]) -> bool:
    normalized_hint = normalize_spaces(hint).lower()
    if not normalized_hint:
        return True
    if not citations:
        return False

    citations_lower = [citation.lower() for citation in citations]
    expected_citation = infer_expected_citation_from_hint(hint).lower()
    if expected_citation:
        return expected_citation in citations_lower

    if "aip" in normalized_hint and any(citation.startswith("aip ") for citation in citations_lower):
        return True
    for token in ("casr", "car", "cao", "mos", "caa"):
        if token in normalized_hint and any(citation.startswith(f"{token} ") for citation in citations_lower):
            return True

    part_match = PART_HINT_PATTERN.search(normalized_hint)
    if part_match:
        part_value = part_match.group(1)
        if any(re.search(rf"\b{re.escape(part_value)}(?:\.\d+)?\b", citation) for citation in citations_lower):
            return True

    # If hint is generic educational context, citation match is optional.
    if any(pattern.search(normalized_hint) for pattern in OUT_OF_SCOPE_HINT_PATTERNS):
        return True
    return False


def _is_out_of_scope_case(reference_hint: str, category: str) -> bool:
    hint = normalize_spaces(reference_hint).lower()
    cat = normalize_spaces(category).lower()
    combined = f"{hint} {cat}".strip()
    if not combined:
        return False
    if any(pattern.search(combined) for pattern in OUT_OF_SCOPE_HINT_PATTERNS):
        return True

    for family, aliases in CATALOG_FAMILY_ALIASES.items():
        if any(alias in combined for alias in aliases):
            if not any(alias.strip() in token for alias in aliases for token in AVAILABLE_CATALOG_TOKENS):
                return True
    return False


def load_query_bank_cases(paths_csv: str) -> list[QueryCase]:
    paths = [Path(item.strip()) for item in paths_csv.split(",") if item.strip()]
    cases: list[QueryCase] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question = normalize_spaces(
                    row.get("question") or row.get("Question") or row.get("query") or row.get("Query") or ""
                )
                if not question:
                    continue
                category = normalize_spaces(row.get("category") or row.get("Category") or "query_bank")
                hint = normalize_spaces(
                    row.get("reference_context")
                    or row.get("Reference/Context")
                    or row.get("primary_reference")
                    or row.get("Primary Reference")
                    or row.get("reference")
                    or ""
                )
                expected_citation = infer_expected_citation_from_hint(hint)
                regulation_type = ""
                if expected_citation:
                    regulation_type = expected_citation.split()[0].upper()
                out_of_scope = _is_out_of_scope_case(hint, category)
                cases.append(
                    QueryCase(
                        query=question,
                        expected_citation=expected_citation,
                        regulation_type=regulation_type,
                        source_file=f"bank::{path.name}",
                        expected_hint=hint,
                        category=category,
                        is_bank_case=True,
                        out_of_scope=out_of_scope,
                    )
                )
    return cases


def load_random_sections(count: int) -> list[RegulationSection]:
    with session_scope() as session:
        rows = session.scalars(
            select(RegulationSection).where(RegulationSection.citation.is_not(None), RegulationSection.text.is_not(None))
        ).all()
    usable: list[RegulationSection] = []
    for row in rows:
        if not row.citation or not row.text:
            continue
        expected = expected_output_citation(row)
        if not expected:
            continue
        if not is_precise_citation(expected, row.regulation_type):
            continue
        usable.append(row)
    if not usable:
        return []

    by_manual: dict[str, list[RegulationSection]] = {}
    for row in usable:
        manual_key = normalize_spaces(row.source_file) or "unknown_source"
        by_manual.setdefault(manual_key, []).append(row)
    manual_keys = list(by_manual.keys())
    random.shuffle(manual_keys)

    if not manual_keys:
        return [random.choice(usable) for _ in range(count)]

    selected: list[RegulationSection] = []
    guaranteed_per_manual = max(1, MIN_QUERIES_PER_MANUAL)
    guaranteed_total = len(manual_keys) * guaranteed_per_manual
    if count >= guaranteed_total:
        for manual_key in manual_keys:
            selected.extend(_sample_manual_rows(by_manual[manual_key], guaranteed_per_manual))
    else:
        # If query budget is lower than total manuals, still maximize manual coverage.
        for manual_key in manual_keys:
            if len(selected) >= count:
                break
            selected.extend(_sample_manual_rows(by_manual[manual_key], 1))

    if len(selected) > count:
        random.shuffle(selected)
        selected = selected[:count]

    weighted_manual_keys: list[str] = []
    for manual_key, manual_rows in by_manual.items():
        weight = min(25, max(1, len(manual_rows)))
        weighted_manual_keys.extend([manual_key] * weight)
    if not weighted_manual_keys:
        weighted_manual_keys = manual_keys

    while len(selected) < count:
        manual_key = random.choice(weighted_manual_keys)
        selected.append(random.choice(by_manual[manual_key]))
    random.shuffle(selected)
    return selected[:count]


def query_api(case: QueryCase) -> QueryResult:
    url = f"{BASE_URL}/search"
    payload = {"query": case.query, "top_k": TOP_K}
    response = None
    for attempt in range(1, max(2, RETRY_ATTEMPTS + 1)):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            if attempt >= RETRY_ATTEMPTS + 1:
                return QueryResult(
                    ok=False,
                    reason=f"request_error:{exc}",
                    query=case.query,
                    expected_citation=case.expected_citation,
                    citations=[],
                    status_code=0,
                    is_bank_case=case.is_bank_case,
                    out_of_scope=case.out_of_scope,
                )
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        if response.status_code == 200:
            break
        if response.status_code in {429, 503} and attempt < RETRY_ATTEMPTS + 1:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        return QueryResult(
            ok=False,
            reason=f"http_{response.status_code}",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=[],
            status_code=response.status_code,
            is_bank_case=case.is_bank_case,
            out_of_scope=case.out_of_scope,
        )

    payload = response.json() if response is not None else {}
    citations = [normalize_spaces(item) for item in (payload.get("citations") or []) if normalize_spaces(item)]
    if not citations:
        return QueryResult(
            ok=False,
            reason="empty_citations",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=[],
            status_code=200,
            is_bank_case=case.is_bank_case,
            out_of_scope=case.out_of_scope,
        )

    if len({c.lower() for c in citations}) != len(citations):
        return QueryResult(
            ok=False,
            reason="duplicate_citations",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
            is_bank_case=case.is_bank_case,
            out_of_scope=case.out_of_scope,
        )

    if not all(is_precise_citation(citation, case.regulation_type) for citation in citations):
        return QueryResult(
            ok=False,
            reason="malformed_citation",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
            is_bank_case=case.is_bank_case,
            out_of_scope=case.out_of_scope,
        )

    hint_match = _hint_matches_citations(case.expected_hint or case.expected_citation, citations)
    if case.expected_citation and case.expected_citation not in citations:
        if case.is_bank_case:
            return QueryResult(
                ok=True,
                reason="ok_hint_miss",
                query=case.query,
                expected_citation=case.expected_citation,
                citations=citations,
                status_code=200,
                is_bank_case=True,
                out_of_scope=case.out_of_scope,
                hint_match=hint_match,
            )
        return QueryResult(
            ok=False,
            reason="expected_missing",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
            is_bank_case=case.is_bank_case,
            out_of_scope=case.out_of_scope,
            hint_match=hint_match,
        )

    return QueryResult(
        ok=True,
        reason="ok" if hint_match else "ok_hint_miss",
        query=case.query,
        expected_citation=case.expected_citation,
        citations=citations,
        status_code=200,
        is_bank_case=case.is_bank_case,
        out_of_scope=case.out_of_scope,
        hint_match=hint_match,
    )


def run_round(round_number: int, cases: list[QueryCase]) -> tuple[float, list[QueryResult]]:
    print(f"\nRound {round_number}: {len(cases)} queries")
    results: list[QueryResult] = []
    passed = 0
    for idx, case in enumerate(cases, start=1):
        result = query_api(case)
        results.append(result)
        if result.ok:
            passed += 1
        if idx % 100 == 0 or idx == len(cases):
            accuracy = passed / idx
            print(f"  progress {idx}/{len(cases)} accuracy={accuracy:.4f}")
        if QUERY_DELAY_SECONDS > 0:
            time.sleep(QUERY_DELAY_SECONDS)
    accuracy = passed / max(len(cases), 1)
    return accuracy, results


def main() -> int:
    print(f"Target URL: {BASE_URL}")
    print(f"Target accuracy: {TARGET_ACCURACY:.2%}")
    print(f"Queries per round: {QUERY_COUNT}")
    print(f"Minimum per manual: {MIN_QUERIES_PER_MANUAL}")
    print(f"Query banks: {QUERY_BANK_PATHS}")

    sampled_rows = load_random_sections(QUERY_COUNT)
    if not sampled_rows:
        print("No usable indexed sections found for stress generation.")
        return 1
    manuals = sorted({normalize_spaces(row.source_file) or "unknown_source" for row in sampled_rows})
    print(f"Manuals covered in sample set: {len(manuals)}")
    bank_cases = load_query_bank_cases(QUERY_BANK_PATHS)
    print(f"Loaded bank queries: {len(bank_cases)}")

    all_rounds: list[dict] = []
    for round_number in range(1, MAX_ROUNDS + 1):
        random_cases: list[QueryCase] = []
        for row in sampled_rows:
            case = build_query_case(row)
            if case:
                random_cases.append(case)
        if not random_cases:
            print("No query cases generated.")
            return 1
        cases = [*random_cases, *bank_cases]
        manual_case_counts = Counter(case.source_file or "unknown_source" for case in random_cases)
        print(f"Round {round_number} manual coverage: {len(manual_case_counts)} manuals")
        print(f"Round {round_number} total queries: random={len(random_cases)} bank={len(bank_cases)}")

        accuracy, results = run_round(round_number, cases)
        reason_counts = Counter(result.reason for result in results)
        in_scope_results = [result for result in results if not result.out_of_scope]
        out_of_scope_results = [result for result in results if result.out_of_scope]
        in_scope_passed = sum(1 for result in in_scope_results if result.ok)
        in_scope_accuracy = in_scope_passed / max(len(in_scope_results), 1)
        print(f"Round {round_number} accuracy(all)={accuracy:.4f}")
        print(
            f"Round {round_number} accuracy(in_scope)={in_scope_accuracy:.4f} "
            f"out_of_scope={len(out_of_scope_results)}"
        )
        print(f"Reasons: {dict(reason_counts)}")

        failed_examples = [
            {
                "reason": result.reason,
                "query": result.query,
                "expected_citation": result.expected_citation,
                "citations": result.citations,
            }
            for result in in_scope_results
            if not result.ok
        ][:30]
        out_of_scope_examples = [
            {
                "reason": result.reason,
                "query": result.query,
                "expected_citation": result.expected_citation,
                "citations": result.citations,
            }
            for result in out_of_scope_results[:20]
        ]

        all_rounds.append(
            {
                "round": round_number,
                "accuracy": accuracy,
                "in_scope_accuracy": in_scope_accuracy,
                "reason_counts": dict(reason_counts),
                "manual_case_counts": dict(manual_case_counts),
                "failed_examples": failed_examples,
                "out_of_scope_examples": out_of_scope_examples,
            }
        )
        REPORT_PATH.write_text(
            json.dumps(
                {
                    "base_url": BASE_URL,
                    "query_count": QUERY_COUNT,
                    "manual_count": len(manuals),
                    "manuals": manuals,
                    "bank_query_count": len(bank_cases),
                    "target_accuracy": TARGET_ACCURACY,
                    "rounds": all_rounds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if in_scope_accuracy >= TARGET_ACCURACY:
            print(f"PASS: in-scope accuracy {in_scope_accuracy:.4f} >= {TARGET_ACCURACY:.4f}")
            return 0

    print(f"FAIL: in-scope accuracy stayed below {TARGET_ACCURACY:.4f} after {MAX_ROUNDS} rounds")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
