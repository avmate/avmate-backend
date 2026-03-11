from __future__ import annotations

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

NON_AIP_CITATION_PATTERN = re.compile(
    r"^(?:CASR|CAR|CAO|MOS|CAA)\s+(?:Part\s+)?[0-9]+[0-9A-Za-z.\-]*(?:\([0-9A-Za-z]+\))*$|^CAO\s+DOC\s+[0-9A-Za-z.\-]+$",
    re.IGNORECASE,
)
AIP_OUTPUT_CITATION_PATTERN = re.compile(
    r"^AIP\s+(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\(?\d+\)?\s+subsection\s+[0-9]+(?:\.[0-9]+){1,4}(?:\.\s*Table\s+\d+(?:\.\d+)*)?$",
    re.IGNORECASE,
)
SUBSECTION_PATTERN = re.compile(r"\bAIP\s+([0-9]+(?:\.[0-9]+){1,4})\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_citation: str
    regulation_type: str


@dataclass
class QueryResult:
    ok: bool
    reason: str
    query: str
    expected_citation: str
    citations: list[str]
    status_code: int


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

    return QueryCase(query=random.choice(templates), expected_citation=expected, regulation_type=regulation_type)


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
    return [random.choice(usable) for _ in range(count)]


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
        )

    if len({c.lower() for c in citations}) != len(citations):
        return QueryResult(
            ok=False,
            reason="duplicate_citations",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
        )

    if not all(is_precise_citation(citation, case.regulation_type) for citation in citations):
        return QueryResult(
            ok=False,
            reason="malformed_citation",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
        )

    if case.expected_citation not in citations:
        return QueryResult(
            ok=False,
            reason="expected_missing",
            query=case.query,
            expected_citation=case.expected_citation,
            citations=citations,
            status_code=200,
        )

    return QueryResult(
        ok=True,
        reason="ok",
        query=case.query,
        expected_citation=case.expected_citation,
        citations=citations,
        status_code=200,
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

    sampled_rows = load_random_sections(QUERY_COUNT)
    if not sampled_rows:
        print("No usable indexed sections found for stress generation.")
        return 1

    all_rounds: list[dict] = []
    for round_number in range(1, MAX_ROUNDS + 1):
        cases: list[QueryCase] = []
        for row in sampled_rows:
            case = build_query_case(row)
            if case:
                cases.append(case)
        if not cases:
            print("No query cases generated.")
            return 1

        accuracy, results = run_round(round_number, cases)
        reason_counts = Counter(result.reason for result in results)
        print(f"Round {round_number} accuracy={accuracy:.4f}")
        print(f"Reasons: {dict(reason_counts)}")

        failed_examples = [
            {
                "reason": result.reason,
                "query": result.query,
                "expected_citation": result.expected_citation,
                "citations": result.citations,
            }
            for result in results
            if not result.ok
        ][:30]

        all_rounds.append(
            {
                "round": round_number,
                "accuracy": accuracy,
                "reason_counts": dict(reason_counts),
                "failed_examples": failed_examples,
            }
        )
        REPORT_PATH.write_text(
            json.dumps(
                {
                    "base_url": BASE_URL,
                    "query_count": QUERY_COUNT,
                    "target_accuracy": TARGET_ACCURACY,
                    "rounds": all_rounds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if accuracy >= TARGET_ACCURACY:
            print(f"PASS: accuracy {accuracy:.4f} >= {TARGET_ACCURACY:.4f}")
            return 0

    print(f"FAIL: accuracy stayed below {TARGET_ACCURACY:.4f} after {MAX_ROUNDS} rounds")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

