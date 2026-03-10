from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import requests


BASE_URL = os.getenv("AVMATE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = int(os.getenv("AVMATE_REGRESSION_TIMEOUT", "90"))
READY_WAIT_SECONDS = int(os.getenv("AVMATE_READY_WAIT_SECONDS", "120"))


@dataclass(frozen=True)
class ExpectedQuery:
    query: str
    expected_citation: str
    expected_phrase: str


TEST_CASES = [
    ExpectedQuery(
        query="what is the circling radius for a cat C aircraft",
        expected_citation="AIP ENR 1.5 - 4 subsection 1.6.5. Table 1.1",
        expected_phrase="4.11NM",
    ),
    ExpectedQuery(
        query="How can a QNH be considered accurate on receipt?",
        expected_citation="AIP ENR 1.7 - 2 subsection 1.4.1",
        expected_phrase="QNH can be considered accurate",
    ),
    ExpectedQuery(
        query="What is the special alternate weather minima?",
        expected_citation="AIP ENR 1.5 - 39 subsection 6.2",
        expected_phrase="Special Alternate Weather Minima",
    ),
]


def run_case(case: ExpectedQuery) -> tuple[bool, str]:
    response = requests.post(
        f"{BASE_URL}/search",
        json={"query": case.query, "top_k": 5},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        return False, f"status={response.status_code} body={response.text[:240]}"

    payload = response.json()
    citations = payload.get("citations") or []
    answer = payload.get("answer") or ""
    references = payload.get("references") or []
    top_text = (references[0].get("text") if references else "") or ""

    has_citation = case.expected_citation in citations
    has_phrase = case.expected_phrase.lower() in answer.lower() or case.expected_phrase.lower() in top_text.lower()

    if not has_citation:
        return False, f"expected citation '{case.expected_citation}' not found in {citations}"
    if not has_phrase:
        return False, f"expected phrase '{case.expected_phrase}' missing from answer/reference"
    return True, f"citation ok: {case.expected_citation}"


def wait_until_ready() -> tuple[bool, str]:
    deadline = time.time() + READY_WAIT_SECONDS
    last_detail = "ready endpoint not reached yet"
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/ready", timeout=15)
            payload = response.json()
            if response.status_code == 200 and payload.get("ready"):
                return True, "ready"
            last_detail = payload.get("reason", response.text[:160])
        except Exception as exc:  # noqa: BLE001
            last_detail = str(exc)
        time.sleep(2)
    return False, last_detail


def main() -> int:
    print(f"Regression Target: {BASE_URL}")
    print("=" * 56)
    ready, detail = wait_until_ready()
    if not ready:
        print(f"[FAIL] API not ready before checks: {detail}")
        return 1
    failed = 0
    for case in TEST_CASES:
        ok, detail = run_case(case)
        label = "[PASS]" if ok else "[FAIL]"
        print(f"{label} {case.query}\n  -> {detail}")
        if not ok:
            failed += 1
    print("=" * 56)
    if failed:
        print(f"Result: {len(TEST_CASES) - failed} passed, {failed} failed")
        return 1
    print(f"Result: {len(TEST_CASES)} passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
