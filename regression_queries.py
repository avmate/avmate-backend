from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import requests


BASE_URL = os.getenv("AVMATE_BASE_URL", "https://avmate-backend-production.up.railway.app").rstrip("/")
TIMEOUT_SECONDS = int(os.getenv("AVMATE_REGRESSION_TIMEOUT", "90"))
READY_WAIT_SECONDS = int(os.getenv("AVMATE_READY_WAIT_SECONDS", "120"))
CASE_RETRY_ATTEMPTS = int(os.getenv("AVMATE_CASE_RETRY_ATTEMPTS", "4"))
CASE_RETRY_DELAY_SECONDS = int(os.getenv("AVMATE_CASE_RETRY_DELAY_SECONDS", "8"))
RETRYABLE_503_MARKERS = (
    "index build in progress",
    "indexing_in_progress",
    "embedding model is warming up",
    "embedding_warmup",
)


@dataclass(frozen=True)
class ExpectedQuery:
    query: str
    expected_citation: str
    expected_phrase: str
    expected_additional_citations: tuple[str, ...] = ()
    top_k: int = 5


TEST_CASES = [
    ExpectedQuery(
        query="what is the circling radius for a cat C aircraft",
        expected_citation="AIP ENR 1.5 1.6.5",
        expected_phrase="4.11NM",
    ),
    ExpectedQuery(
        query="How can a QNH be considered accurate on receipt?",
        expected_citation="AIP ENR 1.7 1.4.1",
        expected_phrase="QNH can be considered accurate",
        expected_additional_citations=("AIP ENR 1.5 5.3",),
    ),
    ExpectedQuery(
        query="What is the special alternate weather minima?",
        expected_citation="AIP ENR 1.5 6.2",
        expected_phrase="Special Alternate Weather Minima",
    ),
    ExpectedQuery(
        query="What is the circling radius for category C aircraft?",
        expected_citation="AIP ENR 1.5 1.6.5",
        expected_phrase="4.11NM",
    ),
    ExpectedQuery(
        query="When are special alternate weather minima not available?",
        expected_citation="AIP ENR 1.5 6.2",
        expected_phrase="Special alternate weather minima",
    ),
    ExpectedQuery(
        query="What does ENR 1.5 subsection 6.2 say?",
        expected_citation="AIP ENR 1.5 6.2",
        expected_phrase="Special alternate weather minima",
    ),
    # --- CASR exact-rule queries ---
    ExpectedQuery(
        query="What are the minimum flight hours for a commercial pilot licence?",
        expected_citation="CASR 61",
        expected_phrase="hour",
    ),
    ExpectedQuery(
        query="What are the recent experience requirements for night VFR?",
        expected_citation="CASR 61",
        expected_phrase="night",
    ),
    ExpectedQuery(
        query="minimum fuel reserve for a VFR day flight CASR Part 91",
        expected_citation="CASR 91",
        expected_phrase="fuel",
    ),
    ExpectedQuery(
        query="What is the minimum safe altitude for low flying?",
        expected_citation="CASR 91",
        expected_phrase="500",
    ),
    # --- AIP table-driven operational queries ---
    ExpectedQuery(
        query="What is the transition altitude in Australia?",
        expected_citation="AIP ENR 1.7",
        expected_phrase="transition",
    ),
    ExpectedQuery(
        query="VFR visibility requirements by day in class G airspace",
        expected_citation="AIP ENR 1.2",
        expected_phrase="visibility",
    ),
    ExpectedQuery(
        query="CTAF procedures non-towered aerodrome radio calls",
        expected_citation="AIP",
        expected_phrase="CTAF",
    ),
    ExpectedQuery(
        query="What does AIP ENR 1.5 section 10.2 cover?",
        expected_citation="AIP ENR 1.5 10.2",
        expected_phrase="speed",
    ),
]


def run_case(case: ExpectedQuery) -> tuple[bool, str]:
    attempts = max(1, CASE_RETRY_ATTEMPTS + 1)
    response = None
    for attempt in range(1, attempts + 1):
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": case.query, "top_k": case.top_k},
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            break
        body_lower = response.text.lower()
        retryable_503 = response.status_code == 503 and any(marker in body_lower for marker in RETRYABLE_503_MARKERS)
        retryable_429 = response.status_code == 429
        if attempt < attempts and (retryable_503 or retryable_429):
            time.sleep(CASE_RETRY_DELAY_SECONDS)
            continue
        return False, f"status={response.status_code} body={response.text[:240]}"

    payload = response.json()
    citations = payload.get("citations") or []
    answer = payload.get("answer") or ""
    references = payload.get("references") or []
    all_texts = [answer] + [r.get("text", "") for r in references]

    # Citation check: prefix match — "AIP ENR 1.5 6.2" matches "AIP ENR 1.5 6.2.1" etc.
    exp_lower = case.expected_citation.lower()
    has_citation = any(c.lower().startswith(exp_lower) or c.lower() == exp_lower for c in citations)
    # Phrase check: search across answer + all reference texts
    phrase_lower = case.expected_phrase.lower()
    has_phrase = any(phrase_lower in t.lower() for t in all_texts if t)
    # Additional citations: prefix match too
    missing_additional = [
        citation
        for citation in case.expected_additional_citations
        if not any(c.lower().startswith(citation.lower()) for c in citations)
    ]

    if not has_citation:
        return False, f"expected citation '{case.expected_citation}' (prefix match) not found in {citations}"
    if not has_phrase:
        return False, f"expected phrase '{case.expected_phrase}' missing from answer/references"
    if missing_additional:
        return False, f"missing additional citations: {missing_additional}; got {citations}"
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
