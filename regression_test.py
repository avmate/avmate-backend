#!/usr/bin/env python3
"""
AvMate Regression Test Suite
=============================
10 query/citation assertions against the live Railway endpoint.

Each test verifies:
  - HTTP 200 (not a 503 or error)
  - At least one returned citation starts with the expected prefix
  - At least one returned text contains an expected keyword

Run after a reindex completes:
    python regression_test.py

Override the base URL:
    AVMATE_BASE_URL=https://... python regression_test.py
"""

import os
import sys
import time

import requests

BASE_URL = os.getenv("AVMATE_BASE_URL", "https://avmate-backend-production.up.railway.app")
TIMEOUT = 30

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

# ---------------------------------------------------------------------------
# Test cases — (description, query, citation_prefix, text_keyword)
#
# citation_prefix: any returned citation must START WITH this string (case-insensitive).
#                  Use "" to skip citation check.
# text_keyword:    any returned text must CONTAIN this word (case-insensitive).
#                  Use "" to skip text check.
# ---------------------------------------------------------------------------
TESTS = [
    (
        "Speed limit below 10,000 ft (250KT rule)",
        "what is the maximum speed below 10000 feet",
        "AIP ENR",
        "250",
    ),
    (
        "AIP ENR 1.5 speed restrictions",
        "speed restrictions ENR 1.5",
        "AIP ENR 1.5",
        "speed",  # 250KT rule is in a table; prose sections reference "speed restrictions"
    ),
    (
        "VFR meteorological minima",
        "VFR visibility and cloud clearance requirements",
        "AIP ENR",
        "visibility",
    ),
    (
        "Transition altitude / altimeter setting",
        "transition altitude altimeter setting Australia",
        "AIP ENR",
        "transition",
    ),
    (
        "CTAF radio call procedures",
        "CTAF radio calls non-towered aerodrome",
        "AIP",   # CTAF is in AIP GEN 3.4 (COM procedures), not ENR
        "CTAF",
    ),
    (
        "Commercial pilot licence flight hours CASR 61",
        "minimum flight hours for commercial pilot licence",
        "CASR 61",
        "hour",
    ),
    (
        "Class 1 medical certificate requirements",
        "Class 1 medical certificate requirements CASR",
        "CASR",
        "medical",
    ),
    (
        "Night VFR currency CASR Part 61",
        "night VFR recent experience currency requirements",
        "CASR",
        "night",
    ),
    (
        "Fuel requirements VFR flight CASR Part 91",
        "fuel reserve requirements VFR day flight",
        "MOS",
        "fuel",
    ),
    (
        "Low-level flight below 500 ft AGL",
        "minimum safe altitude low flying below 500 feet",
        "CASR",
        "500",
    ),
]


def wait_for_ready(max_wait: int = 300) -> bool:
    """Poll /ready until index is up, up to max_wait seconds."""
    print(f"{INFO} Checking /ready ...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/ready", timeout=10)
            body = r.json()
            status = body.get("vector_store_status", "")
            count = body.get("collection_count", 0)
            print(f"  status={status} count={count}")
            if r.status_code == 200 and body.get("ready"):
                return True
            if status == "indexing":
                time.sleep(15)
                continue
        except Exception as e:
            print(f"  /ready error: {e}")
        time.sleep(10)
    return False


def run_test(description: str, query: str, citation_prefix: str, text_keyword: str) -> bool:
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE_URL}/search", json={"query": query, "top_k": 5}, timeout=TIMEOUT)
        except Exception as e:
            print(f"{FAIL} {description} — request error: {e}")
            return False

        if r.status_code == 503:
            # Transient: index warming up between tests — wait and retry
            time.sleep(15)
            continue
        break
    else:
        body = r.text[:200]
        print(f"{FAIL} {description} — HTTP 503 after 3 attempts: {body}")
        return False

    if r.status_code != 200:
        body = r.text[:200]
        print(f"{FAIL} {description} — HTTP {r.status_code}: {body}")
        return False

    data = r.json()
    references = data.get("references", [])
    citations = [ref.get("citation", "") for ref in references]
    texts = [ref.get("text", "") for ref in references]

    citation_ok = True
    text_ok = True

    if citation_prefix:
        citation_ok = any(c.lower().startswith(citation_prefix.lower()) for c in citations)

    if text_keyword:
        text_ok = any(text_keyword.lower() in t.lower() for t in texts)

    if citation_ok and text_ok:
        top_citation = citations[0] if citations else "(none)"
        print(f"{PASS} {description}")
        print(f"       top citation: {top_citation}")
        return True

    print(f"{FAIL} {description}")
    if not citation_ok:
        print(f"       expected citation starting with '{citation_prefix}'")
        print(f"       got: {citations}")
    if not text_ok:
        print(f"       expected text containing '{text_keyword}'")
    return False


def main() -> None:
    print(f"\n{'=' * 60}")
    print("  AvMate Regression Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"{'=' * 60}\n")

    if not wait_for_ready():
        print(f"{FAIL} Service not ready after timeout — aborting.")
        sys.exit(1)

    print(f"\n{INFO} Running {len(TESTS)} regression tests...\n")

    passed = 0
    failed = 0
    for description, query, citation_prefix, text_keyword in TESTS:
        ok = run_test(description, query, citation_prefix, text_keyword)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed / {len(TESTS)} total")
    print(f"{'=' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
