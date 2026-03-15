from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


BASE_URL = os.getenv("AVMATE_BASE_URL", "https://avmate-backend-production.up.railway.app").rstrip("/")
CONCURRENCY = max(1, int(os.getenv("AVMATE_STABILITY_CONCURRENCY", "8")))
REPEATS = max(1, int(os.getenv("AVMATE_STABILITY_REPEATS", "10")))
TIMEOUT = int(os.getenv("AVMATE_REQUEST_TIMEOUT_SECONDS", "90"))
REPORT_PATH = Path(os.getenv("AVMATE_STABILITY_REPORT", "stability_audit_report.json"))

DEFAULT_QUERIES = [
    "What are the alcohol restrictions for flight crew in Australia?",
    "What documents must be carried on board for a local VFR flight?",
    "What are your privileges and limitations under a Part 61 PPL?",
    "How do you determine if an aircraft is Airworthy under Australian regs?",
    "What are the VFR cloud clearance requirements in Class C airspace?",
    "Explain the VFR fuel requirements for a fixed-wing aircraft (Day/Night).",
    "What is a Maintenance Release and how long is it valid for?",
    "Under CASR Part 91, what are the fuel requirements for a small aeroplane (fixed-wing)?",
    "What is the purpose of an Airworthiness Directive (AD)?",
    "Which document contains the foundational general operating rules for all Australian pilots?",
]


def _work_items() -> list[tuple[str, int]]:
    return [(query, run) for query in DEFAULT_QUERIES for run in range(REPEATS)]


def _hit(item: tuple[str, int]) -> dict[str, object]:
    query, run = item
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": query, "top_k": 5},
            timeout=TIMEOUT,
        )
        elapsed = time.perf_counter() - started
        payload = response.json()
        citations = payload.get("citations") or []
        return {
            "query": query,
            "run": run,
            "status": response.status_code,
            "elapsed_seconds": elapsed,
            "citation_count": len(citations),
            "citations": citations[:5],
            "request_id": payload.get("request_id"),
            "answer_preview": (payload.get("answer") or "")[:180].replace("\n", " "),
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return {
            "query": query,
            "run": run,
            "status": "ERR",
            "elapsed_seconds": elapsed,
            "citation_count": 0,
            "citations": [],
            "request_id": None,
            "answer_preview": str(exc),
        }


def main() -> int:
    results: list[dict[str, object]] = []
    items = _work_items()
    print(f"Target URL: {BASE_URL}")
    print(f"Queries: {len(DEFAULT_QUERIES)}")
    print(f"Repeats per query: {REPEATS}")
    print(f"Concurrency: {CONCURRENCY}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for result in executor.map(_hit, items):
            results.append(result)

    summary: dict[str, dict[str, object]] = {}
    overall_empty = 0
    overall_errors = 0
    for query in DEFAULT_QUERIES:
        rows = [row for row in results if row["query"] == query]
        latencies = [float(row["elapsed_seconds"]) for row in rows]
        empty = sum(1 for row in rows if row["status"] == 200 and row["citation_count"] == 0)
        errors = sum(1 for row in rows if row["status"] != 200)
        overall_empty += empty
        overall_errors += errors
        summary[query] = {
            "runs": len(rows),
            "empty_200": empty,
            "non_200_or_errors": errors,
            "p50_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(max(latencies) if len(latencies) < 20 else statistics.quantiles(latencies, n=20)[18], 3),
        }

    report = {
        "base_url": BASE_URL,
        "concurrency": CONCURRENCY,
        "repeats": REPEATS,
        "overall_empty_200": overall_empty,
        "overall_non_200_or_errors": overall_errors,
        "summary": summary,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for query, item in summary.items():
        print(
            f"{query}\n"
            f"  empty_200={item['empty_200']} errors={item['non_200_or_errors']} "
            f"p50={item['p50_seconds']}s p95={item['p95_seconds']}s"
        )

    print(f"Report written to {REPORT_PATH}")
    if overall_empty or overall_errors:
        print("FAIL: stability issues detected")
        return 1
    print("PASS: no stability issues detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
