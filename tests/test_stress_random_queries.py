from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from stress_random_queries import (
    QueryCase,
    _load_query_bank_cases,
    _is_out_of_scope_case,
    citation_matches_expected,
    citation_family_matches_regulation_type,
    is_precise_citation,
    normalize_expected_citation,
    query_api,
)


class StressRandomQueriesTests(unittest.TestCase):
    def test_marks_p_factor_as_out_of_scope(self) -> None:
        self.assertTrue(_is_out_of_scope_case("Aerodynamics", "PPL & Student", "What is P-Factor?"))

    def test_marks_scuba_diving_as_out_of_scope(self) -> None:
        self.assertTrue(
            _is_out_of_scope_case(
                "AIP ENR 1.1",
                "Aviation Medicine",
                "How long must a pilot wait after SCUBA diving before flying?",
            )
        )

    def test_keeps_raim_query_in_scope(self) -> None:
        self.assertFalse(
            _is_out_of_scope_case(
                "AIP ENR 1.1",
                "AIP",
                "What is RAIM and when must it be checked for an IFR flight?",
            )
        )

    def test_marks_drone_part_101_query_as_out_of_scope(self) -> None:
        self.assertTrue(
            _is_out_of_scope_case(
                "CASR Part 101",
                "Drone / Part 101",
                "Can I fly a drone within 5.5km of a controlled airport (with a tower)?",
            )
        )

    def test_marks_security_id_card_query_as_out_of_scope(self) -> None:
        self.assertTrue(
            _is_out_of_scope_case(
                "Aviation Transport Security Act",
                "Security",
                "When is a Security Identification Card (ASIC) required?",
            )
        )

    def test_marks_theory_only_navigation_query_as_out_of_scope(self) -> None:
        self.assertTrue(
            _is_out_of_scope_case(
                "Navigation",
                "CPL & IFR",
                "Explain the 3:1 descent profile rule.",
            )
        )

    def test_marks_tas_calculation_query_as_out_of_scope(self) -> None:
        self.assertTrue(
            _is_out_of_scope_case(
                "AIP",
                "Navigation",
                "What is 'TAS' and how is it calculated?",
            )
        )

    def test_query_api_retries_empty_citations(self) -> None:
        case = QueryCase(
            query="What is a Maintenance Release and how long is it valid for?",
            expected_citation="",
            regulation_type="CAR",
            source_file="bank",
            is_bank_case=True,
        )

        class _Response:
            def __init__(self, payload: dict[str, object]) -> None:
                self.status_code = 200
                self._payload = payload

            def json(self) -> dict[str, object]:
                return self._payload

        responses = iter(
            [
                _Response({"citations": [], "answer": "No matching regulatory text found for this query."}),
                _Response({"citations": ["CAR 43"], "answer": "CAR 43: Maintenance release"}),
            ]
        )

        with patch("stress_random_queries.requests.post", side_effect=lambda *args, **kwargs: next(responses)):
            with patch("stress_random_queries.RETRY_ATTEMPTS", 2), patch("stress_random_queries.RETRY_DELAY_SECONDS", 0):
                result = query_api(case)

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ["CAR 43"])

    def test_accepts_structured_aip_citations(self) -> None:
        self.assertTrue(is_precise_citation("AIP ENR 1.5 6.2.1", "AIP"))
        self.assertTrue(is_precise_citation("AIP GEN 1.2.2", "AIP"))

    def test_accepts_mos_schedule_citations(self) -> None:
        self.assertTrue(is_precise_citation("MOS Schedule 4 Section 2", "MOS"))

    def test_normalizes_part_citations_for_matching(self) -> None:
        self.assertEqual(normalize_expected_citation("CASR Part 61.215"), "CASR 61.215")
        self.assertTrue(citation_matches_expected("CASR Part 61.215", "CASR 61.215"))

    def test_allows_descendant_match_for_broad_part_expectation(self) -> None:
        self.assertTrue(citation_matches_expected("CASR 26", "CASR 26.001"))
        self.assertTrue(citation_matches_expected("CAR Part 5", "CAR 5.02"))

    def test_query_bank_clears_unsupported_exact_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank.csv"
            path.write_text(
                "Question,Primary Reference\n"
                "\"What does CASR 14.5.2 require?\",CASR 14.5.2\n",
                encoding="utf-8",
            )
            cases = _load_query_bank_cases(str(path), supported_citations={"CASR 61.215"})
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].expected_citation, "")

    def test_rejects_cross_family_seed_citations(self) -> None:
        self.assertFalse(citation_family_matches_regulation_type("CASR 4.2", "Manual"))
        self.assertFalse(citation_family_matches_regulation_type("CASR 1998.", "MOS"))
        self.assertTrue(citation_family_matches_regulation_type("MOS Schedule 4 Section L", "MOS"))


if __name__ == "__main__":
    unittest.main()
