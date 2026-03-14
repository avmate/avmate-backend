from __future__ import annotations

import unittest

from stress_random_queries import (
    _is_out_of_scope_case,
    citation_matches_expected,
    is_precise_citation,
    normalize_expected_citation,
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

    def test_accepts_structured_aip_citations(self) -> None:
        self.assertTrue(is_precise_citation("AIP ENR 1.5 6.2.1", "AIP"))

    def test_accepts_mos_schedule_citations(self) -> None:
        self.assertTrue(is_precise_citation("MOS Schedule 4 Section 2", "MOS"))

    def test_normalizes_part_citations_for_matching(self) -> None:
        self.assertEqual(normalize_expected_citation("CASR Part 61.215"), "CASR 61.215")
        self.assertTrue(citation_matches_expected("CASR Part 61.215", "CASR 61.215"))

    def test_allows_descendant_match_for_broad_part_expectation(self) -> None:
        self.assertTrue(citation_matches_expected("CASR 26", "CASR 26.001"))
        self.assertTrue(citation_matches_expected("CAR Part 5", "CAR 5.02"))


if __name__ == "__main__":
    unittest.main()
