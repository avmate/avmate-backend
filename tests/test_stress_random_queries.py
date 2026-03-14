from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stress_random_queries import (
    _load_query_bank_cases,
    _is_out_of_scope_case,
    citation_matches_expected,
    citation_family_matches_regulation_type,
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
