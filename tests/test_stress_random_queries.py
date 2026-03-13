from __future__ import annotations

import unittest

from stress_random_queries import _is_out_of_scope_case


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


if __name__ == "__main__":
    unittest.main()
