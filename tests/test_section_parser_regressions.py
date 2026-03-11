from __future__ import annotations

import unittest

from app.services.section_parser import extract_citations


class SectionParserRegressionTests(unittest.TestCase):
    def test_extract_citations_maps_enr_subsection_query_to_aip_subsection(self) -> None:
        query = "What does ENR 1.5 subsection 6.2 say?"
        citations = extract_citations(query)
        self.assertIn("AIP 6.2", citations)

    def test_extract_citations_preserves_standard_aip_citations(self) -> None:
        query = "Explain AIP 1.4.1 and AIP 5.3 requirements."
        citations = extract_citations(query)
        self.assertIn("AIP 1.4.1", citations)
        self.assertIn("AIP 5.3", citations)

    def test_extract_citations_ignores_non_numeric_cao_car_tokens(self) -> None:
        query = "CAO level standards and CAR do not apply."
        citations = extract_citations(query)
        self.assertNotIn("CAO level", citations)
        self.assertNotIn("CAR do", citations)


if __name__ == "__main__":
    unittest.main()
