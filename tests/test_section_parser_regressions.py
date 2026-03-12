from __future__ import annotations

import unittest

from app.services.section_parser import derive_precise_citation, extract_citations, split_into_sections


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

    def test_derive_precise_citation_uses_regulation_heading_for_non_aip(self) -> None:
        section_text = """
61.235 Aeronautical experience requirements for commercial pilot licence
To be granted a commercial pilot licence, an applicant must have completed
the aeronautical experience hours required for this regulation.
        """.strip()
        self.assertEqual(derive_precise_citation(section_text, "CASR 1998"), "CASR 61.235")

    def test_split_into_sections_extracts_structured_legislation_citations(self) -> None:
        text = """
Civil Aviation Safety Regulations 1998
Compilation No. 100
Part 61 Flight crew licensing
Table of contents
61.005 Applicability
61.010 Definitions for Part 61
61.015 General requirements
61.020 Grant of licence
61.025 Record keeping

Part 61 Flight crew licensing
61.005 Applicability
This Part applies to flight crew licensing and establishes baseline requirements for applicants.
Additional explanatory text continues so this section is long enough to be indexed correctly.

61.010 Definitions for Part 61
Definitions in this section explain terms used in Part 61 and how those terms apply in operation.
Further explanatory wording is included to make this a realistic regulation block.

61.015 General requirements
An applicant for a flight crew licence must satisfy eligibility criteria, knowledge standards and
training requirements stated in this regulation before grant.

61.020 Grant of licence
CASA may grant a licence when the applicant demonstrates competency, satisfies aeronautical
experience requirements and provides all required evidence.

61.025 Record keeping
The holder must keep records that demonstrate continuing compliance and produce those records
when requested by CASA under this Part.
        """.strip()

        sections = split_into_sections(text, regulation_type="CASR")
        citations = [section["citation"] for section in sections]

        self.assertIn("CASR 61.005", citations)
        self.assertIn("CASR 61.010", citations)
        self.assertIn("CASR 61.020", citations)
        self.assertIn("CASR 61.025", citations)


if __name__ == "__main__":
    unittest.main()
