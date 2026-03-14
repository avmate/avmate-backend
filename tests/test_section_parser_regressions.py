from __future__ import annotations

import unittest

from app.services.section_parser import chunk_chars, derive_precise_citation, extract_citations, split_into_sections


class SectionParserRegressionTests(unittest.TestCase):
    def test_extract_citations_maps_enr_subsection_query_to_aip_subsection(self) -> None:
        query = "What does ENR 1.5 subsection 6.2 say?"
        citations = extract_citations(query)
        self.assertIn("AIP ENR 1.5 6.2", citations)

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

    def test_split_into_sections_prefers_operative_legislation_text_over_toc_duplicates(self) -> None:
        text = """
Civil Aviation Safety Regulations 1998
Compilation No. 100
Part 61 Flight crew licensing
Table of contents
61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities......................................123
61.565 Aeronautical experience requirements for grant of private pilot licences—airship category........................................................................141
61.570 Privileges of commercial pilot licences....................................................142
61.590 Aeronautical experience requirements for grant of commercial pilot licences—aeroplane category...................................................................144
61.610 Aeronautical experience requirements for grant of commercial pilot licences—aeroplane category...................................................................146

Part 61 Flight crew licensing
Subpart 61.E Pilot licences
61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities
61.400 Limitations on exercise of privileges of pilot licences—flight review

61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities
(1) The holder of a pilot licence is authorised to pilot, during take-off or landing, an aircraft of a particular category carrying a passenger by day only if the holder has, within the previous 90 days, conducted at least 3 take-offs and 3 landings.
(2) The holder of a pilot licence is authorised to pilot, during take-off or landing, an aircraft of a particular category carrying a passenger at night only if the holder has, within the previous 90 days, conducted at night at least 3 take-offs and 3 landings.

61.400 Limitations on exercise of privileges of pilot licences—flight review
(1) The holder of a pilot licence is authorised to exercise the privileges of the licence only if the holder has completed a flight review for the licence within the previous 24 months.

Flight crew licensing
Part 61 Private pilot licences
Subpart 61.H Aeronautical experience requirements for private pilot licences—applicants who have not completed integrated training courses
Division 61.H.3
Regulation 61.565
61.565 Aeronautical experience requirements for grant of private pilot licences—airship category
(1) An applicant for a private pilot licence with the airship category rating must have completed at least 25 hours of flight time as pilot of an airship.
(2) The cross-country flight time required by paragraph (1) must include a flight of at least 25 nautical miles.
Civil Aviation Safety Regulations 1998
Compilation No. 100
registered 29/10/2024

Part 61 Flight crew licensing
Subpart 61.I—Commercial pilot licences
Division 61.I.2
61.580 Requirements for grant of commercial pilot licences—general
(1) An applicant for a commercial pilot licence must be at least 18 years of age.

Part 61 Flight crew licensing
Subpart 61.I—Commercial pilot licences
Division 61.I.2
61.590 Aeronautical experience requirements for grant of commercial pilot licences—aeroplane category
(1) An applicant for a commercial pilot licence with the aeroplane category rating must have at least 150 hours of aeronautical experience.

Part 61 Flight crew licensing
Subpart 61.I—Commercial pilot licences
Division 61.I.3
61.610 Aeronautical experience requirements for grant of commercial pilot licences—aeroplane category
(1) An applicant for a commercial pilot licence with the aeroplane category rating must have at least 200 hours of aeronautical experience.
        """.strip()

        sections = split_into_sections(text, regulation_type="CASR")
        by_citation = {section["citation"]: section for section in sections}

        self.assertIn("CASR 61.395", by_citation)
        self.assertIn("CASR 61.565", by_citation)
        self.assertIn("CASR 61.590", by_citation)
        self.assertIn("CASR 61.610", by_citation)
        self.assertIn("(1) The holder of a pilot licence is authorised", by_citation["CASR 61.395"]["text"])
        self.assertIn("25 hours of flight time", by_citation["CASR 61.565"]["text"])
        self.assertIn("150 hours of aeronautical experience", by_citation["CASR 61.590"]["text"])
        self.assertIn("200 hours of aeronautical experience", by_citation["CASR 61.610"]["text"])
        self.assertEqual(sum(1 for section in sections if section["citation"] == "CASR 61.395"), 1)
        self.assertEqual(sum(1 for section in sections if section["citation"] == "CASR 61.565"), 1)

    def test_split_into_sections_skips_aip_toc_entries_and_keeps_real_section(self) -> None:
        text = """
1.18 Speed Restrictions.....ENR 1.5-18

1.18 Speed Restrictions
Pilots must comply with the published airspace speed limitations. This section contains the operative
rule text and enough prose to be indexed correctly instead of being mistaken for a table-of-contents
pointer.
        """.strip()

        sections = split_into_sections(text, regulation_type="AIP")

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["citation"], "AIP ENR 1.5 1.18")
        self.assertNotIn(".....", sections[0]["text"])
        self.assertIn("operative", sections[0]["text"])
        self.assertIn("rule text", sections[0]["text"])

    def test_split_into_sections_parses_aip_top_level_numbered_headings(self) -> None:
        text = """
AIP Australia
GEN 2.2 - 1
GEN 2.2 DEFINITIONS AND ABBREVIATIONS
1. DEFINITIONS
Aerodrome: An area of land or water used for the arrival, departure and movement of aircraft, including
associated buildings, installations and equipment, where the area is intended for use wholly or partly
for the arrival, departure and movement of aircraft.
Aerodrome Beacon: An aeronautical beacon used to indicate the location of an aerodrome from the air.
Aerodrome Control Service: Air traffic control service for aerodrome traffic and aircraft in the circuit.

2. GENERAL AND METEOROLOGICAL ABBREVIATIONS
ABN Aerodrome beacon.
ACC Area control centre.
ATIS Automatic terminal information service.
AWIS Aerodrome weather information service.
The abbreviation list continues with standard operational terms used in weather reporting, navigation
and air traffic service phraseology so the block is long enough to be treated as real content.
        """.strip()

        sections = split_into_sections(text, regulation_type="AIP")
        citations = [section["citation"] for section in sections]

        self.assertIn("AIP GEN 2.2 1", citations)
        self.assertIn("AIP GEN 2.2 2", citations)

    def test_split_into_sections_synthesizes_gen22_definition_entries(self) -> None:
        text = """
AIP Australia
GEN 2.2 - 1
GEN 2.2 DEFINITIONS AND ABBREVIATIONS
1. DEFINITIONS
Aerodrome: An area of land or water used for the arrival, departure and movement of aircraft, including
associated buildings, installations and equipment, where the area is intended for use wholly or partly
for the arrival, departure and movement of aircraft.
Aerodrome Beacon: An aeronautical beacon used to indicate the location of an aerodrome from the air.
Aerodrome Control Service: Air traffic control service for aerodrome traffic and aircraft in the circuit.
Aerodrome Elevation: The elevation of the highest point of the landing area.
Aerodrome Reference Point: The designated geographical location of an aerodrome.
125
GEN 2.2 - 2 27 NOV 2025 AIP Australia
Aerodrome Traffic: All traffic on the manoeuvring area of an aerodrome and all aircraft flying in,
entering, or leaving the traffic circuit.

2. GENERAL AND METEOROLOGICAL ABBREVIATIONS
ABN Aerodrome beacon.
ACC Area control centre.
        """.strip()

        sections = split_into_sections(text, regulation_type="AIP")
        titles = [section["title"] for section in sections]
        citations = [section["citation"] for section in sections]

        self.assertIn("AIP GEN 2.2 1 Aerodrome", titles)
        self.assertIn("AIP GEN 2.2 1 Aerodrome Beacon", titles)
        self.assertIn("AIP GEN 2.2 1 Aerodrome Traffic", titles)
        self.assertNotIn("AIP GEN 2.2 125", citations)
        self.assertNotIn("AIP GEN 2.2 1 Note", titles)
        self.assertFalse(any(title.startswith("AIP GEN 2.2 1 Note ") for title in titles))

    def test_split_into_sections_parses_part61_mos_schedule_units(self) -> None:
        text = """
Schedule 2 Part 61 Manual of Standards
Competency standards

A3 Control aeroplane in normal flight
1 Unit description
This unit describes the skills and knowledge required to control an aeroplane while performing normal flight manoeuvres.
2 Elements and performance criteria
2.1 A3.1 - Climb aeroplane
(a) operate and monitor all aircraft systems when commencing a climbing flight manoeuvre.

A4 Conduct approach and landing
1 Unit description
This unit describes the skills and knowledge required to conduct an approach and landing in an aeroplane.
2 Elements and performance criteria
2.1 A4.1 - Conduct a normal approach
(a) configure the aeroplane and maintain the required flight path.
        """.strip()

        sections = split_into_sections(text, regulation_type="MOS")
        citations = [section["citation"] for section in sections]

        self.assertIn("MOS Schedule 2 A3", citations)
        self.assertIn("MOS Schedule 2 A4", citations)

    def test_split_into_sections_parses_part61_mos_schedule_appendices(self) -> None:
        text = """
Schedule 4 Part 61 Manual of Standards
Aeronautical examinations

APPENDIX 1.5 FLIGHT ENGINEER LICENCE
Flight Engineer Licence Examination
Examination Subject Pass Standard % Hours
FENG Flight Engineer 70 2.0
This appendix sets out the examination code, the subject and the pass standard for the flight engineer licence examination.

APPENDIX 1.6 INSTRUMENT RATING
Instrument Rating Examination
The examination covers privileges, limitations and operational requirements.
It also sets out the knowledge areas, pass standard and time limit for the instrument rating examination.
        """.strip()

        sections = split_into_sections(text, regulation_type="MOS")
        citations = [section["citation"] for section in sections]

        self.assertIn("MOS Schedule 4 FENG", citations)
        self.assertIn("MOS Schedule 4 Appendix 1.6", citations)

    def test_chunk_chars_prefers_legal_boundaries_before_hard_split(self) -> None:
        text = (
            "6.2 Special Alternate Weather Minima\n\n"
            "(a) The aircraft must meet the approach capability criteria;\n"
            "(b) The crew must be authorised for the procedure;\n"
            "(c) The aerodrome entry must identify the special alternate minima."
        )

        chunks = chunk_chars(text, chunk_size=110, overlap=20)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("6.2 Special Alternate Weather Minima"))
        self.assertIn("(a) The aircraft must meet the approach capability criteria", " ".join(chunks))
        self.assertIn("(b) The crew must be authorised for the procedure", " ".join(chunks))
        self.assertIn("(c) The aerodrome entry must identify the special alternate minima.", " ".join(chunks))
        self.assertTrue(all("criteria" in chunk or "(a) The aircraft" not in chunk for chunk in chunks))
        self.assertTrue(all("procedure" in chunk or "(b) The crew" not in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
