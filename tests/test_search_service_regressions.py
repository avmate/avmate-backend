from __future__ import annotations

import unittest

from app.schemas import ReferenceItem
from app.services.search_service import SearchService


def _reference(
    citation: str,
    *,
    text: str,
    score: float = 0.6,
    regulation_type: str = "AIP",
) -> ReferenceItem:
    return ReferenceItem(
        section_id=f"section-{abs(hash(citation)) % 100000}",
        regulation_id=citation,
        citation=citation,
        title=f"Title for {citation}",
        regulation_type=regulation_type,
        source_file="source.pdf",
        source_url="https://example.invalid/source.pdf",
        text=text,
        part="",
        page_ref="",
        table_ref="",
        section_index=0,
        chunk_index=0,
        score=score,
    )


def _section(
    citation: str,
    *,
    text: str,
    title: str,
    page_ref: str = "ENR 1.5 - 39",
    regulation_type: str = "AIP",
) -> dict:
    return {
        "section_id": f"row-{abs(hash((citation, title))) % 100000}",
        "regulation_id": citation,
        "citation": citation,
        "part": "",
        "section_label": "",
        "title": title,
        "text": text,
        "page_ref": page_ref,
        "table_ref": "",
        "source_file": "source.pdf",
        "source_url": "https://example.invalid/source.pdf",
        "regulation_type": regulation_type,
        "section_order": 0,
    }


class _StubCanonicalStore:
    def __init__(self, sections: list[dict]) -> None:
        self._sections = sections

    def search_sections_by_terms(
        self,
        terms: list[str],
        *,
        limit: int = 120,
        regulation_type: str | None = None,
    ) -> list[dict]:
        if regulation_type:
            return [row for row in self._sections if row.get("regulation_type", "").lower() == regulation_type.lower()][:limit]
        return self._sections[:limit]


class SearchServiceRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Tests target ranking/citation internals only; external services are not used.
        self.service = SearchService(embeddings=None, vector_store=None, canonical_store=None)

    def test_ensure_parent_subsection_reference_inserts_5_3_parent(self) -> None:
        references = [
            _reference(
                "AIP ENR 1.7 - 2 subsection 1.4.1",
                text="QNH can be considered accurate only if provided by approved sources.",
                score=0.93,
            ),
            _reference(
                "AIP ENR 1.5 - 38 subsection 5.3.2",
                text="5.3.2 Source details for QNH",
                score=0.88,
            ),
            _reference(
                "AIP ENR 1.5 - 38 subsection 5.3.3",
                text="5.3.3 Additional QNH source conditions",
                score=0.86,
            ),
        ]

        augmented = self.service._ensure_parent_subsection_reference(references, parent_label="5.3", limit=6)
        citations = [item.citation for item in augmented]

        self.assertIn("AIP ENR 1.5 - 38 subsection 5.3", citations)
        self.assertLess(
            citations.index("AIP ENR 1.5 - 38 subsection 5.3"),
            citations.index("AIP ENR 1.5 - 38 subsection 5.3.2"),
        )
        parent_item = next(item for item in augmented if item.citation == "AIP ENR 1.5 - 38 subsection 5.3")
        child_item = next(item for item in augmented if item.citation == "AIP ENR 1.5 - 38 subsection 5.3.2")
        self.assertNotEqual(parent_item.text, child_item.text)
        self.assertTrue(parent_item.text.startswith("5.3"))

    def test_prioritize_weather_minima_references_prefers_subsection_6_2(self) -> None:
        references = [
            _reference(
                "AIP GEN 2.1 - 3 subsection 4.2.1",
                text="Airspace specified as active on public holidays.",
                score=0.97,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.2",
                text="Special alternate weather minima are identified on applicable charts.",
                score=0.70,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="6.2 Special Alternate Weather Minima",
                score=0.64,
            ),
        ]

        ranked = self.service._prioritize_weather_minima_references(references, top_k=3)
        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")

    def test_ensure_special_weather_parent_reference_promotes_page_39_parent(self) -> None:
        references = [
            _reference(
                "AIP ENR 1.5 - 38 subsection 6.2",
                text="6.2 Alternate minima section",
                score=0.91,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.2",
                text="6.2.2 Special alternate weather minima are identified on charts.",
                score=0.89,
            ),
        ]

        normalized = self.service._ensure_special_weather_parent_reference(references, limit=5)
        self.assertEqual(normalized[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")

    def test_ensure_special_weather_parent_reference_prefers_phrase_for_explicit_query(self) -> None:
        references = [
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.3",
                text="6.2.3 Where there is a protracted unserviceability of approach aids.",
                score=0.93,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.1",
                text="6.2.1 Special alternate weather minima are available for specified approaches.",
                score=0.86,
            ),
            _reference(
                "AIP ENR 1.5 - 38 subsection 6.2",
                text="6.2 Special Alternate Weather Minima......................................ENR 1.5 - 39",
                score=0.84,
            ),
        ]

        normalized = self.service._ensure_special_weather_parent_reference(
            references,
            limit=5,
            prefer_special_phrase=True,
        )
        self.assertEqual(normalized[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")
        self.assertIn("Special alternate weather minima", normalized[0].text)

    def test_combine_score_prefers_explicit_subsection_match(self) -> None:
        profile = self.service._build_query_profile("What does ENR 1.5 subsection 6.2 say?")

        match_score, match_passes = self.service._combine_score(
            query_profile=profile,
            document="6.2 Special Alternate Weather Minima apply in specified chart conditions.",
            citation="AIP 6.2",
            regulation_type="AIP",
            semantic_score=0.62,
            requested_citations=[],
            page_ref="ENR 1.5 - 39",
        )
        miss_score, miss_passes = self.service._combine_score(
            query_profile=profile,
            document="6.2 Unrelated section from a different ENR chapter.",
            citation="AIP 6.2",
            regulation_type="AIP",
            semantic_score=0.62,
            requested_citations=[],
            page_ref="ENR 1.10 - 12",
        )

        self.assertTrue(match_passes)
        self.assertFalse(miss_passes)
        self.assertGreater(match_score, miss_score)

    def test_select_best_aip_subsection_honors_explicit_6_2_query(self) -> None:
        profile = self.service._build_query_profile("What does ENR 1.5 subsection 6.2 say?")
        sample_text = """
6.1 Standard Alternate Weather Minima
Standard alternate minima text applies to baseline operations.
6.2 Special Alternate Weather Minima
Special alternate weather minima are identified with a double asterisk.
6.2.2 Special alternate weather minima are identified on applicable instrument approach charts.
6.3 Other Minima Notes
Additional notes unrelated to special alternate minima.
        """.strip()

        label, block = self.service._select_best_aip_subsection(sample_text, "AIP 6.1", profile)
        self.assertEqual(label, "6.2")
        self.assertIn("Special alternate weather minima", block)

    def test_build_query_profile_sets_heading_rollup_for_criteria_queries(self) -> None:
        profile = self.service._build_query_profile(
            "What is the approach criteria for the Special Alternate Weather Minima?"
        )
        self.assertTrue(profile["heading_rollup_intent"])

    def test_build_query_profile_sets_passenger_recency_intent(self) -> None:
        profile = self.service._build_query_profile(
            "What are the specific recency requirements for a PPL to carry passengers?"
        )
        self.assertTrue(profile["passenger_recency_intent"])
        self.assertFalse(profile["heading_rollup_intent"])

    def test_build_query_profile_sets_fuel_requirement_intent(self) -> None:
        profile = self.service._build_query_profile(
            "Under CASR Part 91, what are the fuel requirements for a small aeroplane?"
        )
        self.assertTrue(profile["fuel_requirement_intent"])

    def test_expand_heading_subsection_references_includes_parent_and_children(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "AIP 6.2.1",
                        title="AIP 6.2.1 Special alternate weather minima are available",
                        text="6.2.1 Special alternate weather minima are available for specified approaches with dual ILS/VOR capability.",
                    ),
                    _section(
                        "AIP 6.2.2",
                        title="AIP 6.2.2 Special alternate weather minima are identified on charts",
                        text="6.2.2 Special alternate weather minima are identified on applicable instrument approach charts.",
                    ),
                    _section(
                        "AIP 6.2.3",
                        title="AIP 6.2.3 Where special minima are not available",
                        text="6.2.3 Where there is a protracted unserviceability, special alternate minima may be unavailable.",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("What is the approach criteria for the Special Alternate Weather Minima?")
        seed_refs = [
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="6.2 Special Alternate Weather Minima",
                score=0.94,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.2",
                text="6.2.2 Special alternate weather minima are identified on applicable charts.",
                score=0.9,
            ),
        ]

        expanded = service._expand_heading_subsection_references(seed_refs, profile, top_k=6)
        citations = [item.citation for item in expanded]

        self.assertIn("AIP ENR 1.5 - 39 subsection 6.2", citations)
        self.assertIn("AIP ENR 1.5 - 39 subsection 6.2.1", citations)
        self.assertIn("AIP ENR 1.5 - 39 subsection 6.2.2", citations)
        self.assertIn("AIP ENR 1.5 - 39 subsection 6.2.3", citations)

    def test_select_answer_reference_prefers_operational_child_for_criteria_query(self) -> None:
        profile = self.service._build_query_profile("What is the approach criteria for the Special Alternate Weather Minima?")
        references = [
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="6.2.2 Special alternate weather minima are identified on charts.",
                score=0.95,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.1",
                text="6.2.1 Dual ILS/VOR approach capability must include duplicated LOC and GP.",
                score=0.9,
            ),
        ]

        selected = self.service._select_answer_reference(profile, references)
        self.assertEqual(selected.citation, "AIP ENR 1.5 - 39 subsection 6.2.1")

    def test_plain_english_and_example_for_weather_minima_are_operational(self) -> None:
        query = "What is the approach criteria for the Special Alternate Weather Minima?"
        references = [
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="6.2 Special Alternate Weather Minima",
                score=0.94,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.1",
                text="6.2.1 Special alternate weather minima are available for specified approaches with dual ILS/VOR approach capability.",
                score=0.9,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.2",
                text="6.2.2 Special minima are identified on charts and not available if required MET/ATS services are unavailable.",
                score=0.88,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2.3",
                text="6.2.3 NOTAM advises non-availability or revision during prolonged aid/facility outages.",
                score=0.86,
            ),
        ]
        answer_reference = references[1]

        plain_english = self.service._build_plain_english(query, answer_reference, references)
        example = self.service._build_example(query, answer_reference, references)

        self.assertIn("plain english", plain_english.lower())
        self.assertIn("dual ils/vor", plain_english.lower())
        self.assertIn("standard alternate minima", plain_english.lower())
        self.assertIn("example:", example.lower())
        self.assertIn("pilot", example.lower())
        self.assertIn("notam", example.lower())

    def test_format_readable_text_adds_sentence_spacing_and_preserves_paragraphs(self) -> None:
        raw = "Sentence one.Second sentence.\n\n- bullet    one\n- bullet two"
        formatted = self.service._format_readable_text(raw)
        self.assertIn("Sentence one. Second sentence.", formatted)
        self.assertIn("\n\n- bullet one\n- bullet two", formatted)

    def test_is_precise_citation_rejects_malformed_non_aip_tokens(self) -> None:
        self.assertFalse(self.service._is_precise_citation("CAO level", "CAO"))
        self.assertFalse(self.service._is_precise_citation("CAR do", "CAR"))
        self.assertTrue(self.service._is_precise_citation("CASR Part 61", "CASR"))

    def test_drop_malformed_citations_filters_invalid_items(self) -> None:
        refs = [
            _reference("CAO level", text="Invalid citation text", regulation_type="CAO", score=0.91),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="Valid AIP citation text",
                regulation_type="AIP",
                score=0.9,
            ),
        ]
        filtered = self.service._drop_malformed_citations(refs, limit=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")

    def test_enforce_explicit_page_hints_prefers_matching_page_block(self) -> None:
        refs = [
            _reference(
                "AIP ENR 1.6 - 5 subsection 6.2",
                text="6.2 Radio Failure Procedure",
                score=0.91,
            ),
            _reference(
                "AIP ENR 1.5 - 39 subsection 6.2",
                text="6.2 Special Alternate Weather Minima",
                score=0.88,
            ),
        ]
        filtered = self.service._enforce_explicit_page_hints(refs, ["enr 1.5"], limit=5)

        self.assertTrue(filtered)
        self.assertEqual(filtered[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")

    def test_explicit_subsection_seed_references_prefers_matching_page_hint(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "AIP 6.2",
                        title="AIP 6.2 Radio Failure Procedure",
                        text="6.2 Radio Failure Procedure for communications contingencies.",
                        page_ref="ENR 1.6 - 5",
                        regulation_type="AIP",
                    ),
                    _section(
                        "AIP 6.2",
                        title="AIP 6.2 Special Alternate Weather Minima",
                        text="6.2 Special Alternate Weather Minima with linked approach criteria.",
                        page_ref="ENR 1.5 - 39",
                        regulation_type="AIP",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("What does ENR 1.5 subsection 6.2 say?")
        seeded = service._explicit_subsection_seed_references(profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "AIP ENR 1.5 - 39 subsection 6.2")

    def test_requested_citation_seed_references_match_exact_non_aip_citation(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "CASR 61.215",
                        title="CASR 61.215 Unrelated Part 61 provision",
                        text="61.215 Some unrelated provision.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                    _section(
                        "CASR 61.395",
                        title="CASR 61.395 Passenger recency",
                        text="61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("CASR 61.395 passenger recency")
        seeded = service._requested_citation_seed_references(["casr 61.395"], profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "CASR 61.395")

    def test_requested_citation_seed_references_prefer_operational_text_over_toc_line(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "CASR 61.395",
                        title="CASR 61.395 Passenger recency contents line",
                        text="61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities......................................123",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                    _section(
                        "CASR 61.395",
                        title="CASR 61.395 Passenger recency operative text",
                        text="61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities (1) The holder of a pilot licence is authorised to pilot an aircraft carrying passengers only if the holder meets the recent experience requirements in this regulation.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("CASR 61.395 passenger recency")
        seeded = service._requested_citation_seed_references(["casr 61.395"], profile, top_k=5)

        self.assertTrue(seeded)
        self.assertIn("authorised to pilot an aircraft carrying passengers", seeded[0].text.lower())

    def test_intent_seed_references_for_passenger_recency_prefers_casr_61_395(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "CASR 121.250",
                        title="CASR 121.250 Carriage of restricted persons",
                        text="The aeroplane operator exposition must include procedures for carrying a restricted person.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                    _section(
                        "CASR 61.395",
                        title="CASR 61.395 Passenger recency",
                        text="61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("What are the specific recency requirements for a PPL to carry passengers?")
        seeded = service._intent_seed_references(profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "CASR 61.395")

    def test_intent_seed_references_for_fuel_requirements_prefers_casr_91_455(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "CASR 91.475",
                        title="CASR 91.475 Fuelling aircraft—fire fighting equipment",
                        text="91.475 Fuelling aircraft—fire fighting equipment.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                    _section(
                        "CASR 91.455",
                        title="CASR 91.455 Fuel requirements",
                        text="91.455 Fuel requirements. The Part 91 Manual of Standards may prescribe requirements relating to fuel for aircraft.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile(
            "Under CASR Part 91, what are the fuel requirements for a small aeroplane (fixed-wing)?"
        )
        seeded = service._intent_seed_references(profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "CASR 91.455")

    def test_prioritize_passenger_recency_references_promotes_casr_61_395(self) -> None:
        refs = [
            _reference(
                "CASR 121.250",
                text="121.250 Carriage of restricted persons.",
                regulation_type="CASR",
                score=0.98,
            ),
            _reference(
                "CASR 61.395",
                text="61.395 Limitations on exercise of privileges of pilot licences—recent experience for certain passenger flight activities.",
                regulation_type="CASR",
                score=0.91,
            ),
        ]
        ranked = self.service._prioritize_passenger_recency_references(refs, top_k=5)
        self.assertEqual(ranked[0].citation, "CASR 61.395")

    def test_prioritize_fuel_requirement_references_promotes_casr_91_455_over_rotorcraft(self) -> None:
        profile = self.service._build_query_profile(
            "Under CASR Part 91, what are the fuel requirements for a small aeroplane (fixed-wing)?"
        )
        refs = [
            _reference(
                "CASR 91.430",
                text="91.430 Safety when rotorcraft operating on ground.",
                regulation_type="CASR",
                score=0.99,
            ),
            _reference(
                "CASR 91.455",
                text="91.455 Fuel requirements. The Part 91 Manual of Standards may prescribe requirements relating to fuel for aircraft.",
                regulation_type="CASR",
                score=0.91,
            ),
        ]
        ranked = self.service._prioritize_fuel_requirement_references(refs, profile, top_k=5)
        self.assertEqual(ranked[0].citation, "CASR 91.455")

    def test_intent_seed_references_for_cpl_prefers_part_61_hour_requirements(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "CASR Part 61",
                        title="CASR Part 61 Commercial pilot licence aeronautical experience",
                        text="The applicant for a commercial pilot licence must meet aeronautical experience hours requirements under Part 61.",
                        page_ref="",
                        regulation_type="CASR",
                    ),
                    _section(
                        "CAR 206",
                        title="General operations rule",
                        text="This section discusses non-CPL operational requirements.",
                        page_ref="",
                        regulation_type="CAR",
                    ),
                    _section(
                        "CAO level",
                        title="Malformed CAO heading",
                        text="Not a precise citation and must not be used.",
                        page_ref="",
                        regulation_type="CAO",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("CPL minimum flight hours")
        seeded = service._intent_seed_references(profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "CASR Part 61")
        self.assertTrue(all("CAO level" != item.citation for item in seeded))
        self.assertTrue(all(service._is_precise_citation(item.citation, item.regulation_type) for item in seeded))

    def test_intent_seed_references_for_speed_prefers_250_knots_below_10000(self) -> None:
        service = SearchService(
            embeddings=None,
            vector_store=None,
            canonical_store=_StubCanonicalStore(
                [
                    _section(
                        "AIP 1.7.4",
                        title="AIP 1.7.4 Speed restrictions",
                        text="Below 10 000 FT, indicated airspeed must not exceed 250 knots unless authorised by ATC.",
                        page_ref="ENR 1.1 - 7",
                        regulation_type="AIP",
                    ),
                    _section(
                        "AIP 3.1.2",
                        title="AIP 3.1.2 unrelated heading",
                        text="Unrelated text without the controlling speed limit numbers.",
                        page_ref="ENR 1.10 - 12",
                        regulation_type="AIP",
                    ),
                ]
            ),
        )
        profile = service._build_query_profile("Speed limit below 10,000ft")
        seeded = service._intent_seed_references(profile, top_k=5)

        self.assertTrue(seeded)
        self.assertEqual(seeded[0].citation, "AIP ENR 1.1 - 7 subsection 1.7.4")
        self.assertTrue(any("250 knots" in item.text.lower() for item in seeded))
        self.assertTrue(all(service._is_precise_citation(item.citation, item.regulation_type) for item in seeded))


if __name__ == "__main__":
    unittest.main()
