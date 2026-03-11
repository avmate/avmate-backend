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


if __name__ == "__main__":
    unittest.main()
