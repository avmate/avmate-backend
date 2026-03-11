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


if __name__ == "__main__":
    unittest.main()
