from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from app.services.search_service import SearchService



def _section(
    section_id: str,
    citation: str,
    *,
    text: str,
    title: str | None = None,
    regulation_type: str = "AIP",
) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "regulation_id": citation,
        "citation": citation,
        "part": "",
        "section_label": "",
        "title": title or f"Title for {citation}",
        "text": text,
        "page_ref": "",
        "table_ref": "",
        "source_file": "source.pdf",
        "source_url": "https://example.invalid/source.pdf",
        "regulation_type": regulation_type,
        "section_order": 0,
    }


class _FakeEmbeddings:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeVectorStore:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def query(self, embeddings: list[list[float]], fetch_k: int, where: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append({"embeddings": embeddings, "fetch_k": fetch_k, "where": where})
        if self._responses:
            return self._responses.pop(0)
        return {"metadatas": [[]], "distances": [[]]}


class _FakeCanonicalStore:
    def __init__(
        self,
        *,
        sections: list[dict[str, Any]],
        exact_prefix_map: dict[str, list[str]] | None = None,
        bm25_results: list[tuple[str, float]] | None = None,
    ) -> None:
        self._sections_by_id = {section["section_id"]: section for section in sections}
        self._exact_prefix_map = exact_prefix_map or {}
        self._bm25_results = bm25_results or []
        self.bm25_calls: list[dict[str, Any]] = []

    def get_sections_by_citation_prefix(self, prefix: str, *, limit: int = 20) -> list[dict[str, Any]]:
        section_ids = self._exact_prefix_map.get(prefix, [])[:limit]
        return [self._sections_by_id[section_id] for section_id in section_ids]

    def search_sections_bm25(
        self,
        query: str,
        *,
        limit: int = 40,
        regulation_type: str | None = None,
    ) -> list[tuple[str, float]]:
        self.bm25_calls.append({"query": query, "limit": limit, "regulation_type": regulation_type})
        return self._bm25_results[:limit]

    def get_sections_by_ids(self, section_ids: list[str]) -> list[dict[str, Any]]:
        return [self._sections_by_id[section_id] for section_id in section_ids if section_id in self._sections_by_id]


@dataclass
class _FakeLLM:
    interpreted: dict[str, Any] | None = None

    def interpret_query(self, query: str) -> dict[str, Any] | None:
        return self.interpreted


class SearchServiceRegressionTests(unittest.TestCase):
    def test_search_prefers_exact_citation_match(self) -> None:
        exact = _section(
            "sec-exact",
            "CASR 61.395",
            text="Recent experience requirements for carrying passengers.",
            regulation_type="CASR",
        )
        unrelated = _section(
            "sec-other",
            "CASR 61.400",
            text="Flight review requirements.",
            regulation_type="CASR",
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=_FakeVectorStore(
                [{"metadatas": [[{"section_id": "sec-other"}]], "distances": [[0.01]]}]
            ),
            canonical_store=_FakeCanonicalStore(
                sections=[exact, unrelated],
                exact_prefix_map={"CASR 61.395": ["sec-exact"]},
            ),
        )

        response = service.search("CASR 61.395 passenger recency", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 61.395")
        self.assertEqual(response.references[0].score, 1.0)
        self.assertIn("CASR 61.395", response.citations)

    def test_search_uses_bm25_when_semantic_results_are_empty(self) -> None:
        target = _section(
            "sec-bm25",
            "CASR 61.395",
            text="Passenger recency requirements for pilot licence holders.",
            regulation_type="CASR",
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=_FakeVectorStore([{"metadatas": [[]], "distances": [[]]}]),
            canonical_store=_FakeCanonicalStore(
                sections=[target],
                bm25_results=[("sec-bm25", 0.72)],
            ),
        )

        response = service.search("passenger recency requirements", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 61.395")
        self.assertGreaterEqual(response.references[0].score, 0.7)

    def test_search_filters_aip_results_to_explicit_chapter(self) -> None:
        wrong_chapter = _section(
            "sec-enr14",
            "AIP ENR 1.4 6.2.1",
            text="Unrelated ENR 1.4 content.",
            regulation_type="AIP",
        )
        right_chapter = _section(
            "sec-enr15",
            "AIP ENR 1.5 6.2.1",
            text="Special alternate weather minima are available for specified approaches.",
            regulation_type="AIP",
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=_FakeVectorStore(
                [
                    {
                        "metadatas": [[{"section_id": "sec-enr14"}, {"section_id": "sec-enr15"}]],
                        "distances": [[0.01, 0.2]],
                    }
                ]
            ),
            canonical_store=_FakeCanonicalStore(sections=[wrong_chapter, right_chapter]),
        )

        response = service.search("What does ENR 1.5 subsection 6.2 say?", top_k=3)

        self.assertTrue(response.references)
        self.assertEqual(response.references[0].citation, "AIP ENR 1.5 6.2.1")
        self.assertTrue(all(ref.citation.startswith("AIP ENR 1.5") for ref in response.references))

    def test_search_reranks_candidates_using_explicit_citation_prefix(self) -> None:
        broad = _section(
            "sec-broad",
            "CASR 61.390",
            text="Another Part 61 rule.",
            regulation_type="CASR",
        )
        direct = _section(
            "sec-direct",
            "CASR 61.395",
            text="Passenger recency requirements.",
            regulation_type="CASR",
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=_FakeVectorStore(
                [
                    {
                        "metadatas": [[{"section_id": "sec-broad"}, {"section_id": "sec-direct"}]],
                        "distances": [[0.10, 0.12]],
                    }
                ]
            ),
            canonical_store=_FakeCanonicalStore(sections=[broad, direct]),
        )

        response = service.search("CASR 61.395 passenger recency", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 61.395")

    def test_search_uses_llm_regulation_hint_for_vector_and_bm25_queries(self) -> None:
        aip_section = _section(
            "sec-raim",
            "AIP ENR 1.1 4.8.1",
            text="If GNSS integrity is not assured due to loss of RAIM, the following procedures apply.",
            regulation_type="AIP",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-raim"}]], "distances": [[0.2]]}]
        )
        canonical_store = _FakeCanonicalStore(
            sections=[aip_section],
            bm25_results=[("sec-raim", 0.65)],
        )
        llm = _FakeLLM(
            interpreted={
                "intent": "raim",
                "regulation_type": "AIP",
                "rewritten_query": "RAIM GNSS integrity procedures",
                "keywords": ["raim", "gnss"],
            }
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=canonical_store,
            llm_answer_service=llm,
            enable_llm_query_assist=True,
        )

        response = service.search("What is RAIM and when must it be checked?", top_k=3)

        self.assertEqual(response.references[0].citation, "AIP ENR 1.1 4.8.1")
        self.assertEqual(vector_store.calls[0]["where"], {"regulation_type": {"$eq": "AIP"}})
        self.assertEqual(canonical_store.bm25_calls[0]["regulation_type"], "AIP")

    def test_search_routes_low_flying_query_to_casr_91_267(self) -> None:
        low_flying = _section(
            "sec-low-flying",
            "CASR 91.267",
            text="The aircraft must not be flown below 500 ft above the highest feature or obstacle within 300 m.",
            regulation_type="CASR",
        )
        unrelated = _section(
            "sec-mos",
            "MOS 2.4.1",
            text="Low-level competency standard for training operations.",
            regulation_type="MOS",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-mos"}]], "distances": [[0.02]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[low_flying, unrelated],
                exact_prefix_map={"CASR 91.267": ["sec-low-flying"]},
            ),
        )

        response = service.search("minimum safe altitude low flying below 500 feet AGL", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 91.267")
        self.assertEqual(vector_store.calls[0]["where"], {"regulation_type": {"$eq": "CASR"}})

    def test_search_routes_class_g_vmc_query_to_mos_2_07(self) -> None:
        vmc = _section(
            "sec-vmc",
            "MOS 2.07",
            text="In Class G at or below 5,000 ft AMSL, aircraft must be clear of cloud and in sight of ground or water.",
            regulation_type="MOS",
        )
        rule = _section(
            "sec-casr-vmc",
            "CASR 91.280",
            text="A VFR flight must comply with the VMC criteria for the airspace in which it is conducted.",
            regulation_type="CASR",
        )
        unrelated = _section(
            "sec-aip",
            "AIP GEN 3.4 6.11",
            text="General phraseology content unrelated to VMC minima.",
            regulation_type="AIP",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-aip"}]], "distances": [[0.01]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[vmc, rule, unrelated],
                exact_prefix_map={
                    "MOS 2.07": ["sec-vmc"],
                    "CASR 91.280": ["sec-casr-vmc"],
                },
            ),
        )

        response = service.search("VFR visibility and cloud clearance requirements in class G airspace", top_k=3)

        self.assertEqual(response.references[0].citation, "MOS 2.07")
        self.assertEqual(vector_store.calls[0]["where"], {"regulation_type": {"$eq": "MOS"}})


if __name__ == "__main__":
    unittest.main()
