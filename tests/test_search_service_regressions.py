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
            text="VMC criteria means meteorological conditions expressed in terms of the flight visibility and distance from cloud (horizontal and vertical).",
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

    def test_search_routes_flight_review_query_to_casr_61_745(self) -> None:
        flight_review = _section(
            "sec-flight-review",
            "CASR 61.745",
            text="Limitations on exercise of privileges of aircraft class ratings—flight review.",
            regulation_type="CASR",
        )
        unrelated = _section(
            "sec-balloon",
            "CASR 101.135",
            text="What to do if tethered balloon escapes from its mooring.",
            regulation_type="CASR",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-balloon"}]], "distances": [[0.02]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[flight_review, unrelated],
                exact_prefix_map={"CASR 61.745": ["sec-flight-review"]},
            ),
        )

        response = service.search("What is a flight review and when is it required?", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 61.745")
        self.assertEqual(vector_store.calls[0]["where"], {"regulation_type": {"$eq": "CASR"}})

    def test_search_routes_instrument_currency_query_to_casr_61_870(self) -> None:
        recency = _section(
            "sec-ifr-recency",
            "CASR 61.870",
            text="Limitations on exercise of privileges of instrument ratings—recent experience: general.",
            regulation_type="CASR",
        )
        customs = _section(
            "sec-customs",
            "AIP GEN 1.3 4.2.5",
            text="All persons entering Australia who are in possession of Australian currency, foreign currency.",
            regulation_type="AIP",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-customs"}]], "distances": [[0.01]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[recency, customs],
                exact_prefix_map={"CASR 61.870": ["sec-ifr-recency"]},
            ),
        )

        response = service.search("What are the instrument currency requirements?", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 61.870")
        self.assertEqual(vector_store.calls[0]["where"], {"regulation_type": {"$eq": "CASR"}})

    def test_search_routes_passenger_briefing_query_to_casr_91_565(self) -> None:
        ga_briefing = _section(
            "sec-ga-briefing",
            "CASR 91.565",
            text="The pilot in command must brief passengers on safety before departure.",
            regulation_type="CASR",
        )
        air_transport = _section(
            "sec-airtransport",
            "CASR 133.235",
            text="Safety briefing cards for rotorcraft operators.",
            regulation_type="CASR",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-airtransport"}]], "distances": [[0.02]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[ga_briefing, air_transport],
                exact_prefix_map={"CASR 91.565": ["sec-ga-briefing"]},
            ),
        )

        response = service.search("What are the requirements for a passenger safety briefing?", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 91.565")

    def test_search_routes_mel_query_to_casr_91_925(self) -> None:
        mel = _section(
            "sec-mel",
            "CASR 91.925",
            text="Definitions: master minimum equipment list or MMEL, for a type of aircraft.",
            regulation_type="CASR",
        )
        unrelated = _section(
            "sec-ratings",
            "CASR 10.3",
            text="CASR Part 61 Rating Comments Command Instrument Rating.",
            regulation_type="CASR",
        )
        vector_store = _FakeVectorStore(
            [{"metadatas": [[{"section_id": "sec-ratings"}]], "distances": [[0.02]]}]
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=vector_store,
            canonical_store=_FakeCanonicalStore(
                sections=[mel, unrelated],
                exact_prefix_map={"CASR 91.925": ["sec-mel"]},
            ),
        )

        response = service.search("Can a pilot fly with an inoperative instrument?", top_k=3)

        self.assertEqual(response.references[0].citation, "CASR 91.925")

    def test_search_deduplicates_citation_list_when_multiple_refs_share_official_citation(self) -> None:
        glossary_term = _section(
            "sec-arp",
            "AIP GEN 2.2 1",
            text="Aerodrome Reference Point (ARP): The designated geographical location of an aerodrome.",
            title="AIP GEN 2.2 1 Aerodrome Reference Point (ARP)",
            regulation_type="AIP",
        )
        glossary_term_2 = _section(
            "sec-aerodrome",
            "AIP GEN 2.2 1",
            text="Aerodrome: An area of land or water used for the arrival, departure and movement of aircraft.",
            title="AIP GEN 2.2 1 Aerodrome",
            regulation_type="AIP",
        )
        service = SearchService(
            embeddings=_FakeEmbeddings(),
            vector_store=_FakeVectorStore(
                [
                    {
                        "metadatas": [[{"section_id": "sec-arp"}, {"section_id": "sec-aerodrome"}]],
                        "distances": [[0.02, 0.04]],
                    }
                ]
            ),
            canonical_store=_FakeCanonicalStore(sections=[glossary_term, glossary_term_2]),
        )

        response = service.search("What is an aerodrome reference point?", top_k=5)

        self.assertEqual(response.references[0].title, "AIP GEN 2.2 1 Aerodrome Reference Point (ARP)")
        self.assertEqual(response.citations, ["AIP GEN 2.2 1"])


if __name__ == "__main__":
    unittest.main()
