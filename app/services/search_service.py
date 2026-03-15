"""Aviation regulatory search service.

Retrieval flow (per query):
  1. Extract explicit citations from query text
  2. Exact citation lookup → canonical store direct match (score=1.0, bypasses semantic)
  3. LLM interprets query  → regulation_type hint + rewritten query + keywords
  4. Semantic search       → ChromaDB (filtered by regulation_type when known)
  5. Canonical join        → resolve section_ids to full metadata from SQLite
  6. Chapter-aware filter  → prevent cross-chapter drift (e.g. ENR 1.4 for ENR 1.5 queries)
  7. Merge                 → exact hits first, semantic fill after
  8. Lexical fallback      → canonical store keyword search when results are sparse
  9. LLM answer generation → Claude reads actual regulation text, returns structured answer
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable

from app.schemas import ReferenceItem, SearchResponse
from app.services.canonical_store import CanonicalStore
from app.services.embedding_service import EmbeddingService
from app.services.section_parser import extract_citations
from app.services.vector_store import VectorStore

if TYPE_CHECKING:
    from app.services.llm_answer_service import LLMAnswerService


REGULATION_FAMILIES = {"AIP", "CASR", "CAR", "CAO", "MOS", "CAA"}
_FAMILY_RE = re.compile(r"^(AIP|CASR|CAR|CAO|MOS|CAA)\b", re.IGNORECASE)
AVIATION_QUERY_TERMS = {
    "aerodrome",
    "aeroplane",
    "aircraft",
    "airspace",
    "ais",
    "aip",
    "alternate",
    "altimeter",
    "approach",
    "aviation",
    "banner",
    "casa",
    "casr",
    "cao",
    "cavok",
    "ctaf",
    "emergency",
    "enr",
    "equipment",
    "flight",
    "fuel",
    "gen",
    "gnss",
    "ifr",
    "ils",
    "instrument",
    "landing",
    "licence",
    "license",
    "logbook",
    "lsalt",
    "mayday",
    "medical",
    "mel",
    "metar",
    "mos",
    "night vfr",
    "notam",
    "pan-pan",
    "parachute",
    "passenger",
    "pbn",
    "ped",
    "pic",
    "pilot",
    "qnh",
    "qfe",
    "qne",
    "raim",
    "radio",
    "rating",
    "refuelling",
    "runway",
    "rvsm",
    "sartime",
    "seatbelt",
    "speed",
    "ssr",
    "taf",
    "takeoff",
    "take-off",
    "taxi",
    "transponder",
    "transition",
    "altitude",
    "cloud",
    "clearance",
    "journey log",
    "maintenance release",
    "visibility",
    "cruising",
    "vfr",
    "vmc",
    "weather",
}
AVIATION_QUERY_PATTERN = re.compile(
    r"\b("
    r"part\s+\d+[a-z]?"
    r"|ppl|cpl|atpl|rpl"
    r"|aoc|damp|tcas|gpws|tem|sms|nts|kdr"
    r"|multicrew|multi-crew|endorsement|examiner"
    r"|point of no return|pnr|pressure height|density height"
    r"|1-in-60|stopway|balanced field|microburst"
    r")\b",
    re.IGNORECASE,
)


class SearchService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        canonical_store: CanonicalStore,
        llm_answer_service: "LLMAnswerService | None" = None,
        enable_llm_answers: bool = False,
        enable_llm_query_assist: bool = False,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._canonical_store = canonical_store
        self._llm = llm_answer_service
        self._enable_llm_answers = enable_llm_answers
        self._enable_llm_query_assist = enable_llm_query_assist

    def search(self, query: str, top_k: int = 5, request_id: str = "") -> SearchResponse:
        # 1. Extract any explicit citations written in the query (e.g. "CASR 135.175")
        explicit_citations = [_normalize_structured_citation(c) for c in extract_citations(query)]
        explicit_families = {_detect_family(c) for c in explicit_citations} - {None}
        has_specific_explicit_citation = any(_is_specific_citation(citation) for citation in explicit_citations)
        query_route = _route_known_query(query) if not has_specific_explicit_citation else None
        if not query_route and explicit_citations:
            query_route = _route_explicit_citation_query(query, explicit_citations)
        if not explicit_citations and not query_route and not _looks_aviation_query(query):
            return _empty_response(request_id)

        # 2. Exact citation lookup — deterministic, bypasses semantic for structured queries
        #    Only fires for specific section citations, not broad family/part references.
        #    "CASR 61.385" or "AIP ENR 1.5 6.2" → yes; "CASR 91" or "AIP ENR 1.5" → no.
        seen_ids: set[str] = set()
        exact_candidates: list[tuple[float, dict]] = []
        for citation in explicit_citations:
            if not _is_specific_citation(citation):
                continue
            for sec in self._canonical_store.get_sections_by_citation_tree(citation):
                if _is_candidate_family_consistent(sec) and sec["section_id"] not in seen_ids:
                    exact_candidates.append((1.0, sec))
                    seen_ids.add(sec["section_id"])
        if query_route:
            exact_candidates.extend(
                _collect_prefix_candidates(
                    self._canonical_store,
                    query_route["preferred_citations"],
                    seen_ids,
                    query_route.get("search_text", query),
                )
            )

        # 3. LLM interprets query for better semantic retrieval
        regulation_hint: str | None = None
        search_text = query
        keywords: list[str] = []

        if query_route:
            regulation_hint = query_route["regulation_hint"]
            search_text = query_route["search_text"]
        elif self._llm and self._enable_llm_query_assist:
            try:
                interpreted = self._llm.interpret_query(query)
                if interpreted:
                    search_text = interpreted.get("rewritten_query") or query
                    keywords = interpreted.get("keywords") or []
                    raw_hint = interpreted.get("regulation_type")
                    if isinstance(raw_hint, str) and raw_hint.upper() in REGULATION_FAMILIES:
                        regulation_hint = raw_hint.upper()
            except Exception:
                pass

        # Explicit citation family overrides LLM hint
        if explicit_families:
            regulation_hint = next(iter(explicit_families))

        # 4. Semantic retrieval from ChromaDB
        fetch_k = max(top_k * 8, 40)
        embedding = self._embeddings.encode([search_text])[0]

        where_filter: dict[str, Any] | None = (
            {"regulation_type": {"$eq": regulation_hint}} if regulation_hint else None
        )
        raw = self._vector_store.query([embedding], fetch_k, where=where_filter)
        section_ids = _extract_section_ids(raw)
        distances = (raw.get("distances") or [[]])[0]

        # Retry without filter if filtered search returned nothing
        if not section_ids and regulation_hint:
            raw = self._vector_store.query([embedding], fetch_k)
            section_ids = _extract_section_ids(raw)
            distances = (raw.get("distances") or [[]])[0]

        # 5. BM25 retrieval (parallel path — runs alongside semantic, not just fallback)
        bm25_scores: dict[str, float] = {
            sid: score
            for sid, score in self._canonical_store.search_sections_bm25(
                search_text, limit=fetch_k, regulation_type=regulation_hint
            )
            if sid not in seen_ids
        }

        # 6. Resolve all unique section_ids → full metadata
        all_retrieve_ids = list(dict.fromkeys(section_ids + [s for s in bm25_scores if s not in section_ids]))
        sections_by_id = {
            s["section_id"]: s
            for s in self._canonical_store.get_sections_by_ids(all_retrieve_ids)
        }

        # Build merged candidates: score = max(semantic, bm25) per section
        semantic_candidates: list[tuple[float, dict]] = []
        merged_sids: set[str] = set()
        for sid, dist in zip(section_ids, distances):
            if sid in seen_ids or sid in merged_sids:
                continue
            section = sections_by_id.get(sid)
            if not section:
                continue
            if not _is_candidate_family_consistent(section):
                continue
            sem_score = max(0.0, 1.0 - float(dist))
            score = max(sem_score, bm25_scores.get(sid, 0.0))
            semantic_candidates.append((score, section))
            merged_sids.add(sid)

        # Add BM25-only hits (not in semantic results)
        for sid, bm25_score in bm25_scores.items():
            if sid in seen_ids or sid in merged_sids:
                continue
            section = sections_by_id.get(sid)
            if section and _is_candidate_family_consistent(section):
                semantic_candidates.append((bm25_score, section))
                merged_sids.add(sid)

        # 7. Chapter-aware filter — prevents cross-chapter drift
        #    e.g. "ENR 1.5 subsection 6.2" must not pull ENR 1.4 results
        if explicit_citations:
            chapter_prefixes = [
                p for p in (_extract_chapter_prefix(c) for c in explicit_citations) if p
            ]
            if chapter_prefixes:
                chapter_matched = [
                    (s, sec)
                    for s, sec in semantic_candidates
                    if any(
                        sec.get("citation", "").lower().startswith(pfx.lower())
                        for pfx in chapter_prefixes
                    )
                ]
                if chapter_matched:
                    semantic_candidates = chapter_matched

            broad_family_prefixes = [
                prefix
                for prefix in (_extract_broad_prefix(c) for c in explicit_citations if not _is_specific_citation(c))
                if prefix
            ]
            if broad_family_prefixes:
                broad_matched = [
                    (score, sec)
                    for score, sec in semantic_candidates
                    if any(_citation_matches_family_part(sec.get("citation", ""), prefix) for prefix in broad_family_prefixes)
                ]
                if broad_matched or exact_candidates:
                    semantic_candidates = broad_matched

            family_part_prefixes = [
                prefix
                for prefix in (_extract_family_part_prefix(c) for c in explicit_citations if _is_specific_citation(c))
                if prefix
            ]
            if family_part_prefixes:
                family_matched = [
                    (score, sec)
                    for score, sec in semantic_candidates
                    if any(_citation_matches_family_part(sec.get("citation", ""), prefix) for prefix in family_part_prefixes)
                ]
                if family_matched or exact_candidates:
                    semantic_candidates = family_matched

        # 8. Merge: exact hits first (score=1.0), hybrid semantic+BM25 fill after
        candidates: list[tuple[float, dict]] = list(exact_candidates)
        for score, sec in semantic_candidates:
            if sec["section_id"] not in seen_ids:
                candidates.append((score, sec))
                seen_ids.add(sec["section_id"])

        # 9. Citation reranking — boost candidates that match the explicit citation prefix
        if explicit_citations and candidates:
            candidates = _rerank_by_citation(candidates, explicit_citations)

        # 10. Rank, deduplicate, take top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = _dedupe_candidates_by_citation(candidates)
        references = [_section_to_ref(score, sec) for score, sec in candidates[:top_k]]

        if not references:
            return _empty_response(request_id)

        # 10. LLM generates the answer from the actual regulatory text
        fallback = _build_fallback(query, references)
        answer_payload: dict = fallback

        if self._llm and self._enable_llm_answers:
            try:
                generated = self._llm.generate(
                    query=query,
                    references=references,
                    fallback_payload=fallback,
                )
                if generated:
                    answer_payload = generated
            except Exception:
                pass

        top = references[0]
        return SearchResponse(
            references=references,
            citations=_unique_preserve_order(r.citation for r in references),
            verbatim_text=" ".join(top.text.split())[:600],
            contextual_notes=[],
            confidence=85 if (self._llm and self._enable_llm_answers) else 50,
            request_id=request_id,
            **answer_payload,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_family(citation: str) -> str | None:
    m = _FAMILY_RE.match(citation.strip())
    return m.group(1).upper() if m else None


def _is_specific_citation(citation: str) -> bool:
    """Return True only for citations specific enough to warrant exact DB lookup.

    Broad family/part references like "CASR 91" or "AIP ENR 1.5" return alphabetically-early
    sections that have nothing to do with the query. Only look up section-level citations.

    AIP:  requires a section number — at least 4 whitespace-separated tokens
          e.g. "AIP ENR 1.5 6.2" (4 tokens) → yes; "AIP ENR 1.5" (3 tokens) → no
    CASR: requires a dot in the part number — e.g. "CASR 61.385" → yes; "CASR 61" → no
    """
    parts = _normalize_structured_citation(citation).split()
    if not parts:
        return False
    family = parts[0].upper()
    if family == "AIP":
        return len(parts) >= 4
    if family == "MOS" and len(parts) >= 3 and parts[1].lower() == "schedule":
        return True
    if len(parts) >= 2:
        token = parts[1]
        if family in {"CAR", "CAA"}:
            return "." in token or bool(re.fullmatch(r"\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))?", token))
        return "." in token or "(" in token
    return False


def _is_candidate_family_consistent(section: dict) -> bool:
    citation_family = _detect_family(section.get("citation", ""))
    if not citation_family:
        return True
    section_family = (section.get("regulation_type") or "").strip().upper()
    return not section_family or citation_family == section_family


def _collect_prefix_candidates(
    canonical_store: CanonicalStore,
    citations: list[str],
    seen_ids: set[str],
    query_text: str,
) -> list[tuple[float, dict]]:
    """Collect deterministic citation-prefix matches for known query intents."""
    candidates: list[tuple[float, dict]] = []
    for index, citation in enumerate(citations):
        score = max(0.995, 1.0 - (index * 0.001))
        matches = canonical_store.get_sections_by_citation_prefix(citation, limit=200)
        matches.sort(key=lambda sec: _candidate_query_overlap(sec, query_text), reverse=True)
        for sec in matches:
            if _is_candidate_family_consistent(sec) and sec["section_id"] not in seen_ids:
                candidates.append((score, sec))
                seen_ids.add(sec["section_id"])
    return candidates


def _route_known_query(query: str) -> dict[str, Any] | None:
    """Provide deterministic routing for stable legal intents that underperform semantically."""
    normalized = " ".join(re.sub(r"[\"'`]", "", query.lower()).split())
    competency_terms = ("competency", "standard", "standards", "training", "mos", "schedule", "syllabus")

    if (
        any(
            term in normalized
            for term in (
                "low flying",
                "below 500 feet agl",
                "below 500 ft agl",
                "500 feet agl",
                "500 ft agl",
            )
        )
        and any(term in normalized for term in ("minimum safe altitude", "minimum height", "low flying"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.267 minimum height rules other areas 500 ft above the highest "
                "feature or obstacle within a horizontal radius of 300 m"
            ),
            "preferred_citations": ["CASR 91.267"],
        }

    if (
        "speed" in normalized
        and any(term in normalized for term in ("10,000", "10000", "below 10", "below ten"))
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.4 4.1 speed limitation 250 knots below 10000 ft "
                "class G airspace summary table"
            ),
            "preferred_citations": ["AIP ENR 1.4 4.1"],
        }

    if (
        "fuel" in normalized
        and "vfr" in normalized
        and "day" in normalized
        and any(
            term in normalized
            for term in (
                "requirement",
                "requirements",
                "reserve",
                "minimum",
                "carry",
                "must i carry",
                "must carry",
                "how much fuel",
                "need to carry",
            )
        )
    ):
        return {
            "regulation_hint": "MOS",
            "search_text": (
                "MOS 19.02 fuel requirements VFR flight by day final reserve fuel "
                "minimum safe fuel aeroplane"
            ),
            "preferred_citations": ["MOS 19.02"],
        }

    if (
        "fuel" in normalized
        and any(term in normalized for term in ("part 91", "small aeroplane", "small airplane", "fixed-wing", "fixed wing"))
    ):
        return {
            "regulation_hint": "MOS",
            "search_text": (
                "MOS 19.02 fuel requirements for VFR flight by day small aeroplane "
                "fixed wing Part 91 final reserve fuel"
            ),
            "preferred_citations": ["MOS 19.02"],
        }

    if (
        "class g" in normalized
        and any(term in normalized for term in ("visibility", "cloud clearance", "clear of cloud", "vmc"))
    ):
        return {
            "regulation_hint": "MOS",
            "search_text": (
                "MOS 2.07 VMC criteria class G airspace visibility clear of cloud "
                "in sight of ground or water CASR 91.280"
            ),
            "preferred_citations": ["MOS 2.07", "CASR 91.280"],
        }

    if (
        "medical" in normalized
        and any(term in normalized for term in ("ppl", "private pilot", "class 2 medical", "class 2"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 67.160 class 2 medical certificate requirements private pilot "
                "licence PPL CASR 67.165"
            ),
            "preferred_citations": ["CASR 67.160", "CASR 67.165"],
        }

    if "air transport" in normalized and "aerial work" in normalized:
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 119.010 definition of Australian air transport operation "
                "CASR 138.010 definition of aerial work operation"
            ),
            "preferred_citations": ["CASR 119.010", "CASR 138.010"],
        }

    if (
        "equipment" in normalized
        and any(term in normalized for term in ("ifr", "instrument flight rules", "instrument flight"))
        and not any(term in normalized for term in competency_terms)
    ):
        return {
            "regulation_hint": "MOS",
            "search_text": (
                "MOS 26.08 aeroplane IFR flight equipment requirements "
                "MOS 26.12 rotorcraft IFR flight"
            ),
            "preferred_citations": ["MOS 26.08", "MOS 26.12"],
        }

    if "journey log" in normalized:
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.120 journey logs flights outside Australian territory "
                "MOS 5.02 journey log information before international flight "
                "MOS 5.03 journey log information after international flight"
            ),
            "preferred_citations": ["CASR 91.120", "MOS 5.02", "MOS 5.03"],
        }

    if "maintenance release" in normalized:
        return {
            "regulation_hint": "CAR",
            "search_text": (
                "CAR 43 maintenance releases in respect of Australian aircraft "
                "CAR 47 maintenance release cease to be in force validity"
            ),
            "preferred_citations": ["CAR 43", "CAR 47"],
        }

    if "airworthiness directive" in normalized or re.search(r"\bad\b", normalized):
        if any(term in normalized for term in ("purpose", "what is", "meaning", "requirements", "directive")):
            return {
                "regulation_hint": "CASR",
                "search_text": (
                    "CASR 202.170 airworthiness directives meaning purpose compliance "
                    "CASR 202.171 exemption variation CASR 202.172 exemption from requirement"
                ),
                "preferred_citations": ["CASR 202.170", "CASR 202.171", "CASR 202.172"],
            }

    if (
        any(term in normalized for term in ("private pilot licence", "private pilot license", "ppl holder", "part 61 ppl", "ppl"))
        and any(term in normalized for term in ("compensation", "hire"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.505 privileges of private pilot licences private operation "
                "passengers compensation hire limitations"
            ),
            "preferred_citations": ["CASR 61.505"],
        }

    if (
        any(term in normalized for term in ("head of operations", "hoo"))
        and any(term in normalized for term in ("flying school", "part 141", "part 142", "training provider"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 141.020 definition of key personnel head of operations part 141 "
                "CASR 141.030 chief flying instructor head of operations duties "
                "CASR 141.045 operator responsibilities key personnel"
            ),
            "preferred_citations": ["CASR 141.020", "CASR 141.030", "CASR 141.045"],
        }

    if (
        any(term in normalized for term in ("foreign pilot licence", "foreign pilot license", "overseas flight crew licence"))
        or (
            "convert" in normalized
            and "foreign" in normalized
            and any(term in normalized for term in ("licence", "license"))
        )
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.275 overseas flight crew authorisations recognition "
                "CASR 61.280 bilateral agreements CASR 61.290 certificates of validation"
            ),
            "preferred_citations": ["CASR 61.275", "CASR 61.280", "CASR 61.290"],
        }

    if "mayday" in normalized and "pan-pan" in normalized:
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP GEN 3.4 7.14.2 mayday pan-pan distress urgency message procedures "
                "AIP ENR 1.14 4.2.1 emergency declaration"
            ),
            "preferred_citations": ["AIP GEN 3.4 7.14.2", "AIP ENR 1.14 4.2.1"],
        }

    if "knowledge deficiency" in normalized or re.search(r"\bkdrs?\b", normalized):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.230 aeronautical knowledge examinations knowledge "
                "deficiency reports KDR"
            ),
            "preferred_citations": ["CASR 61.230"],
        }

    if (
        "night vfr" in normalized
        and any(term in normalized for term in ("rules", "operations", "requirements", "conduct"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.970 limitations on exercise of privileges of night VFR ratings "
                "CASR 61.965 recent experience night VFR CASR 61.980 night VFR endorsement"
            ),
            "preferred_citations": ["CASR 61.970", "CASR 61.965", "CASR 61.980"],
        }

    if (
        "general operating rules" in normalized
        and any(term in normalized for term in ("all australian pilots", "foundational", "which document", "contains"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.010 CASR 91.015 Part 91 general operating and flight rules "
                "all Australian pilots application of Part 91"
            ),
            "preferred_citations": ["CASR 91.010", "CASR 91.015"],
        }

    if (
        ("performance based navigation" in normalized or re.search(r"\bpbn\b", normalized))
        and any(term in normalized for term in ("recency", "recent experience", "currency"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.870 limitations exercise privileges instrument rating recent "
                "experience PBN CASR 61.875"
            ),
            "preferred_citations": ["CASR 61.870", "CASR 61.875"],
        }

    if "performance based navigation" in normalized or re.search(r"\bpbn\b", normalized):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP GEN 1.5 8.1 performance based navigation PBN requirements advice "
                "and information area navigation"
            ),
            "preferred_citations": ["AIP GEN 1.5 8.1"],
        }

    # Flight review / biennial flight review / proficiency check → CASR 61.745
    if any(term in normalized for term in ("flight review", "biennial flight review", "proficiency check")):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.745 limitations exercise privileges aircraft class rating "
                "flight review biennial check CASR 61.126"
            ),
            "preferred_citations": ["CASR 61.745", "CASR 61.126"],
        }

    # Instrument recency / IFR currency — "currency" alone hits financial-currency text in AIP GEN 1.3
    if any(
        term in normalized
        for term in (
            "instrument currency",
            "ifr currency",
            "instrument recency",
            "iir recency",
            "recent instrument experience",
            "instrument recent experience",
        )
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.870 limitations exercise privileges instrument rating recent experience "
                "IFR recency single pilot CASR 61.875"
            ),
            "preferred_citations": ["CASR 61.870", "CASR 61.875"],
        }

    if (
        (
            any(term in normalized for term in ("pilot in command time", "pic time", "pilot in command hours"))
            and any(term in normalized for term in ("log", "logging", "record", "logbook"))
        )
        or "log pilot in command" in normalized
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 61.090 definition of flight time as pilot in command for Part 61 "
                "CASR 61.345 personal logbooks pilots CASR 61.355 retention "
                "CASR 61.365 production"
            ),
            "preferred_citations": ["CASR 61.090", "CASR 61.345", "CASR 61.355", "CASR 61.365"],
        }

    # Passenger safety briefing — GA rule is CASR 91.565, not CASR 121/133/135 (air transport)
    if any(
        term in normalized
        for term in ("passenger safety briefing", "passenger briefing", "pre-flight briefing passengers")
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.565 passengers safety briefings instructions pilot in command "
                "pre-flight passenger briefing requirements"
            ),
            "preferred_citations": ["CASR 91.565"],
        }

    # MEL / minimum equipment list / inoperative instruments → CASR 91.925
    if any(
        term in normalized
        for term in (
            "minimum equipment list",
            "mel ",
            " mel",
            "inoperative instrument",
            "inoperative equipment",
            "fly with inoperative",
            "operate with inoperative",
        )
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.925 minimum equipment list MEL master minimum equipment list "
                "inoperative aircraft instrument"
            ),
            "preferred_citations": ["CASR 91.925", "CASR 91.930"],
        }

    # VFR weather minima (general) → AIP ENR 1.1
    # Guard: don't match queries already routed by the class-G VMC rule above
    if any(
        term in normalized
        for term in (
            "minimum weather",
            "vfr weather",
            "vfr met minima",
            "vfr meteorological",
            "minimum visibility vfr",
            "weather minima vfr",
        )
    ) and "class g" not in normalized:
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.1 VFR meteorological minima visibility cloud clearance "
                "VMC flight under VFR day night"
            ),
            "preferred_citations": ["AIP ENR 1.1 2.8.2.2", "AIP ENR 1.1 4.2.1"],
        }

    if any(term in normalized for term in ("circle-to-land", "circle to land", "circling approach", "visual circling")):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.5 1.6 circling approaches visual circling limitations "
                "AIP ENR 1.5 1.6.1 AIP ENR 1.5 1.6.2 AIP ENR 1.5 1.6.3"
            ),
            "preferred_citations": [
                "AIP ENR 1.5 1.6",
                "AIP ENR 1.5 1.6.1",
                "AIP ENR 1.5 1.6.2",
                "AIP ENR 1.5 1.6.3",
            ],
        }

    if (
        any(term in normalized for term in ("descend below the mda", "descend below mda", "below the mda/da", "below mda/da"))
        or (
            "descend below" in normalized
            and any(term in normalized for term in ("decision altitude", "minimum descent altitude", "mda", "da"))
        )
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.5 1.7.2 descent below straight-in MDA "
                "AIP ENR 1.5 1.8.4 descent below MDA"
            ),
            "preferred_citations": ["AIP ENR 1.5 1.7.2", "AIP ENR 1.5 1.8.4"],
        }

    if (
        any(term in normalized for term in ("da vs mda", "mda vs da", "mda/da"))
        or ("decision altitude" in normalized and "minimum descent altitude" in normalized)
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP GEN 2.2 1 decision altitude height DA H minimum descent altitude "
                "height MDA H definitions"
            ),
            "preferred_citations": ["AIP GEN 2.2 1"],
        }

    if "rnp" in normalized and "rnav" in normalized:
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP GEN 2.2 1 Required Navigation Performance RNP definition "
                "RNAV Specification definition area navigation"
            ),
            "preferred_citations": ["AIP GEN 2.2 1"],
        }

    if (
        "aerodrome elevation" in normalized
        and any(term in normalized for term in ("define", "definition", "what is"))
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": "AIP GEN 2.2 1 Aerodrome Elevation definition",
            "preferred_citations": ["AIP GEN 2.2 1"],
        }

    if (
        "standard terminal arrival" in normalized
        or ("star" in normalized and any(term in normalized for term in ("define", "what is", "meaning")))
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP GEN 2.2 1 Standard Instrument Arrival STAR "
                "AIP ENR 1.1 2.2.5"
            ),
            "preferred_citations": ["AIP GEN 2.2 1", "AIP ENR 1.1 2.2.5"],
        }

    if "visual docking guidance" in normalized or "vdgs" in normalized:
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP AD 1.1 5.1 Visual Docking Guidance Systems VDGS used in Australia "
                "AIP AD 1.1 5.2 apron chart bays stands"
            ),
            "preferred_citations": ["AIP AD 1.1 5.1", "AIP AD 1.1 5.2"],
        }

    if "clearway" in normalized or "cwy" in normalized:
        return {
            "regulation_hint": "AIP",
            "search_text": "AIP GEN 2.2 1 Clearway CWY definition declared distances performance",
            "preferred_citations": ["AIP GEN 2.2 1"],
        }

    if (
        any(term in normalized for term in ("instrument approach", "ifr approach"))
        and any(term in normalized for term in competency_terms)
    ):
        return {
            "regulation_hint": "MOS",
            "search_text": (
                "MOS Schedule 4 Section M instrument rating instrument approach "
                "competency standards CPL training"
            ),
            "preferred_citations": ["MOS Schedule 4 Section M"],
        }

    # Instrument approach procedures / approach minima → AIP ENR 1.5
    if any(
        term in normalized
        for term in (
            "instrument approach",
            "ifr approach",
            "approach procedure",
            "approach minima",
            "approach chart",
        )
    ) and not any(term in normalized for term in competency_terms):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.5 instrument approach procedures minima ceiling visibility "
                "alternate aerodrome approach chart"
            ),
            "preferred_citations": ["AIP ENR 1.5 6.1.1", "AIP ENR 1.5 1.6"],
        }

    if (
        any(term in normalized for term in ("cruising level", "cruising levels", "flight level", "semicircular rule"))
        and any(term in normalized for term in ("ifr", "northbound", "southbound", "track", "cruising"))
    ):
        return {
            "regulation_hint": "CASR",
            "search_text": (
                "CASR 91.290 specified IFR cruising levels northbound southbound "
                "track CASR 91.295 AIP ENR 1.1 4.6"
            ),
            "preferred_citations": ["CASR 91.290", "CASR 91.295"],
        }

    if (
        "gnss" in normalized
        and any(term in normalized for term in ("ifr", "instrument flight", "requirements", "approval"))
    ):
        return {
            "regulation_hint": "AIP",
            "search_text": (
                "AIP ENR 1.1 4.8 GNSS requirements IFR RNP RNAV approval navigation "
                "specification AIP ENR 1.1 4.9"
            ),
            "preferred_citations": ["AIP ENR 1.1 4.8", "AIP ENR 1.1 4.9"],
        }

    return None


def _route_explicit_citation_query(query: str, citations: list[str]) -> dict[str, Any] | None:
    """Provide structured routing for explicit but broad legal citations."""
    normalized_query = " ".join(query.lower().split())
    for citation in citations:
        if _is_specific_citation(citation):
            continue
        family = _detect_family(citation)
        if not family:
            continue
        if family == "CASR" and re.fullmatch(r"CASR\s+1998", citation, re.IGNORECASE):
            return {
                "regulation_hint": "CASR",
                "search_text": "Civil Aviation Safety Regulations 1998 commencement definitions applicability",
                "preferred_citations": ["CASR 1."],
            }
        broad_prefix = _extract_broad_prefix(citation)
        if broad_prefix:
            return {
                "regulation_hint": family,
                "search_text": f"{citation} {' '.join(normalized_query.split()[:12])}",
                "preferred_citations": [broad_prefix],
            }
    return None


def _rerank_by_citation(
    candidates: list[tuple[float, dict]],
    explicit_citations: list[str],
) -> list[tuple[float, dict]]:
    """Boost candidates matching the explicit citation; penalise over-broad hits.

    Boost  +0.15: citation starts with the requested prefix (direct match / child section).
    Penalty -0.10: requested citation starts with this candidate's citation (candidate is
                   a parent/ancestor — too broad for a specific section request).
    Score is clamped to [0.0, 1.0] after adjustment; exact-lookup hits at 1.0 are unchanged.
    """
    result = []
    for score, sec in candidates:
        if score >= 1.0:
            result.append((score, sec))
            continue
        cit_lower = sec.get("citation", "").lower()
        adjustment = 0.0
        for ec in explicit_citations:
            ec_lower = ec.lower()
            if cit_lower.startswith(ec_lower):
                adjustment = 0.15
                break
            if ec_lower.startswith(cit_lower + " ") or ec_lower.startswith(cit_lower + "."):
                adjustment = -0.10
                break
        result.append((max(0.0, min(1.0, score + adjustment)), sec))
    return result


def _candidate_query_overlap(section: dict, query_text: str) -> tuple[int, int, int]:
    normalized_query = re.sub(r"[^a-z0-9 ]+", " ", query_text.lower())
    query_tokens = {
        token
        for token in normalized_query.split()
        if len(token) >= 3 and token not in {"what", "when", "with", "that", "this", "from"}
    }
    title = str(section.get("title", "")).lower()
    haystack = " ".join(
        (
            title,
            str(section.get("text", "")).lower(),
            str(section.get("citation", "")).lower(),
        )
    )
    overlap = sum(1 for token in query_tokens if token in haystack)
    title_hits = sum(1 for token in query_tokens if token in title)
    return (title_hits, overlap, -len(title))


def _dedupe_candidates_by_citation(candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    deduped: list[tuple[float, dict]] = []
    seen: set[str] = set()
    for score, sec in candidates:
        citation = " ".join(str(sec.get("citation", "")).split())
        key = citation.lower() if citation else str(sec.get("section_id", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, sec))
    return deduped


def _extract_chapter_prefix(citation: str) -> str:
    """Return the AIP chapter prefix from a full citation.

    "AIP ENR 1.5 6.2" → "AIP ENR 1.5"
    "AIP GEN 3.4 2.1" → "AIP GEN 3.4"
    "CASR 61.385"     → ""  (not an AIP citation)
    """
    m = re.match(r"^(AIP\s+(?:GEN|ENR|AD)\s+\d+\.\d+)", citation.strip(), re.IGNORECASE)
    return m.group(1) if m else ""


def _normalize_structured_citation(citation: str) -> str:
    parts = citation.strip().split()
    if not parts:
        return ""
    family = parts[0].upper()
    if len(parts) >= 3 and parts[1].lower() == "part":
        return " ".join([family, parts[2], *parts[3:]]).strip()
    return " ".join([family, *parts[1:]]).strip()


def _extract_broad_prefix(citation: str) -> str:
    parts = _normalize_structured_citation(citation).split()
    if len(parts) < 2:
        return ""
    family, token = parts[0].upper(), parts[1]
    if family == "CASR" and token == "1998":
        return ""
    if family in {"CASR", "MOS"} and re.fullmatch(r"\d+(?:\.)?", token):
        return f"{family} {token.rstrip('.') }."
    if family == "CAR" and re.fullmatch(r"\d+(?:\.)?", token):
        return f"{family} {token.rstrip('.') }."
    return ""


def _extract_family_part_prefix(citation: str) -> str:
    parts = _normalize_structured_citation(citation).split()
    if len(parts) < 2:
        return ""
    family, token = parts[0].upper(), parts[1]
    if family == "AIP":
        return _extract_chapter_prefix(citation)
    if family in {"CASR", "MOS", "CAO"} and "." in token:
        return f"{family} {token.split('.', 1)[0]}."
    if family in {"CAR", "CAA"}:
        return f"{family} {token}"
    return ""


def _citation_matches_family_part(citation: str, prefix: str) -> bool:
    lowered = citation.lower()
    prefix_lower = prefix.lower()
    if prefix_lower.endswith("."):
        return lowered.startswith(prefix_lower)
    return (
        lowered == prefix_lower
        or lowered.startswith(prefix_lower + ".")
        or lowered.startswith(prefix_lower + "(")
        or lowered.startswith(prefix_lower + " ")
    )


def _extract_section_ids(raw: dict) -> list[str]:
    metas = (raw.get("metadatas") or [[]])[0]
    return [m["section_id"] for m in metas if "section_id" in m]


def _looks_aviation_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return bool(AVIATION_QUERY_PATTERN.search(normalized)) or any(term in normalized for term in AVIATION_QUERY_TERMS)


def _section_to_ref(score: float, sec: dict) -> ReferenceItem:
    return ReferenceItem(
        section_id=sec.get("section_id", ""),
        regulation_id=sec.get("regulation_id", ""),
        citation=sec.get("citation", ""),
        title=sec.get("title", ""),
        regulation_type=sec.get("regulation_type", ""),
        source_file=sec.get("source_file", ""),
        source_url=sec.get("source_url", ""),
        text=sec.get("text", ""),
        part=sec.get("part", ""),
        page_ref=sec.get("page_ref", ""),
        table_ref=sec.get("table_ref", ""),
        section_index=sec.get("section_order", 0),
        chunk_index=0,
        score=round(score, 4),
    )


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _build_fallback(query: str, references: list[ReferenceItem]) -> dict:
    top = references[0]
    citations_used = [r.citation for r in references]
    return {
        "answer": f"{top.citation}: {' '.join(top.text.split())[:400]}",
        "legal_explanation": f"See {top.citation} — {top.title}.",
        "plain_english": f"Refer to {top.citation} for the applicable requirement.",
        "example": "",
        "study_questions": [
            f"What does {citations_used[0]} require in exact wording?",
            "What conditions or exceptions apply to this rule?",
            "Who does this regulation apply to?",
            "What are the consequences of non-compliance?",
            "How does this interact with other related regulations?",
        ],
        "study_answers": [
            "Refer to the cited regulation text for the exact wording.",
            "Review the exceptions and conditions listed in the cited section.",
            "The regulation specifies the applicable persons and operations.",
            "Penalties and enforcement are detailed in the Act and associated regulations.",
            "Cross-reference with related CASR parts and AIP sections as applicable.",
        ],
    }


def _empty_response(request_id: str) -> SearchResponse:
    return SearchResponse(
        answer="No matching regulatory text found for this query.",
        legal_explanation="",
        plain_english="",
        example="",
        study_questions=[],
        study_answers=[],
        references=[],
        citations=[],
        verbatim_text="",
        contextual_notes=[],
        confidence=0,
        request_id=request_id,
    )
