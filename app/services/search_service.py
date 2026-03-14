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
        explicit_citations = extract_citations(query)
        explicit_families = {_detect_family(c) for c in explicit_citations} - {None}
        query_route = _route_known_query(query) if not explicit_citations else None

        # 2. Exact citation lookup — deterministic, bypasses semantic for structured queries
        #    Only fires for specific section citations, not broad family/part references.
        #    "CASR 61.385" or "AIP ENR 1.5 6.2" → yes; "CASR 91" or "AIP ENR 1.5" → no.
        seen_ids: set[str] = set()
        exact_candidates: list[tuple[float, dict]] = []
        for citation in explicit_citations:
            if not _is_specific_citation(citation):
                continue
            for sec in self._canonical_store.get_sections_by_citation_prefix(citation):
                if sec["section_id"] not in seen_ids:
                    exact_candidates.append((1.0, sec))
                    seen_ids.add(sec["section_id"])
        if query_route:
            exact_candidates.extend(
                _collect_prefix_candidates(
                    self._canonical_store,
                    query_route["preferred_citations"],
                    seen_ids,
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
            sem_score = max(0.0, 1.0 - float(dist))
            score = max(sem_score, bm25_scores.get(sid, 0.0))
            semantic_candidates.append((score, section))
            merged_sids.add(sid)

        # Add BM25-only hits (not in semantic results)
        for sid, bm25_score in bm25_scores.items():
            if sid in seen_ids or sid in merged_sids:
                continue
            section = sections_by_id.get(sid)
            if section:
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
    parts = citation.strip().split()
    if not parts:
        return False
    family = parts[0].upper()
    if family == "AIP":
        return len(parts) >= 4
    if len(parts) >= 2:
        return "." in parts[1]
    return False


def _collect_prefix_candidates(
    canonical_store: CanonicalStore,
    citations: list[str],
    seen_ids: set[str],
) -> list[tuple[float, dict]]:
    """Collect deterministic citation-prefix matches for known query intents."""
    candidates: list[tuple[float, dict]] = []
    for index, citation in enumerate(citations):
        score = max(0.95, 0.99 - (index * 0.01))
        for sec in canonical_store.get_sections_by_citation_prefix(citation):
            if sec["section_id"] not in seen_ids:
                candidates.append((score, sec))
                seen_ids.add(sec["section_id"])
    return candidates


def _route_known_query(query: str) -> dict[str, Any] | None:
    """Provide deterministic routing for stable legal intents that underperform semantically."""
    normalized = " ".join(query.lower().split())

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


def _extract_chapter_prefix(citation: str) -> str:
    """Return the AIP chapter prefix from a full citation.

    "AIP ENR 1.5 6.2" → "AIP ENR 1.5"
    "AIP GEN 3.4 2.1" → "AIP GEN 3.4"
    "CASR 61.385"     → ""  (not an AIP citation)
    """
    m = re.match(r"^(AIP\s+(?:GEN|ENR|AD)\s+\d+\.\d+)", citation.strip(), re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_section_ids(raw: dict) -> list[str]:
    metas = (raw.get("metadatas") or [[]])[0]
    return [m["section_id"] for m in metas if "section_id" in m]


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
