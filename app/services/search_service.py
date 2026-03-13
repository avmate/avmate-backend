"""Aviation regulatory search service.

Retrieval flow (per query):
  1. LLM interprets query  → regulation_type hint + rewritten query + keywords
  2. Semantic search       → ChromaDB (filtered by regulation_type when known)
  3. Canonical join        → resolve section_ids to full metadata from SQLite
  4. Citation filter       → narrow to explicitly requested citations when present
  5. Lexical fallback      → canonical store keyword search when results are sparse
  6. LLM answer generation → Claude reads actual regulation text, returns structured answer
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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

        # 2. LLM interprets query for better semantic retrieval
        regulation_hint: str | None = None
        search_text = query
        keywords: list[str] = []

        if self._llm and self._enable_llm_query_assist:
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

        # 3. Semantic retrieval from ChromaDB
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

        # 4. Resolve section_ids → full metadata (regulation_type, citation, text, etc.)
        sections_by_id = {
            s["section_id"]: s
            for s in self._canonical_store.get_sections_by_ids(section_ids)
        }

        seen_ids: set[str] = set()
        candidates: list[tuple[float, dict]] = []
        for sid, dist in zip(section_ids, distances):
            if sid in seen_ids:
                continue
            section = sections_by_id.get(sid)
            if not section:
                continue
            score = max(0.0, 1.0 - float(dist))
            candidates.append((score, section))
            seen_ids.add(sid)

        # 5. Narrow to explicit citations when present
        if explicit_citations:
            explicit_lower = [c.lower() for c in explicit_citations]
            citation_matched = [
                (s, sec)
                for s, sec in candidates
                if any(
                    sec.get("citation", "").lower().startswith(pfx)
                    for pfx in explicit_lower
                )
            ]
            if citation_matched:
                candidates = citation_matched

        # 6. Lexical fallback when semantic results are sparse OR low-confidence
        # (table-heavy sections like ENR 1.4 don't embed well, so keyword search supplements)
        top_score = candidates[0][0] if candidates else 0.0
        if len(candidates) < top_k or top_score < 0.3:
            terms = keywords or [t for t in query.lower().split() if len(t) >= 4]
            lexical = self._canonical_store.search_sections_by_terms(
                terms,
                limit=top_k * 10,
                regulation_type=regulation_hint,
            )
            for sec in lexical:
                if sec["section_id"] not in seen_ids:
                    candidates.append((0.4, sec))
                    seen_ids.add(sec["section_id"])

        # 7. Rank, deduplicate, take top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        references = [_section_to_ref(score, sec) for score, sec in candidates[:top_k]]

        if not references:
            return _empty_response(request_id)

        # 8. LLM generates the answer from the actual regulatory text
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
            citations=[r.citation for r in references],
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
