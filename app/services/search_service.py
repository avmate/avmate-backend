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


STOP_WORDS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
    "why",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "with",
    "from",
    "is",
    "are",
    "be",
    "was",
    "were",
    "do",
    "does",
    "can",
    "could",
    "should",
    "would",
    "please",
    "tell",
    "about",
    "under",
    "using",
}
LOW_SIGNAL_TERMS = {
    "aircraft",
    "pilot",
    "flight",
    "category",
    "cat",
}

PAGE_REF_PATTERN = re.compile(r"\b(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\(?\d+\)?\b", re.IGNORECASE)
TABLE_REF_PATTERN = re.compile(r"\bTable\s+\d+(?:\.\d+)+\b", re.IGNORECASE)
SUBSECTION_PATTERN = re.compile(r"\b([1-9]\d?(?:\.\d+){1,4})\b")
AIP_SUBSECTION_LINE_PATTERN = re.compile(
    r"(?m)^(?P<label>[1-9]\d?(?:\.\d+){1,4})\s+(?P<heading>[^\n]{2,220})$"
)
PAGE_REF_PARSE_PATTERN = re.compile(
    r"^(?P<prefix>(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?)\s*-\s*\(?(?P<page>\d+)\)?$",
    re.IGNORECASE,
)
CIRCLING_ROW_PATTERN = re.compile(
    r"\b(?P<from>\d+)\s*FT\s+(?P<to>\d+)\s*FT\s+"
    r"(?P<a>[0-9.]+)\s*NM\s+(?P<b>[0-9.]+)\s*NM\s+(?P<c>[0-9.]+)\s*NM\s+(?P<d>[0-9.]+)\s*NM\b",
    re.IGNORECASE,
)
QNH_PRIORITY_SUBSECTIONS = ("1.4.1", "1.4", "5.3", "5.3.1", "5.3.2", "5.3.3", "5.3.4")
WEATHER_MINIMA_PRIORITY_SUBSECTIONS = ("6.2", "6.2.1", "6.2.2", "6.2.3", "6.2.4")


class SearchService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        canonical_store: CanonicalStore,
        llm_answer_service: "LLMAnswerService | None" = None,
        enable_llm_answers: bool = False,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._canonical_store = canonical_store
        self._llm_answer_service = llm_answer_service
        self._enable_llm_answers = enable_llm_answers

    def search(self, query: str, top_k: int, use_llm: bool = True) -> SearchResponse:
        query_profile = self._build_query_profile(query)
        query_embedding = self._embeddings.encode([query])
        if query_profile.get("strict_single_reference"):
            candidate_k = min(max(top_k * 24, 140), 260)
        else:
            candidate_k = min(max(top_k * 24, 120), 320)
        results = self._vector_store.query(query_embeddings=query_embedding, top_k=candidate_k)
        requested_citations = [citation.lower() for citation in extract_citations(query)]

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        section_ids: list[str] = []
        for metadata in metadatas:
            if metadata and metadata.get("section_id"):
                section_ids.append(metadata["section_id"])
        canonical_sections = {
            section["section_id"]: section for section in self._canonical_store.get_sections_by_ids(section_ids)
        }

        ranked_items: list[tuple[float, ReferenceItem]] = []
        fallback_semantic_items: list[tuple[float, ReferenceItem]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            semantic_score = max(0.0, min(1.0, 1.0 - distance))
            section_id = metadata.get("section_id", "")
            canonical = canonical_sections.get(section_id, {})
            raw_citation = str(canonical.get("citation", metadata.get("citation", "Unknown")) or "Unknown")
            regulation_type = str(canonical.get("regulation_type", "UNKNOWN") or "UNKNOWN")
            canonical_text = canonical.get("text", document)
            subsection_hint = ""
            focused_text = canonical_text
            if regulation_type.upper() == "AIP":
                subsection_hint, subsection_text = self._select_best_aip_subsection(
                    canonical_text,
                    raw_citation,
                    query_profile,
                )
                if subsection_text:
                    focused_text = subsection_text

            page_ref = self._resolve_page_ref(
                canonical_text,
                raw_citation,
                canonical.get("page_ref", ""),
                subsection_hint,
            )
            table_ref = canonical.get("table_ref", "") or self._infer_table_ref(focused_text)
            citation = self._format_output_citation(
                raw_citation,
                regulation_type,
                page_ref,
                table_ref,
                focused_text,
                subsection_hint=subsection_hint,
            )
            combined_score, passes_gate = self._combine_score(
                query_profile=query_profile,
                document=focused_text,
                citation=raw_citation,
                regulation_type=regulation_type,
                semantic_score=semantic_score,
                requested_citations=requested_citations,
                page_ref=page_ref,
            )
            reference = ReferenceItem(
                section_id=section_id,
                regulation_id=canonical.get("regulation_id", raw_citation),
                citation=citation,
                title=self._refine_title(canonical.get("title", "Untitled"), raw_citation, focused_text),
                regulation_type=regulation_type,
                source_file=canonical.get("source_file", ""),
                source_url=canonical.get("source_url", ""),
                text=focused_text,
                part=canonical.get("part", ""),
                page_ref=page_ref,
                table_ref=table_ref,
                section_index=int(metadata.get("section_index", 0)),
                chunk_index=int(metadata.get("chunk_index", 0)),
                score=round(combined_score, 4),
            )
            fallback_semantic_items.append((semantic_score, reference))
            if passes_gate:
                ranked_items.append((combined_score, reference))

        if not ranked_items:
            ranked_items = self._rescue_ranked_items(fallback_semantic_items, query_profile, requested_citations)

        references = self._dedupe_references(ranked_items, top_k)
        references = self._filter_final_references(references, query_profile, top_k)
        lexical_fallback_refs: list[ReferenceItem] = []
        should_try_lexical_fallback = (
            not references
            or bool(query_profile.get("weather_minima_intent"))
            or bool(query_profile.get("qnh_intent"))
        )
        if should_try_lexical_fallback:
            lexical_top_k = max(top_k, 12) if query_profile.get("qnh_intent") else top_k
            lexical_fallback_refs = self._lexical_fallback_references(query_profile, requested_citations, lexical_top_k)
            if lexical_fallback_refs:
                if not references:
                    references = lexical_fallback_refs
                else:
                    top = references[0]
                    top_text = " ".join((top.text or "").split()).lower()
                    top_toc = self._table_of_contents_penalty(top.text) >= 0.1
                    top_weather_hit = bool(re.search(r"\b(?:special\s+)?alternate\s+weather\s+minima\b", top_text))
                    if top_toc or not top_weather_hit:
                        references = lexical_fallback_refs
                    elif query_profile.get("qnh_intent"):
                        references = self._merge_references(references, lexical_fallback_refs, limit=max(top_k * 2, 10))
        if query_profile.get("weather_minima_intent"):
            references = self._prioritize_weather_minima_references(references, top_k)
            references = self._ensure_parent_subsection_reference(references, parent_label="6.2", limit=top_k)
        if query_profile.get("qnh_intent"):
            references = self._prioritize_qnh_references(references, query_profile, top_k)
            references = self._ensure_parent_subsection_reference(references, parent_label="5.3", limit=top_k)

        if not references:
            return SearchResponse(
                answer="No matching regulation text was found in the current index.",
                legal_explanation="The current regulation index did not produce any section that could be cited responsibly.",
                plain_english="No section passed the relevance checks for this query.",
                example="Try a narrower query using exact regulation wording or the expected citation.",
                study_questions=["Which exact regulation number is most relevant to your question?"],
                study_answers=["Identify the controlling citation first, then read the operative text before forming an answer."],
                references=[],
                citations=[],
                verbatim_text="",
                contextual_notes=["Add or rebuild index data if expected regulations are missing."],
                confidence=0,
                explanation="No results passed semantic and lexical relevance gates.",
            )

        top_reference = references[0]
        citations: list[str] = []
        for item in references:
            if item.citation not in citations:
                citations.append(item.citation)

        answer = self._build_answer(query, top_reference)
        legal_explanation = self._build_legal_explanation(query, references)
        plain_english = self._build_plain_english(query, top_reference)
        example = self._build_example(query, top_reference)
        study_questions = [
            f"What does {citations[0]} require in the exact wording of the regulation?",
            "Which conditions, exceptions, or definitions in the cited text could change the answer?",
            "What related regulation or MOS provision should be cross-checked before concluding?",
            "What source document and page reference contain the controlling text?",
            "If the text includes a table or standard, what operational number should you extract from it?",
        ]
        study_answers = [
            f"Read the verbatim text of {citations[0]} and restate the operative requirement without changing its meaning.",
            "Check the same section and nearby cited material for carve-outs, notes, and defined terms.",
            f"Review {', '.join(citations[1:3]) if len(citations) > 1 else 'related AIP or MOS material'} before relying on a conclusion.",
            f"The controlling source is {top_reference.source_file} {top_reference.page_ref or ''}".strip(),
            "Extract the specific table value, limit, or radius from the verbatim source text and tie it back to the aircraft category or condition asked about.",
        ]
        contextual_notes = self._build_contextual_notes(query, references)
        confidence = max(0, min(99, int(round(references[0].score * 100))))
        if self._query_targets_circling_minima(query):
            category = self._extract_aircraft_category(query)
            if category:
                circling = self._extract_circling_radius_data(" ".join(top_reference.text.split()), category)
                if circling.get("radius_nm"):
                    confidence = max(confidence, 74)

        deterministic_payload: dict[str, Any] = {
            "answer": answer,
            "legal_explanation": legal_explanation,
            "plain_english": plain_english,
            "example": example,
            "study_questions": study_questions,
            "study_answers": study_answers,
        }
        if use_llm and self._enable_llm_answers and self._llm_answer_service and self._llm_answer_service.enabled:
            try:
                llm_output = self._llm_answer_service.generate(
                    query=query,
                    references=references,
                    fallback_payload=deterministic_payload,
                )
            except Exception as exc:
                llm_output = None
                contextual_notes.append(f"Grounded LLM synthesis unavailable: {exc}")
            if llm_output:
                answer = llm_output.get("answer", answer)
                legal_explanation = llm_output.get("legal_explanation", legal_explanation)
                plain_english = llm_output.get("plain_english", plain_english)
                example = llm_output.get("example", example)
                study_questions = llm_output.get("study_questions", study_questions)
                study_answers = llm_output.get("study_answers", study_answers)
                contextual_notes.append("Grounded LLM synthesis enabled using only retrieved references.")

        return SearchResponse(
            answer=answer,
            legal_explanation=legal_explanation,
            plain_english=plain_english,
            example=example,
            study_questions=study_questions,
            study_answers=study_answers,
            references=references,
            citations=citations,
            verbatim_text=top_reference.text,
            contextual_notes=contextual_notes,
            confidence=confidence,
            explanation="Results are gated by semantic similarity plus mandatory lexical intent matches before citation selection.",
        )

    def _build_query_profile(self, query: str) -> dict:
        query_lower = query.lower()
        terms = [
            token
            for token in re.findall(r"[a-z0-9.()/-]+", query_lower)
            if len(token) > 2 and token not in STOP_WORDS and token not in LOW_SIGNAL_TERMS
        ]
        phrases = [phrase for phrase in re.findall(r"\b[a-z0-9]+\s+[a-z0-9]+\b", query_lower) if len(phrase) > 7]
        explicit_subsection_labels = [
            label for label in re.findall(r"\b(?:subsection|section)\s+([1-9]\d?(?:\.\d+){1,4})\b", query_lower)
        ]
        explicit_page_hints = [
            " ".join(match.group(1).split())
            for match in re.finditer(r"\b((?:gen|enr|ad)\s+\d+(?:\.\d+)?)\b", query_lower)
        ]
        explicit_page_hints = list(dict.fromkeys(explicit_page_hints))
        for citation in extract_citations(query):
            label = self._citation_subsection_label(citation)
            if label:
                explicit_subsection_labels.append(label)
        explicit_subsection_labels = list(dict.fromkeys(explicit_subsection_labels))

        required_patterns: list[re.Pattern[str]] = []
        category_match = re.search(r"\bcat(?:egory)?\s*([abcd])\b", query_lower)
        if category_match:
            cat = category_match.group(1)
            required_patterns.append(re.compile(rf"\bcat(?:egory)?\s*{cat}\b", re.IGNORECASE))
        if "circling" in query_lower:
            required_patterns.append(re.compile(r"\bcircling\b", re.IGNORECASE))
        has_measure = any(term in query_lower for term in ("radius", "radii", "minima", "minimum"))
        if "radius" in query_lower or "radii" in query_lower or "minima" in query_lower or "minimum" in query_lower:
            required_patterns.append(re.compile(r"\b(?:radi(?:us|i)|minima|minimum)\b", re.IGNORECASE))
        qnh_intent = "qnh" in query_lower
        if qnh_intent:
            required_patterns.append(re.compile(r"\bqnh\b", re.IGNORECASE))
        weather_minima_intent = (
            ("alternate" in query_lower)
            and ("weather" in query_lower)
            and any(token in query_lower for token in ("minima", "minimum"))
        )
        special_weather_minima_intent = weather_minima_intent and ("special" in query_lower)
        if weather_minima_intent:
            required_patterns.append(re.compile(r"\balternate\b", re.IGNORECASE))
            required_patterns.append(re.compile(r"\bweather\b", re.IGNORECASE))
            required_patterns.append(re.compile(r"\b(?:minima|minimum)\b", re.IGNORECASE))
            if special_weather_minima_intent:
                required_patterns.append(re.compile(r"\bspecial\b", re.IGNORECASE))
                required_patterns.append(
                    re.compile(r"\bspecial\s+alternate\s+weather\s+minima\b", re.IGNORECASE)
                )

        intent_tokens = [
            token
            for token in (
                "circling",
                "radius",
                "radii",
                "minima",
                "minimum",
                "table",
                "altitude",
                "fuel",
                "ifr",
                "vfr",
                "alternate",
                "weather",
                "qnh",
                "source",
                "sources",
                "altimeter",
                "forecast",
                "awis",
                "atis",
                "atc",
                "aais",
                "watir",
            )
            if token in query_lower
        ]
        numeric_intent = bool(
            re.search(
                r"\b(?:radius|radii|minima|minimum|nm|feet|foot|ft|kts|knots|altitude|distance)\b",
                query_lower,
            )
        )
        strict_single_reference = bool(category_match and "circling" in query_lower and numeric_intent)
        if weather_minima_intent:
            strict_single_reference = True
        aip_preferred_intent = bool(
            weather_minima_intent
            or ("circling" in query_lower and has_measure)
            or qnh_intent
            or ("aip" in query_lower)
        )
        return {
            "query_lower": query_lower,
            "terms": list(dict.fromkeys(terms)),
            "phrases": list(dict.fromkeys(phrases)),
            "required_patterns": required_patterns,
            "intent_tokens": intent_tokens,
            "numeric_intent": numeric_intent,
            "strict_single_reference": strict_single_reference,
            "weather_minima_intent": weather_minima_intent,
            "special_weather_minima_intent": special_weather_minima_intent,
            "aip_preferred_intent": aip_preferred_intent,
            "qnh_intent": qnh_intent,
            "explicit_subsection_labels": explicit_subsection_labels,
            "explicit_page_hints": explicit_page_hints,
        }

    def _combine_score(
        self,
        query_profile: dict,
        document: str,
        citation: str,
        regulation_type: str,
        semantic_score: float,
        requested_citations: list[str],
        page_ref: str = "",
    ) -> tuple[float, bool]:
        query_lower = query_profile["query_lower"]
        terms = query_profile["terms"]
        phrases = query_profile["phrases"]
        required_patterns = query_profile["required_patterns"]
        intent_tokens = query_profile["intent_tokens"]
        numeric_intent = bool(query_profile.get("numeric_intent"))
        strict_single_reference = bool(query_profile.get("strict_single_reference"))
        aip_preferred_intent = bool(query_profile.get("aip_preferred_intent"))
        qnh_intent = bool(query_profile.get("qnh_intent"))
        weather_minima_intent = bool(query_profile.get("weather_minima_intent"))
        special_weather_minima_intent = bool(query_profile.get("special_weather_minima_intent"))
        explicit_subsection_labels = query_profile.get("explicit_subsection_labels", [])
        explicit_page_hints = query_profile.get("explicit_page_hints", [])

        citation_lower = citation.lower()
        document_lower = document.lower()
        is_aip_result = str(regulation_type or "").upper() == "AIP" or citation_lower.startswith("aip ")
        citation_subsection = self._citation_subsection_label(citation)
        explicit_subsection_match = any(
            citation_subsection == label or citation_subsection.startswith(f"{label}.")
            for label in explicit_subsection_labels
        )
        page_ref_lower = " ".join((page_ref or "").split()).lower()
        explicit_page_hint_match = any(page_ref_lower.startswith(hint) for hint in explicit_page_hints)

        citation_exact = any(requested == citation_lower for requested in requested_citations)
        citation_mentioned = bool(citation_lower) and citation_lower in query_lower

        lexical_hits = sum(1 for term in terms if term in document_lower or term in citation_lower)
        lexical_ratio = lexical_hits / max(len(terms), 1)
        phrase_hits = sum(1 for phrase in phrases if phrase in document_lower)
        required_ok = all(pattern.search(document) or pattern.search(citation) for pattern in required_patterns)
        numeric_evidence = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|ft|m|kts|kt)\b", document_lower))
        circling_table_evidence = bool(
            re.search(r"\bad\s+circling\b", document_lower)
            and re.search(r"\bcat\s+a\b", document_lower)
            and re.search(r"\bcat\s+b\b", document_lower)
            and re.search(r"\bcat\s+c\b", document_lower)
            and re.search(r"\bcat\s+d\b", document_lower)
            and (
                re.search(r"\b0\s*ft\s+1000\s*ft\b", document_lower)
                or re.search(r"\btable\s+1\.1\b", document_lower)
                or re.search(r"\b0ft\s+1000ft\b", document_lower)
            )
        )
        qnh_evidence = bool(re.search(r"\bqnh\b", document_lower) or re.search(r"\bqnh\b", citation_lower))
        weather_phrase_hit = bool(re.search(r"\bspecial\s+alternate\s+weather\s+minima\b", document_lower))
        toc_penalty = self._table_of_contents_penalty(document)

        passes_gate = citation_exact or citation_mentioned
        if not passes_gate:
            passes_gate = semantic_score >= 0.58 or lexical_ratio >= 0.24 or (semantic_score >= 0.5 and lexical_ratio >= 0.16)
            if required_patterns and not required_ok:
                passes_gate = False
            if numeric_intent and not numeric_evidence and semantic_score < 0.72:
                passes_gate = False
            if strict_single_reference and not circling_table_evidence and semantic_score < 0.8:
                passes_gate = False
            if qnh_intent and not qnh_evidence and semantic_score < 0.9:
                passes_gate = False
            if explicit_subsection_labels and not explicit_subsection_match and semantic_score < 0.94:
                passes_gate = False
            if explicit_page_hints and not explicit_page_hint_match and semantic_score < 0.97:
                passes_gate = False
            if weather_minima_intent:
                if citation_subsection == "6.2":
                    pass
                elif citation_subsection.startswith("6.2."):
                    pass
                elif semantic_score < 0.95:
                    passes_gate = False
                if special_weather_minima_intent and not weather_phrase_hit and semantic_score < 0.92:
                    passes_gate = False
            if toc_penalty >= 0.2 and semantic_score < 0.85:
                passes_gate = False
            if aip_preferred_intent and not is_aip_result and semantic_score < 0.92:
                passes_gate = False

        score = semantic_score * 0.62
        score += lexical_ratio * 0.28
        score += min(phrase_hits * 0.04, 0.12)

        if citation_exact:
            score += 0.32
        elif citation_mentioned:
            score += 0.18

        if required_patterns:
            if required_ok:
                score += 0.12
            else:
                score -= 0.22

        if intent_tokens and any(token in document_lower for token in intent_tokens):
            score += 0.05
        if numeric_intent:
            score += 0.06 if numeric_evidence else -0.08
        if circling_table_evidence:
            score += 0.24
        if strict_single_reference and not circling_table_evidence:
            score -= 0.15
        if qnh_intent:
            score += 0.12 if qnh_evidence else -0.2
        if explicit_subsection_labels:
            score += 0.24 if explicit_subsection_match else -0.22
        if explicit_page_hints:
            score += 0.28 if explicit_page_hint_match else -0.32
        if weather_minima_intent:
            if citation_subsection == "6.2":
                score += 0.3
            elif citation_subsection.startswith("6.2."):
                score += 0.18
            else:
                score -= 0.24
            if special_weather_minima_intent:
                score += 0.16 if weather_phrase_hit else -0.16
        if aip_preferred_intent and not is_aip_result:
            score -= 0.35
        score -= toc_penalty

        return max(0.0, min(1.0, score)), passes_gate

    def _rescue_ranked_items(
        self,
        fallback_items: list[tuple[float, ReferenceItem]],
        query_profile: dict,
        requested_citations: list[str],
    ) -> list[tuple[float, ReferenceItem]]:
        rescued: list[tuple[float, ReferenceItem]] = []
        strict_rescue = bool(query_profile.get("strict_single_reference")) or bool(query_profile.get("required_patterns"))
        for semantic_score, item in sorted(fallback_items, key=lambda pair: pair[0], reverse=True):
            combined, passes_gate = self._combine_score(
                query_profile=query_profile,
                document=item.text,
                citation=item.citation,
                regulation_type=item.regulation_type,
                semantic_score=semantic_score,
                requested_citations=requested_citations,
                page_ref=item.page_ref,
            )
            if strict_rescue and not passes_gate:
                continue
            if semantic_score >= 0.5 or combined >= 0.45:
                item.score = round(combined, 4)
                rescued.append((combined, item))
            if len(rescued) >= 10:
                break
        return rescued

    def _lexical_fallback_references(
        self,
        query_profile: dict,
        requested_citations: list[str],
        top_k: int,
    ) -> list[ReferenceItem]:
        terms = [term for term in query_profile.get("terms", []) if len(term) >= 3]
        if not terms:
            return []

        regulation_hint = "AIP" if query_profile.get("aip_preferred_intent") else None
        query_terms = [term for term in terms if term not in {"aircraft", "pilot", "flight", "category", "cat"}]
        if not query_terms:
            query_terms = terms

        sections = self._canonical_store.search_sections_by_terms(
            query_terms,
            limit=5000,
            regulation_type=regulation_hint,
        )
        if not sections:
            return []

        required_patterns = query_profile.get("required_patterns", [])
        phrases = query_profile.get("phrases", [])
        intent_tokens = query_profile.get("intent_tokens", [])
        numeric_intent = bool(query_profile.get("numeric_intent"))
        qnh_intent = bool(query_profile.get("qnh_intent"))
        explicit_subsection_labels = query_profile.get("explicit_subsection_labels", [])
        explicit_page_hints = query_profile.get("explicit_page_hints", [])

        ranked: list[tuple[float, ReferenceItem]] = []
        for section in sections:
            raw_citation = str(section.get("citation", "Unknown") or "Unknown")
            regulation_type = str(section.get("regulation_type", "UNKNOWN") or "UNKNOWN")
            canonical_text = str(section.get("text", "") or "")
            if not canonical_text.strip():
                continue

            subsection_hint = ""
            focused_text = canonical_text
            if regulation_type.upper() == "AIP":
                subsection_hint, subsection_text = self._select_best_aip_subsection(
                    canonical_text,
                    raw_citation,
                    query_profile,
                )
                if subsection_text:
                    focused_text = subsection_text

            page_ref = self._resolve_page_ref(
                canonical_text,
                raw_citation,
                section.get("page_ref", ""),
                subsection_hint,
            )
            table_ref = section.get("table_ref", "") or self._infer_table_ref(focused_text)
            citation = self._format_output_citation(
                raw_citation,
                regulation_type,
                page_ref,
                table_ref,
                focused_text,
                subsection_hint=subsection_hint,
            )

            text_lower = focused_text.lower()
            citation_lower = raw_citation.lower()
            lexical_hits = sum(1 for term in terms if term in text_lower or term in citation_lower)
            lexical_ratio = lexical_hits / max(len(terms), 1)
            phrase_hits = sum(1 for phrase in phrases if phrase in text_lower)
            required_ok = all(pattern.search(focused_text) or pattern.search(citation) for pattern in required_patterns)
            intent_hits = sum(1 for token in intent_tokens if token in text_lower)
            toc_penalty = self._table_of_contents_penalty(canonical_text)
            numeric_evidence = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|ft|m|kts|kt|hpa)\b", text_lower))
            qnh_evidence = "qnh" in text_lower or "qnh" in citation_lower
            subsection = self._citation_subsection_label(citation)
            explicit_subsection_match = any(
                subsection == label or subsection.startswith(f"{label}.")
                for label in explicit_subsection_labels
            )
            page_ref_lower = " ".join((page_ref or "").split()).lower()
            explicit_page_hint_match = any(page_ref_lower.startswith(hint) for hint in explicit_page_hints)

            if required_patterns and not required_ok:
                continue

            score = 0.22
            score += lexical_ratio * 0.52
            score += min(phrase_hits * 0.08, 0.2)
            if required_patterns and required_ok:
                score += 0.14
            if intent_hits:
                score += 0.06
            if numeric_intent:
                score += 0.06 if numeric_evidence else -0.08
            if qnh_intent:
                score += 0.14 if qnh_evidence else -0.2
                if subsection and subsection.startswith("5.3"):
                    score += 0.12
                if subsection == "1.4.1":
                    score += 0.14
            if explicit_subsection_labels:
                score += 0.24 if explicit_subsection_match else -0.24
            if explicit_page_hints:
                score += 0.32 if explicit_page_hint_match else -0.34
            if any(requested == citation_lower for requested in requested_citations):
                score += 0.2
            score -= toc_penalty

            if score < 0.42:
                continue

            reference = ReferenceItem(
                section_id=str(section.get("section_id", "")),
                regulation_id=str(section.get("regulation_id", raw_citation)),
                citation=citation,
                title=self._refine_title(str(section.get("title", "Untitled")), raw_citation, focused_text),
                regulation_type=regulation_type,
                source_file=str(section.get("source_file", "")),
                source_url=str(section.get("source_url", "")),
                text=focused_text,
                part=str(section.get("part", "")),
                page_ref=page_ref,
                table_ref=table_ref,
                section_index=int(section.get("section_order", 0) or 0),
                chunk_index=0,
                score=round(max(0.0, min(1.0, score)), 4),
            )
            ranked.append((reference.score, reference))

        if not ranked:
            return []

        references = self._dedupe_references(ranked, max(top_k * 3, top_k))
        return self._filter_final_references(references, query_profile, top_k)

    def _build_answer(self, query: str, top_reference: ReferenceItem) -> str:
        flattened = " ".join(top_reference.text.split())
        category = self._extract_aircraft_category(query)
        if self._query_targets_circling_minima(query) and category:
            circling = self._extract_circling_radius_data(flattened, category)
            radius_value = circling.get("radius_nm")
            elevation = circling.get("elevation_band")
            if radius_value:
                elevation_text = elevation or "0-1000 FT aerodrome elevation"
                return (
                    f"For Category {category} aircraft, circling radius is {radius_value} at {elevation_text} in {top_reference.citation}. "
                    "Use higher row values for higher aerodrome elevations."
                )

        sentence = self._extract_operational_sentence(flattened)
        return f"{top_reference.citation}: {sentence}"

    def _build_legal_explanation(self, query: str, references: list[ReferenceItem]) -> str:
        lead = references[0]
        query_lower = query.lower()
        if "qnh" in query_lower:
            lines: list[str] = [
                f"Controlling authority: {lead.citation}.",
                "Relevant extracted provisions:",
            ]
            for item in references[:5]:
                sentence = self._extract_operational_sentence(" ".join(item.text.split()))
                lines.append(f"- {item.citation}: {sentence}")
            return "\n".join(lines)
        lines = [
            f"For '{query}', controlling authority: {lead.citation}.",
            "Relevant extracted provisions:",
        ]
        for item in references[:5]:
            sentence = self._extract_operational_sentence(" ".join(item.text.split()))
            lines.append(f"- {item.citation}: {sentence}")
        return "\n".join(lines)

    def _build_plain_english(self, query: str, top_reference: ReferenceItem) -> str:
        flattened = " ".join(top_reference.text.split())
        query_lower = query.lower()
        if "qnh" in query_lower:
            qnh_sources = []
            for token in ("AAIS", "ATC", "ATIS", "AWIS", "CA/GRS", "WATIR"):
                if re.search(rf"\b{re.escape(token)}\b", flattened, re.IGNORECASE):
                    qnh_sources.append(token)
            if qnh_sources:
                return (
                    f"Use {top_reference.citation}. An accurate QNH must come from {', '.join(qnh_sources[:-1])}"
                    + (f", or {qnh_sources[-1]}" if len(qnh_sources) > 1 else qnh_sources[0])
                    + ". Forecast QNH in an authorised weather forecast is not valid for checking pressure-altitude system accuracy."
                )
            return (
                f"Use {top_reference.citation} first. For QNH questions, only use sources explicitly approved in the cited AIP text, "
                "then apply any related minima adjustments from linked QNH subsections."
            )
        category = self._extract_aircraft_category(query)
        if self._query_targets_circling_minima(query) and category:
            circling = self._extract_circling_radius_data(flattened, category)
            radius_value = circling.get("radius_nm")
            elevation = circling.get("elevation_band")
            if radius_value:
                elevation_text = elevation or "low-elevation aerodromes"
                return (
                    f"If you are flying Category {category}, use {radius_value} for {elevation_text} and use larger radii "
                    "from the same table as elevation increases."
                )

        sentence = self._extract_operational_sentence(flattened)
        return (
            f"Use {top_reference.citation} first. In plain terms: {sentence} "
            "Then confirm any limits and exceptions in the cited reference text."
        )

    def _build_example(self, query: str, top_reference: ReferenceItem) -> str:
        flattened = " ".join(top_reference.text.split())
        query_lower = query.lower()
        if "qnh" in query_lower:
            return (
                "Example: A pilot is flying IFR from Archerfield to Sunshine Coast and plans the RNP RWY 31 approach. "
                f"Before descending to DA, the pilot confirms QNH from an approved source under {top_reference.citation}. "
                "If only forecast QNH is available, the pilot does not treat it as valid for pressure-altitude accuracy checks and applies the appropriate minima logic from the cited QNH subsections."
            )
        category = self._extract_aircraft_category(query)
        if self._query_targets_circling_minima(query) and category:
            circling = self._extract_circling_radius_data(flattened, category)
            radius_value = circling.get("radius_nm")
            elevation = circling.get("elevation_band")
            if radius_value:
                elevation_text = elevation or "a low-elevation aerodrome"
                return (
                    f"Example: with a Category {category} aircraft circling at {elevation_text}, plan to stay within {radius_value}. "
                    f"If elevation is higher, move to the next higher Category {category} value in the same table."
                )

        query_clean = " ".join(query.split())
        return (
            f"If your question is '{query_clean}', open {top_reference.citation} and extract the exact number/condition "
            "from the cited text, then apply that value to the aircraft category and aerodrome conditions in your scenario."
        )

    def _build_contextual_notes(self, query: str, references: list[ReferenceItem]) -> list[str]:
        notes = [
            f"Primary citation selected: {references[0].citation}.",
            "References that did not pass lexical and intent relevance gates were excluded.",
            "Use the verbatim text as the controlling source rather than relying on paraphrase alone.",
        ]
        if len(references) > 1:
            notes.append(f"Cross-check {', '.join(item.citation for item in references[1:3])} for linked provisions.")
        if extract_citations(query):
            notes.append("The query included explicit citations; exact citation matches were prioritised.")
        return notes

    def _merge_references(
        self,
        primary: list[ReferenceItem],
        secondary: list[ReferenceItem],
        *,
        limit: int,
    ) -> list[ReferenceItem]:
        merged: list[ReferenceItem] = []
        seen: set[tuple[str, str]] = set()
        for item in [*primary, *secondary]:
            key = (item.citation.lower(), (item.page_ref or "").lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _citation_subsection_label(self, citation: str) -> str:
        match = re.search(r"\bsubsection\s+([0-9]+(?:\.[0-9]+){1,4})\b", citation, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"\bAIP\s+([0-9]+(?:\.[0-9]+){1,4})\b", citation, re.IGNORECASE)
        return match.group(1) if match else ""

    def _ensure_parent_subsection_reference(
        self,
        references: list[ReferenceItem],
        *,
        parent_label: str,
        limit: int,
    ) -> list[ReferenceItem]:
        if not references:
            return []

        has_parent = any(self._citation_subsection_label(item.citation) == parent_label for item in references)
        if has_parent:
            return references[:limit]

        child_prefix = f"{parent_label}."
        augmented = list(references)
        for index, item in enumerate(references):
            subsection = self._citation_subsection_label(item.citation)
            if not subsection.startswith(child_prefix):
                continue
            parent_citation = re.sub(
                rf"(\bsubsection\s+){re.escape(subsection)}\b",
                rf"\g<1>{parent_label}",
                item.citation,
                flags=re.IGNORECASE,
            )
            if parent_citation == item.citation:
                break
            parent_item = item.model_copy(
                update={
                    "citation": parent_citation,
                    "score": round(max(0.0, float(item.score) - 0.005), 4),
                }
            )
            augmented.insert(index, parent_item)
            break

        deduped: list[ReferenceItem] = []
        seen: set[str] = set()
        for item in augmented:
            key = item.citation.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _prioritize_weather_minima_references(self, references: list[ReferenceItem], top_k: int) -> list[ReferenceItem]:
        if not references:
            return []

        scored: list[tuple[float, ReferenceItem]] = []
        for item in references:
            text = f"{item.citation} {item.title} {item.text}".lower()
            subsection = self._citation_subsection_label(item.citation)
            score = float(item.score)
            if subsection == "6.2":
                score += 2.0
            elif subsection.startswith("6.2."):
                score += 1.2
            if "enr 1.5 - 39" in item.citation.lower():
                score += 0.8
            if re.search(r"\bspecial\s+alternate\s+weather\s+minima\b", text):
                score += 0.5
            if item.regulation_type.upper() != "AIP":
                score -= 1.2
            scored.append((score, item))

        ordered = [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]
        unique: list[ReferenceItem] = []
        seen_labels: set[str] = set()
        for item in ordered:
            label = self._citation_subsection_label(item.citation) or item.citation.lower()
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique.append(item)
            if len(unique) >= top_k:
                break
        return unique

    def _prioritize_qnh_references(
        self,
        references: list[ReferenceItem],
        query_profile: dict,
        top_k: int,
    ) -> list[ReferenceItem]:
        if not references:
            return []
        if not query_profile.get("qnh_intent"):
            return references[:top_k]

        query_lower = str(query_profile.get("query_lower", "") or "")
        wants_sources = any(token in query_lower for token in ("source", "sources", "accurate"))
        scored: list[tuple[float, ReferenceItem]] = []
        for item in references:
            text = f"{item.citation} {item.title} {item.text}".lower()
            subsection = self._citation_subsection_label(item.citation)
            qnh_hits = text.count("qnh")
            score = float(item.score)
            if subsection == "1.4.1":
                score += 1.8 if wants_sources else 0.9
            elif subsection == "1.4":
                score += 0.8
            if subsection.startswith("5.3"):
                score += 1.6
            if subsection in QNH_PRIORITY_SUBSECTIONS:
                score += 0.8
            if qnh_hits:
                score += min(0.35, qnh_hits * 0.06)
            scored.append((score, item))

        ordered = [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]
        unique: list[ReferenceItem] = []
        seen_labels: set[str] = set()
        for item in ordered:
            label = self._citation_subsection_label(item.citation) or item.citation.lower()
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique.append(item)

        has_primary_source = any(self._citation_subsection_label(item.citation) == "1.4.1" for item in unique)
        has_qnh_sources_block = any(self._citation_subsection_label(item.citation).startswith("5.3") for item in unique)
        if has_primary_source and not has_qnh_sources_block:
            for item in ordered:
                if self._citation_subsection_label(item.citation).startswith("5.3"):
                    unique.append(item)
                    break

        return unique[:top_k]

    def _dedupe_references(
        self,
        ranked_items: list[tuple[float, ReferenceItem]],
        top_k: int,
    ) -> list[ReferenceItem]:
        references: list[ReferenceItem] = []
        seen_keys: set[tuple[str, str]] = set()
        for _, item in sorted(ranked_items, key=lambda pair: pair[0], reverse=True):
            key = (item.citation.lower(), (item.page_ref or "").lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            references.append(item)
            if len(references) >= top_k:
                break
        return references

    def _filter_final_references(
        self,
        references: list[ReferenceItem],
        query_profile: dict,
        top_k: int,
    ) -> list[ReferenceItem]:
        if not references:
            return []

        top = references[0]
        if query_profile.get("strict_single_reference"):
            return [top]

        filtered: list[ReferenceItem] = [top]
        if query_profile.get("qnh_intent"):
            score_floor = max(0.42, top.score * 0.55)
        else:
            score_floor = max(0.55, top.score * 0.72)

        for item in references[1:]:
            if item.score < score_floor:
                continue
            if not self._is_reference_relevant(item, query_profile):
                continue
            filtered.append(item)
            if len(filtered) >= top_k:
                break
        return filtered

    def _is_reference_relevant(self, item: ReferenceItem, query_profile: dict) -> bool:
        text = f"{item.citation} {item.title} {item.text}".lower()
        required_patterns = query_profile.get("required_patterns", [])
        intent_tokens = query_profile.get("intent_tokens", [])
        terms = query_profile.get("terms", [])
        qnh_intent = bool(query_profile.get("qnh_intent"))

        if required_patterns and not all(pattern.search(text) for pattern in required_patterns):
            return False

        if query_profile.get("aip_preferred_intent") and item.regulation_type.upper() != "AIP":
            return False

        if qnh_intent and "qnh" not in text:
            return False

        if intent_tokens:
            hits = sum(1 for token in intent_tokens if token in text)
            if hits == 0:
                return False

        key_terms = [term for term in terms if term not in {"aircraft", "pilot", "flight", "category", "cat"}]
        if key_terms:
            overlap = sum(1 for term in key_terms if term in text)
            if overlap == 0:
                return False

        return True

    def _query_targets_circling_minima(self, query: str) -> bool:
        query_lower = query.lower()
        has_circling = "circling" in query_lower
        has_measure = any(term in query_lower for term in ("radius", "radii", "minima", "minimum"))
        return has_circling and has_measure

    def _extract_aircraft_category(self, query: str) -> str:
        match = re.search(r"\bcat(?:egory)?\s*([abcd])\b", query, re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).upper()

    def _extract_circling_radius_data(self, text: str, category: str) -> dict[str, str]:
        category_key = category.lower()
        category_column = {"a": "a", "b": "b", "c": "c", "d": "d"}.get(category_key)
        if not category_column:
            return {}

        rows = list(CIRCLING_ROW_PATTERN.finditer(text))
        if not rows:
            return {}

        first_row = rows[0]
        value = first_row.group(category_column)
        from_ft = first_row.group("from")
        to_ft = first_row.group("to")
        return {
            "radius_nm": f"{value}NM",
            "elevation_band": f"{from_ft}-{to_ft} FT aerodrome elevation",
        }

    def _extract_aip_subsection_blocks(self, text: str) -> list[tuple[str, str]]:
        matches = list(AIP_SUBSECTION_LINE_PATTERN.finditer(text[:20000]))
        if not matches:
            return []

        blocks: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            label = match.group("label")
            heading = match.group("heading").strip()
            if not re.search(r"[A-Za-z]{2,}", heading):
                continue
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if len(block) < 20:
                continue
            blocks.append((label, block))
        return blocks

    def _expand_aip_subsection_block(self, blocks: list[tuple[str, str]], label: str) -> str:
        if not blocks or not label:
            return ""

        by_index = {value[0]: idx for idx, value in enumerate(blocks)}
        if label not in by_index:
            return ""

        start_index = by_index[label]
        prefix = f"{label}."
        expanded: list[str] = []
        for idx in range(start_index, len(blocks)):
            current_label, current_block = blocks[idx]
            if idx == start_index:
                expanded.append(current_block)
                continue
            if current_label.startswith(prefix):
                expanded.append(current_block)
                continue
            break
        return "\n".join(expanded).strip()

    def _select_best_aip_subsection(self, text: str, citation: str, query_profile: dict) -> tuple[str, str]:
        blocks = self._extract_aip_subsection_blocks(text)
        if not blocks:
            return "", ""

        by_label = {label: block for label, block in blocks}
        query_lower = str(query_profile.get("query_lower", "") or "")
        explicit_labels = query_profile.get("explicit_subsection_labels") or SUBSECTION_PATTERN.findall(query_lower)
        explicit_labels = list(dict.fromkeys(explicit_labels))
        explicit_labels.sort(key=lambda label: (label.count("."), len(label)), reverse=True)
        for label in explicit_labels:
            if label in by_label:
                return label, by_label[label]

        if self._query_targets_circling_minima(query_lower):
            category = self._extract_aircraft_category(query_lower)
            if category:
                for label, block in blocks:
                    flattened = " ".join(block.split())
                    circling = self._extract_circling_radius_data(flattened, category)
                    if circling.get("radius_nm"):
                        expanded = self._expand_aip_subsection_block(blocks, label)
                        return label, expanded or block

        if query_profile.get("weather_minima_intent"):
            preferred_labels = [label for label in WEATHER_MINIMA_PRIORITY_SUBSECTIONS if label in by_label]
            for label in preferred_labels:
                block = by_label[label]
                if re.search(r"\b(?:special\s+)?alternate\s+weather\s+minima\b", block.lower()):
                    expanded = self._expand_aip_subsection_block(blocks, label)
                    return label, expanded or block
            for label, block in blocks:
                block_lower = block.lower()
                if re.search(r"\b(?:special\s+)?alternate\s+weather\s+minima\b", block_lower):
                    expanded = self._expand_aip_subsection_block(blocks, label)
                    return label, expanded or block
        if query_profile.get("qnh_intent"):
            preferred_labels = [label for label in QNH_PRIORITY_SUBSECTIONS if label in by_label]
            for label in preferred_labels:
                block = by_label[label]
                heading_line = (block.splitlines()[0] if block.splitlines() else "").lower()
                if "qnh" in heading_line or "source" in heading_line:
                    expanded = self._expand_aip_subsection_block(blocks, label)
                    return label, expanded or block
            for label, block in blocks:
                heading_line = (block.splitlines()[0] if block.splitlines() else "").lower()
                if "qnh" in heading_line:
                    expanded = self._expand_aip_subsection_block(blocks, label)
                    return label, expanded or block

        citation_match = re.search(r"\bAIP\s+([1-9]\d?(?:\.\d+){1,4})\b", citation, re.IGNORECASE)
        if citation_match:
            cited_label = citation_match.group(1)
            if cited_label in by_label:
                default_label = cited_label
            else:
                default_label = ""
        else:
            default_label = ""

        terms = query_profile.get("terms", [])
        phrases = query_profile.get("phrases", [])
        required_patterns = query_profile.get("required_patterns", [])
        intent_tokens = query_profile.get("intent_tokens", [])
        explicit_label_set = set(explicit_labels)

        best_label = default_label
        best_block = by_label.get(default_label, "")
        best_score = -1.0

        for label, block in blocks:
            lower = block.lower()
            heading_line = (block.splitlines()[0] if block.splitlines() else "").lower()
            lexical_hits = sum(1 for term in terms if term in lower)
            phrase_hits = sum(1 for phrase in phrases if phrase in lower)
            heading_hits = sum(1 for term in terms if term in heading_line)
            heading_phrase_hits = sum(1 for phrase in phrases if phrase in heading_line)
            required_hits = sum(1 for pattern in required_patterns if pattern.search(block))
            intent_hits = sum(1 for token in intent_tokens if token in lower)
            hierarchy_bonus = max(0.0, 1.2 - (label.count(".") * 0.35))
            label_bonus = 2.0 if label in explicit_label_set else 0.0
            score = (
                (lexical_hits * 0.35)
                + (phrase_hits * 0.55)
                + (heading_hits * 2.2)
                + (heading_phrase_hits * 2.8)
                + (required_hits * 1.4)
                + (intent_hits * 0.8)
                + hierarchy_bonus
                + label_bonus
            )
            if score > best_score:
                best_score = score
                best_label = label
                best_block = block

        if best_score <= 0 and default_label:
            expanded = self._expand_aip_subsection_block(blocks, default_label)
            return default_label, expanded or by_label.get(default_label, "")
        if best_score <= 0:
            return "", ""
        expanded = self._expand_aip_subsection_block(blocks, best_label)
        return best_label, expanded or best_block

    def _table_of_contents_penalty(self, text: str) -> float:
        scan = text[:30000]
        dot_leader_hits = len(re.findall(r"\.{5,}\s*(?:GEN|ENR|AD|AIP)\s+\d", scan, flags=re.IGNORECASE))
        heading_hits = len(
            re.findall(
                r"(?m)^[0-9]+(?:\.[0-9]+){0,4}\s+[A-Z][A-Za-z /,&()-]{4,90}\.{4,}",
                scan,
            )
        )
        if dot_leader_hits == 0 and heading_hits == 0:
            return 0.0
        raw = (dot_leader_hits * 0.045) + (heading_hits * 0.03)
        return min(0.4, raw)

    def _infer_page_ref(self, text: str, citation: str = "", subsection_hint: str = "") -> str:
        scan = text[:12000]
        matches = list(PAGE_REF_PATTERN.finditer(scan))
        if not matches:
            return ""
        subsection = subsection_hint or self._infer_subsection(citation, scan)
        if subsection:
            subsection_match = re.search(rf"\b{re.escape(subsection)}\b", scan)
            if subsection_match:
                preceding = [m for m in matches if m.start() <= subsection_match.start()]
                if preceding:
                    return " ".join(preceding[-1].group(0).split())
                following = [m for m in matches if m.start() > subsection_match.start()]
                if following:
                    return " ".join(following[0].group(0).split())
        return self._select_best_page_ref([" ".join(match.group(0).split()) for match in matches])

    def _resolve_page_ref(
        self,
        canonical_text: str,
        raw_citation: str,
        canonical_page_ref: str,
        subsection_hint: str,
    ) -> str:
        canonical_page = " ".join((canonical_page_ref or "").split())
        if not subsection_hint:
            return canonical_page or self._infer_page_ref(canonical_text, raw_citation)

        inferred = self._infer_page_ref(canonical_text, raw_citation, subsection_hint=subsection_hint)
        if not canonical_page:
            return inferred
        if not inferred:
            return canonical_page
        if inferred.lower() == canonical_page.lower():
            return canonical_page

        scan = canonical_text[:12000]
        subsection_match = re.search(rf"\b{re.escape(subsection_hint)}\b", scan)
        markers = list(PAGE_REF_PATTERN.finditer(scan))
        if subsection_match and markers:
            first_marker_start = markers[0].start()
            if subsection_match.start() < first_marker_start:
                return canonical_page
        return inferred

    def _infer_table_ref(self, text: str) -> str:
        match = TABLE_REF_PATTERN.search(text[:12000])
        return " ".join(match.group(0).split()) if match else ""

    def _infer_subsection(self, citation: str, text: str) -> str:
        citation_match = re.search(r"\bAIP\s+(\d+(?:\.\d+)+)\b", citation, re.IGNORECASE)
        if citation_match:
            return citation_match.group(1)
        text_match = SUBSECTION_PATTERN.search(text[:800])
        if text_match:
            return text_match.group(1)
        return ""

    def _format_output_citation(
        self,
        citation: str,
        regulation_type: str,
        page_ref: str,
        table_ref: str,
        text: str,
        subsection_hint: str = "",
    ) -> str:
        normalized = " ".join((citation or "").split()) or "Unknown"
        if str(regulation_type or "").upper() != "AIP":
            return normalized

        subsection = subsection_hint or self._infer_subsection(normalized, text)
        parts: list[str] = []
        if page_ref:
            parts.append(f"AIP {page_ref}")
        else:
            parts.append("AIP")
        if subsection:
            parts.append(f"subsection {subsection}")
        elif normalized.upper().startswith("AIP "):
            parts.append(normalized[4:])

        out = " ".join(parts).strip()
        if table_ref:
            out = f"{out}. {table_ref}"
        return out

    def _select_best_page_ref(self, refs: list[str]) -> str:
        if not refs:
            return ""

        ordered_unique: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            normalized = " ".join(ref.split())
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered_unique.append(normalized)

        if len(ordered_unique) == 1:
            return ordered_unique[0]

        parsed: list[tuple[str, int, str]] = []
        for ref in ordered_unique:
            match = PAGE_REF_PARSE_PATTERN.match(ref)
            if not match:
                continue
            parsed.append((match.group("prefix").upper(), int(match.group("page")), ref))

        if not parsed:
            return ordered_unique[0]

        first_prefix = parsed[0][0]
        same_prefix = [item for item in parsed if item[0] == first_prefix]
        if len(same_prefix) < 2:
            return ordered_unique[0]

        best = min(same_prefix, key=lambda item: item[1])
        return best[2]

    def _extract_operational_sentence(self, text: str) -> str:
        candidates = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        for sentence in candidates:
            if len(sentence) < 40:
                continue
            lower = sentence.lower()
            if "aip australia" in lower and len(sentence) < 70:
                continue
            return sentence[:320]
        return text[:320].strip() or "No operative sentence available in the top reference."

    def _refine_title(self, title: str, citation: str, text: str) -> str:
        normalized = " ".join((title or "").split())
        if not normalized:
            normalized = "Untitled"

        generic_titles = {
            "untitled",
            "aip australia",
            "aip",
            "casr",
            "car",
            "cao",
            "mos",
            "caa",
        }
        noisy_titles = {
            "mos prior",
            "cao and",
            "car contains",
            "cao document",
        }
        if normalized.lower() not in generic_titles and normalized.lower() not in noisy_titles:
            return normalized[:160]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return citation

        for line in lines[:6]:
            compact = " ".join(line.split())
            if compact.lower().startswith(citation.lower()):
                remainder = compact[len(citation) :].strip(" .:-")
                return (f"{citation} {remainder}" if remainder else citation)[:160]

        if len(lines) > 1 and lines[0].lower().startswith("aip"):
            compact = " ".join(lines[1].split())
            match = re.match(r"^(?P<section>\d+(?:\.\d+)+)\s*(?P<heading>.*)$", compact)
            if match:
                heading = match.group("heading").strip(" .:-")
                if heading:
                    return f"AIP {match.group('section')} {heading}"[:160]
                return f"AIP {match.group('section')}"[:160]

        first_compact = " ".join(lines[0].split())
        match = re.match(r"^(?P<section>\d+(?:\.\d+)+)\s*(?P<heading>.*)$", first_compact)
        if match:
            heading = match.group("heading").strip(" .:-")
            if heading:
                return f"AIP {match.group('section')} {heading}"[:160]
            return f"AIP {match.group('section')}"[:160]

        return citation[:160]
