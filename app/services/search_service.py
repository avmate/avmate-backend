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

PAGE_REF_PATTERN = re.compile(r"\b(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?\s*-\s*\(?\d+\)?\b", re.IGNORECASE)
TABLE_REF_PATTERN = re.compile(r"\bTable\s+\d+(?:\.\d+)+\b", re.IGNORECASE)
SUBSECTION_PATTERN = re.compile(r"\b(\d+(?:\.\d+)+)\b")
PAGE_REF_PARSE_PATTERN = re.compile(
    r"^(?P<prefix>(?:GEN|ENR|AD|AIP)\s+\d+(?:\.\d+)?)\s*-\s*\(?(?P<page>\d+)\)?$",
    re.IGNORECASE,
)
CIRCLING_ROW_PATTERN = re.compile(
    r"\b(?P<from>\d+)\s*FT\s+(?P<to>\d+)\s*FT\s+"
    r"(?P<a>[0-9.]+)\s*NM\s+(?P<b>[0-9.]+)\s*NM\s+(?P<c>[0-9.]+)\s*NM\s+(?P<d>[0-9.]+)\s*NM\b",
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
            candidate_k = min(max(top_k * 8, 24), 80)
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
            canonical_text = canonical.get("text", document)
            page_ref = canonical.get("page_ref", "") or self._infer_page_ref(canonical_text, raw_citation)
            table_ref = canonical.get("table_ref", "") or self._infer_table_ref(canonical_text)
            citation = self._format_output_citation(
                raw_citation,
                str(canonical.get("regulation_type", "UNKNOWN") or "UNKNOWN"),
                page_ref,
                table_ref,
                canonical_text,
            )
            combined_score, passes_gate = self._combine_score(
                query_profile=query_profile,
                document=canonical_text,
                citation=raw_citation,
                semantic_score=semantic_score,
                requested_citations=requested_citations,
            )
            reference = ReferenceItem(
                section_id=section_id,
                regulation_id=canonical.get("regulation_id", raw_citation),
                citation=citation,
                title=self._refine_title(canonical.get("title", "Untitled"), raw_citation, canonical_text),
                regulation_type=str(canonical.get("regulation_type", "UNKNOWN") or "UNKNOWN"),
                source_file=canonical.get("source_file", ""),
                source_url=canonical.get("source_url", ""),
                text=canonical_text,
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
            if len(token) > 2 and token not in STOP_WORDS
        ]
        phrases = [phrase for phrase in re.findall(r"\b[a-z0-9]+\s+[a-z0-9]+\b", query_lower) if len(phrase) > 7]

        required_patterns: list[re.Pattern[str]] = []
        category_match = re.search(r"\bcat(?:egory)?\s*([abcd])\b", query_lower)
        if category_match:
            cat = category_match.group(1)
            required_patterns.append(re.compile(rf"\bcat(?:egory)?\s*{cat}\b", re.IGNORECASE))
        if "circling" in query_lower:
            required_patterns.append(re.compile(r"\bcircling\b", re.IGNORECASE))
        if "radius" in query_lower or "radii" in query_lower or "minima" in query_lower or "minimum" in query_lower:
            required_patterns.append(re.compile(r"\b(?:radi(?:us|i)|minima|minimum)\b", re.IGNORECASE))

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
        return {
            "query_lower": query_lower,
            "terms": list(dict.fromkeys(terms)),
            "phrases": list(dict.fromkeys(phrases)),
            "required_patterns": required_patterns,
            "intent_tokens": intent_tokens,
            "numeric_intent": numeric_intent,
            "strict_single_reference": strict_single_reference,
        }

    def _combine_score(
        self,
        query_profile: dict,
        document: str,
        citation: str,
        semantic_score: float,
        requested_citations: list[str],
    ) -> tuple[float, bool]:
        query_lower = query_profile["query_lower"]
        terms = query_profile["terms"]
        phrases = query_profile["phrases"]
        required_patterns = query_profile["required_patterns"]
        intent_tokens = query_profile["intent_tokens"]
        numeric_intent = bool(query_profile.get("numeric_intent"))
        strict_single_reference = bool(query_profile.get("strict_single_reference"))

        citation_lower = citation.lower()
        document_lower = document.lower()

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

        passes_gate = citation_exact or citation_mentioned
        if not passes_gate:
            passes_gate = semantic_score >= 0.58 or lexical_ratio >= 0.24 or (semantic_score >= 0.5 and lexical_ratio >= 0.16)
            if required_patterns and not required_ok:
                passes_gate = False
            if numeric_intent and not numeric_evidence and semantic_score < 0.72:
                passes_gate = False
            if strict_single_reference and not circling_table_evidence and semantic_score < 0.8:
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

        return max(0.0, min(1.0, score)), passes_gate

    def _rescue_ranked_items(
        self,
        fallback_items: list[tuple[float, ReferenceItem]],
        query_profile: dict,
        requested_citations: list[str],
    ) -> list[tuple[float, ReferenceItem]]:
        rescued: list[tuple[float, ReferenceItem]] = []
        for semantic_score, item in sorted(fallback_items, key=lambda pair: pair[0], reverse=True):
            combined, _ = self._combine_score(
                query_profile=query_profile,
                document=item.text,
                citation=item.citation,
                semantic_score=semantic_score,
                requested_citations=requested_citations,
            )
            if semantic_score >= 0.5 or combined >= 0.45:
                item.score = round(combined, 4)
                rescued.append((combined, item))
            if len(rescued) >= 10:
                break
        return rescued

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
        cross_refs = ", ".join(item.citation for item in references[1:3]) if len(references) > 1 else "no immediate cross-reference"
        return (
            f"For '{query}', {lead.citation} is the controlling authority from the current indexed sources. "
            f"Cross-check {cross_refs} for linked conditions, definitions, or exceptions before operational use."
        )

    def _build_plain_english(self, query: str, top_reference: ReferenceItem) -> str:
        flattened = " ".join(top_reference.text.split())
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

        if required_patterns and not all(pattern.search(text) for pattern in required_patterns):
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

    def _infer_page_ref(self, text: str, citation: str = "") -> str:
        scan = text[:12000]
        matches = list(PAGE_REF_PATTERN.finditer(scan))
        if not matches:
            return ""
        subsection = self._infer_subsection(citation, scan)
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
    ) -> str:
        normalized = " ".join((citation or "").split()) or "Unknown"
        if str(regulation_type or "").upper() != "AIP":
            return normalized

        subsection = self._infer_subsection(normalized, text)
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
        if normalized.lower() not in generic_titles:
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

        return citation[:160]
