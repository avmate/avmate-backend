from __future__ import annotations

import re

from app.schemas import ReferenceItem, SearchResponse
from app.services.canonical_store import CanonicalStore
from app.services.embedding_service import EmbeddingService
from app.services.section_parser import extract_citations
from app.services.vector_store import VectorStore


class SearchService:
    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore, canonical_store: CanonicalStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._canonical_store = canonical_store

    def search(self, query: str, top_k: int) -> SearchResponse:
        query_embedding = self._embeddings.encode([query])
        results = self._vector_store.query(query_embeddings=query_embedding, top_k=top_k)
        requested_citations = extract_citations(query)

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
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            semantic_score = max(0.0, min(1.0, 1.0 - distance))
            section_id = metadata.get("section_id", "")
            canonical = canonical_sections.get(section_id, {})
            citation = canonical.get("citation", metadata.get("citation", "Unknown"))
            combined_score = self._combine_score(query, document, citation, semantic_score, requested_citations)
            ranked_items.append(
                (
                    combined_score,
                    ReferenceItem(
                        section_id=section_id,
                        regulation_id=canonical.get("regulation_id", citation),
                        citation=citation,
                        title=self._refine_title(canonical.get("title", "Untitled"), citation, canonical.get("text", document)),
                        regulation_type=canonical.get("regulation_type", "UNKNOWN"),
                        source_file=canonical.get("source_file", ""),
                        source_url=canonical.get("source_url", ""),
                        text=canonical.get("text", document),
                        part=canonical.get("part", ""),
                        page_ref=canonical.get("page_ref", ""),
                        table_ref=canonical.get("table_ref", ""),
                        section_index=int(metadata.get("section_index", 0)),
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        score=round(combined_score, 4),
                    ),
                )
            )

        references = self._dedupe_references(ranked_items, top_k)

        if not references:
            return SearchResponse(
                answer="No matching regulation text was found in the current index.",
                legal_explanation="The current regulation index did not produce any section that could be cited responsibly.",
                plain_english="The regulation index is empty or does not contain a close match for that query yet.",
                example="Try a narrower query such as 'CASR 61.385 instrument rating requirements'.",
                study_questions=["Which exact regulation number is most relevant to your question?"],
                study_answers=["Identify the controlling citation first, then read the operative text before forming an answer."],
                references=[],
                citations=[],
                verbatim_text="",
                contextual_notes=["Run the indexer or add the missing regulation sources before relying on search output."],
                confidence=0,
                explanation="Index the source regulations before using semantic search.",
            )

        top_reference = references[0]
        citations = []
        for item in references:
            if item.citation not in citations:
                citations.append(item.citation)

        answer = self._build_answer(top_reference)
        legal_explanation = self._build_legal_explanation(query, references)
        plain_english = self._build_plain_english(top_reference)
        example = (
            f"If a pilot asks '{query}', start with {top_reference.citation}, then confirm the exact rule text "
            "in the references tab before relying on it operationally."
        )
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
        confidence = max(0, min(99, int(references[0].score * 100)))

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
            explanation="Results are ranked using vector similarity with additional boosts for explicit citation matches and query term overlap.",
        )

    def _combine_score(
        self,
        query: str,
        document: str,
        citation: str,
        semantic_score: float,
        requested_citations: list[str],
    ) -> float:
        score = semantic_score
        query_lower = query.lower()
        citation_lower = citation.lower()
        document_lower = document.lower()

        if any(requested.lower() == citation_lower for requested in requested_citations):
            score += 0.35
        elif citation_lower in query_lower:
            score += 0.25

        terms = [term for term in re.findall(r"[a-zA-Z0-9.()/-]+", query_lower) if len(term) > 3]
        overlap = sum(1 for term in set(terms) if term in document_lower)
        score += min(overlap * 0.02, 0.15)
        return max(0.0, min(1.0, score))

    def _build_answer(self, top_reference: ReferenceItem) -> str:
        return (
            f"Primary authority: {top_reference.citation}. "
            f"{top_reference.title}. Source type: {top_reference.regulation_type}."
        )

    def _build_legal_explanation(self, query: str, references: list[ReferenceItem]) -> str:
        lead = references[0]
        cross_refs = ", ".join(item.citation for item in references[1:3]) if len(references) > 1 else "no immediate cross-reference"
        return (
            f"For the query '{query}', the strongest indexed authority is {lead.citation}. "
            f"The returned text should be treated as the operative regulation extract, and any conclusion should be checked against {cross_refs} before operational use."
        )

    def _build_plain_english(self, top_reference: ReferenceItem) -> str:
        return (
            f"The search engine thinks {top_reference.citation} is the best match. "
            "Read that text first, then use the extra cited sections to confirm exceptions, definitions, and linked standards."
        )

    def _build_contextual_notes(self, query: str, references: list[ReferenceItem]) -> list[str]:
        notes = [
            f"Start research with {references[0].citation} because it has the highest combined citation and semantic match score.",
            "Use the verbatim text as the controlling source rather than relying on paraphrase alone.",
        ]
        if len(references) > 1:
            notes.append(
                f"Cross-check {', '.join(item.citation for item in references[1:3])} for definitions, conditions, or MOS links."
            )
        if extract_citations(query):
            notes.append("The query included an explicit citation, so exact citation matches were boosted in ranking.")
        return notes

    def _dedupe_references(
        self,
        ranked_items: list[tuple[float, ReferenceItem]],
        top_k: int,
    ) -> list[ReferenceItem]:
        references: list[ReferenceItem] = []
        seen_keys: set[tuple[str, int]] = set()
        for _, item in sorted(ranked_items, key=lambda pair: pair[0], reverse=True):
            key = (item.citation, item.section_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            references.append(item)
            if len(references) >= top_k:
                break
        return references

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

        # Prefer a line that starts with the citation itself.
        for line in lines[:6]:
            compact = " ".join(line.split())
            if compact.lower().startswith(citation.lower()):
                remainder = compact[len(citation) :].strip(" .:-")
                return (f"{citation} {remainder}" if remainder else citation)[:160]

        # AIP fallback from two-line headers.
        if len(lines) > 1 and lines[0].lower().startswith("aip"):
            compact = " ".join(lines[1].split())
            match = re.match(r"^(?P<section>\d+(?:\.\d+)+)\s*(?P<heading>.*)$", compact)
            if match:
                heading = match.group("heading").strip(" .:-")
                if heading:
                    return f"AIP {match.group('section')} {heading}"[:160]
                return f"AIP {match.group('section')}"[:160]

        return citation[:160]
