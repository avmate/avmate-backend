from __future__ import annotations

import re

from app.schemas import ReferenceItem, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.section_parser import extract_citations
from app.services.vector_store import VectorStore


class SearchService:
    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def search(self, query: str, top_k: int) -> SearchResponse:
        query_embedding = self._embeddings.encode([query])
        results = self._vector_store.query(query_embeddings=query_embedding, top_k=top_k)
        requested_citations = extract_citations(query)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        ranked_items: list[tuple[float, ReferenceItem]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            semantic_score = max(0.0, min(1.0, 1.0 - distance))
            citation = metadata.get("citation", "Unknown")
            combined_score = self._combine_score(query, document, citation, semantic_score, requested_citations)
            ranked_items.append(
                (
                    combined_score,
                    ReferenceItem(
                        regulation_id=metadata.get("regulation_id", citation),
                        citation=citation,
                        title=metadata.get("title", "Untitled"),
                        regulation_type=metadata.get("regulation_type", "UNKNOWN"),
                        source_file=metadata.get("source_file", ""),
                        source_url=metadata.get("source_url", ""),
                        text=document,
                        part=metadata.get("part", ""),
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
        ]
        contextual_notes = self._build_contextual_notes(query, references)
        confidence = max(0, min(99, int(references[0].score * 100)))

        return SearchResponse(
            answer=answer,
            legal_explanation=legal_explanation,
            plain_english=plain_english,
            example=example,
            study_questions=study_questions,
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
