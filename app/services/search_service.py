from __future__ import annotations

from typing import Any

from app.schemas import ReferenceItem, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class SearchService:
    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def search(self, query: str, top_k: int) -> SearchResponse:
        query_embedding = self._embeddings.encode([query])
        results = self._vector_store.query(query_embeddings=query_embedding, top_k=top_k)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        references: list[ReferenceItem] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            references.append(
                ReferenceItem(
                    regulation_id=metadata.get("regulation_id", metadata.get("citation", "Unknown")),
                    citation=metadata.get("citation", "Unknown"),
                    title=metadata.get("title", "Untitled"),
                    source_file=metadata.get("source_file", ""),
                    source_url=metadata.get("source_url", ""),
                    text=document,
                    score=round(score, 4),
                )
            )

        if not references:
            return SearchResponse(
                answer="No matching regulation text was found in the current index.",
                plain_english="The regulation index is empty or does not contain a close match for that query yet.",
                example="Try a narrower query such as 'CASR 61.385 instrument rating requirements'.",
                study_questions=["Which exact regulation number is most relevant to your question?"],
                references=[],
                citations=[],
                confidence=0,
                explanation="Index the source regulations before using semantic search.",
            )

        top_reference = references[0]
        citations = []
        for item in references:
            if item.citation not in citations:
                citations.append(item.citation)

        answer = f"{top_reference.citation}: {top_reference.title}"
        plain_english = (
            "The highest ranked regulation chunk is shown first. Read the cited text carefully because this "
            "response is extracted from the indexed source material rather than generated legal advice."
        )
        example = (
            f"If a pilot asks '{query}', start with {top_reference.citation}, then confirm the exact rule text "
            "in the references tab before relying on it operationally."
        )
        study_questions = [
            f"What does {citations[0]} require in the exact wording of the regulation?",
            "Which conditions, exceptions, or definitions in the cited text could change the answer?",
            "What related regulation or MOS provision should be cross-checked before concluding?",
        ]
        confidence = max(0, min(99, int(references[0].score * 100)))

        return SearchResponse(
            answer=answer,
            plain_english=plain_english,
            example=example,
            study_questions=study_questions,
            references=references,
            citations=citations,
            confidence=confidence,
            explanation="Results are ranked by vector similarity against indexed regulation sections.",
        )

