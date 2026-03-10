from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class ReferenceItem(BaseModel):
    regulation_id: str
    citation: str
    title: str
    regulation_type: str
    source_file: str
    source_url: str
    text: str
    part: str
    section_index: int
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    answer: str
    legal_explanation: str
    plain_english: str
    example: str
    study_questions: List[str]
    references: List[ReferenceItem]
    citations: List[str]
    verbatim_text: str
    contextual_notes: List[str]
    confidence: int
    explanation: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    embeddings_loaded: bool
    collection_count: int
    manifest_source: str
    vector_store_status: str
