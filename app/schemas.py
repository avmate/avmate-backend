from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class ReferenceItem(BaseModel):
    section_id: str
    regulation_id: str
    citation: str
    title: str
    regulation_type: str
    source_file: str
    source_url: str
    text: str
    part: str
    page_ref: str
    table_ref: str
    section_index: int
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    answer: str
    legal_explanation: str
    plain_english: str
    example: str
    study_questions: List[str]
    study_answers: List[str]
    references: List[ReferenceItem]
    citations: List[str]
    verbatim_text: str
    contextual_notes: List[str]
    confidence: int
    request_id: str = ""
    explanation: Optional[str] = None


class StudyGuideRequest(BaseModel):
    test_name: str = Field(..., min_length=3, max_length=200)
    max_items: int = Field(default=20, ge=5, le=60)


class StudyGuideItem(BaseModel):
    order: int
    section: str
    form_reference: str
    criterion: str
    regulation_reference: str
    regulation_confidence: int


class StudyGuideResponse(BaseModel):
    test_name: str
    form_title: str
    form_page_url: str
    form_download_url: str
    chronological_items: List[StudyGuideItem]
    notes: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    embeddings_loaded: bool
    collection_count: int
    manifest_source: str
    vector_store_status: str
    request_id: str = ""


class ReadyResponse(BaseModel):
    ready: bool
    service: str
    version: str
    embeddings_loaded: bool
    collection_count: int
    vector_store_status: str
    reason: str = ""
    request_id: str = ""
