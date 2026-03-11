from __future__ import annotations

__all__ = [
    "CanonicalStore",
    "EmbeddingService",
    "SlidingWindowRateLimiter",
    "LLMAnswerService",
    "RegulationCatalog",
    "SearchService",
    "StudyGuideService",
    "VectorStore",
]


def __getattr__(name: str):
    if name == "CanonicalStore":
        from .canonical_store import CanonicalStore

        return CanonicalStore
    if name == "EmbeddingService":
        from .embedding_service import EmbeddingService

        return EmbeddingService
    if name == "SlidingWindowRateLimiter":
        from .rate_limiter import SlidingWindowRateLimiter

        return SlidingWindowRateLimiter
    if name == "LLMAnswerService":
        from .llm_answer_service import LLMAnswerService

        return LLMAnswerService
    if name == "RegulationCatalog":
        from .r2_catalog import RegulationCatalog

        return RegulationCatalog
    if name == "SearchService":
        from .search_service import SearchService

        return SearchService
    if name == "StudyGuideService":
        from .study_guide_service import StudyGuideService

        return StudyGuideService
    if name == "VectorStore":
        from .vector_store import VectorStore

        return VectorStore
    raise AttributeError(name)
