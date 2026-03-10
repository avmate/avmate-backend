from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.canonical_store import CanonicalStore
from app.services.embedding_service import EmbeddingService
from app.services.indexer_service import IndexerService
from app.services.r2_catalog import RegulationCatalog
from app.services.search_service import SearchService
from app.services.vector_store import UnavailableVectorStore, VectorStore


def get_vector_store() -> VectorStore | UnavailableVectorStore:
    settings = get_settings()
    try:
        return VectorStore(settings.chroma_dir, settings.collection_name)
    except Exception as exc:
        print(f"Vector store unavailable: {exc}")
        return UnavailableVectorStore(exc)


@lru_cache(maxsize=1)
def get_canonical_store() -> CanonicalStore:
    return CanonicalStore()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_catalog() -> RegulationCatalog:
    settings = get_settings()
    return RegulationCatalog(
        base_url=settings.r2_base_url,
        manifest_path=settings.local_manifest_path,
        manifest_url=settings.r2_manifest_url,
        timeout_seconds=settings.request_timeout_seconds,
    )


def get_search_service() -> SearchService:
    return SearchService(get_embedding_service(), get_vector_store(), get_canonical_store())


def get_indexer_service() -> IndexerService:
    settings: Settings = get_settings()
    return IndexerService(
        settings,
        get_catalog(),
        get_embedding_service(),
        get_vector_store(),
        get_canonical_store(),
    )
