from __future__ import annotations

import threading

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import (
    get_catalog,
    get_embedding_service,
    get_indexer_service,
    get_search_service,
    get_vector_store,
)
from app.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.search_service import SearchService


settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    if settings.preload_model:
        thread = threading.Thread(target=get_embedding_service().load, daemon=True)
        thread.start()
    if settings.auto_index_on_startup and get_vector_store().count() == 0:
        thread = threading.Thread(target=_run_background_index, daemon=True)
        thread.start()


def _run_background_index() -> None:
    try:
        get_indexer_service().rebuild_index()
    except Exception as exc:
        print(f"Background indexing failed: {exc}")


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    vector_store = get_vector_store()
    embeddings = get_embedding_service()
    catalog = get_catalog()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        embeddings_loaded=embeddings.is_loaded,
        collection_count=vector_store.count(),
        manifest_source=catalog.source_label(),
        vector_store_status="ready" if vector_store.available else f"unavailable: {vector_store.error}",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return root()


@app.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    vector_store = get_vector_store()
    if not vector_store.available:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable: {vector_store.error}. Delete chroma_db and rebuild the index.",
        )
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Regulation index is empty. Run the indexing command before using search.",
        )
    try:
        return search_service.search(query, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}") from exc
