from __future__ import annotations

import threading

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import init_db
from app.dependencies import (
    get_catalog,
    get_canonical_store,
    get_embedding_service,
    get_indexer_service,
    get_search_service,
    get_vector_store,
)
from app.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.search_service import SearchService


settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)
_indexing_lock = threading.Lock()
_indexing_in_progress = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    if settings.preload_model:
        thread = threading.Thread(target=get_embedding_service().load, daemon=True)
        thread.start()
    if settings.auto_index_on_startup and get_vector_store().count() == 0:
        _ensure_background_index()


def _run_background_index() -> None:
    global _indexing_in_progress
    try:
        result = get_indexer_service().rebuild_index()
        print(f"Background indexing complete: {result}")
    except Exception as exc:
        print(f"Background indexing failed: {exc}")
    finally:
        with _indexing_lock:
            _indexing_in_progress = False


def _ensure_background_index() -> None:
    global _indexing_in_progress
    with _indexing_lock:
        if _indexing_in_progress:
            return
        _indexing_in_progress = True
    print("Background indexing started.")
    thread = threading.Thread(target=_run_background_index, daemon=True)
    thread.start()


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    vector_store = get_vector_store()
    embeddings = get_embedding_service()
    catalog = get_catalog()
    if vector_store.available:
        vector_status = "indexing" if _indexing_in_progress else "ready"
    else:
        vector_status = f"unavailable: {vector_store.error}"
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        embeddings_loaded=embeddings.is_loaded,
        collection_count=vector_store.count(),
        manifest_source=catalog.source_label(),
        vector_store_status=vector_status,
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
        if settings.auto_index_on_startup:
            _ensure_background_index()
            detail = "Regulation index build in progress. Retry in 30-120 seconds."
        else:
            detail = "Regulation index is empty. Run the indexing command before using search."
        raise HTTPException(
            status_code=503,
            detail=detail,
        )
    embeddings = get_embedding_service()
    if not embeddings.is_loaded:
        embeddings.ensure_loading()
        detail = "Embedding model is warming up. Retry the search in a few seconds."
        if embeddings.load_error:
            detail = f"Embedding model failed to load: {embeddings.load_error}"
        raise HTTPException(status_code=503, detail=detail)
    try:
        return search_service.search(query, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}") from exc
