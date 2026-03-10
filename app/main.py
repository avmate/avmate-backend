from __future__ import annotations

import threading
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import init_db
from app.dependencies import (
    get_catalog,
    get_canonical_store,
    get_embedding_service,
    get_indexer_service,
    get_search_service,
    get_study_guide_service,
    get_vector_store,
)
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    StudyGuideRequest,
    StudyGuideResponse,
)
from app.services.rate_limiter import SlidingWindowRateLimiter
from app.services.search_service import SearchService
from app.services.study_guide_service import StudyGuideService


settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)
_indexing_lock = threading.Lock()
_indexing_in_progress = False
_rate_limiter = (
    SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if settings.rate_limit_enabled
    else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        print(f"request_id={request_id} method={request.method} path={request.url.path} error={exc} duration_ms={duration_ms}")
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    print(
        f"request_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms}"
    )
    return response


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


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _build_health_response(request_id: str) -> HealthResponse:
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
        request_id=request_id,
    )


@app.get("/", response_model=HealthResponse)
def root(request: Request) -> HealthResponse:
    return _build_health_response(_request_id(request))


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return _build_health_response(_request_id(request))


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request, response: Response) -> ReadyResponse:
    vector_store = get_vector_store()
    embeddings = get_embedding_service()
    if not embeddings.is_loaded:
        embeddings.ensure_loading()
    collection_count = vector_store.count() if vector_store.available else 0
    ready_state = vector_store.available and collection_count > 0 and embeddings.is_loaded
    if ready_state:
        reason = "ready"
        vector_status = "ready"
        response.status_code = 200
    else:
        if not vector_store.available:
            reason = f"vector_store_unavailable: {vector_store.error}"
            vector_status = "unavailable"
        elif collection_count == 0 and _indexing_in_progress:
            reason = "indexing_in_progress"
            vector_status = "indexing"
        elif collection_count == 0:
            reason = "index_empty"
            vector_status = "empty"
        else:
            reason = "embedding_warmup"
            vector_status = "warming"
        response.status_code = 503
    return ReadyResponse(
        ready=ready_state,
        service=settings.app_name,
        version=settings.app_version,
        embeddings_loaded=embeddings.is_loaded,
        collection_count=collection_count,
        vector_store_status=vector_status,
        reason=reason,
        request_id=_request_id(request),
    )


@app.post("/search", response_model=SearchResponse)
def search(
    request: Request,
    payload: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    request_id = _request_id(request)
    if _rate_limiter is not None:
        allowed, retry_after = _rate_limiter.allow(_client_key(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry in {retry_after} seconds. request_id={request_id}",
                headers={"Retry-After": str(retry_after)},
            )
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail=f"Query must not be empty. request_id={request_id}")
    vector_store = get_vector_store()
    if not vector_store.available:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable: {vector_store.error}. Delete chroma_db and rebuild the index. request_id={request_id}",
        )
    if vector_store.count() == 0:
        if settings.auto_index_on_startup:
            _ensure_background_index()
            detail = "Regulation index build in progress. Retry in 30-120 seconds."
        else:
            detail = "Regulation index is empty. Run the indexing command before using search."
        raise HTTPException(
            status_code=503,
            detail=f"{detail} request_id={request_id}",
        )
    embeddings = get_embedding_service()
    if not embeddings.is_loaded:
        embeddings.ensure_loading()
        detail = "Embedding model is warming up. Retry the search in a few seconds."
        if embeddings.load_error:
            detail = f"Embedding model failed to load: {embeddings.load_error}"
        raise HTTPException(status_code=503, detail=f"{detail} request_id={request_id}")
    try:
        result = search_service.search(query, payload.top_k)
        return result.model_copy(update={"request_id": request_id})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}. request_id={request_id}") from exc


@app.post("/study-guide", response_model=StudyGuideResponse)
def study_guide(
    request: Request,
    payload: StudyGuideRequest,
    study_service: StudyGuideService = Depends(get_study_guide_service),
) -> StudyGuideResponse:
    request_id = _request_id(request)
    test_name = payload.test_name.strip()
    if not test_name:
        raise HTTPException(status_code=422, detail=f"test_name must not be empty. request_id={request_id}")
    try:
        return study_service.build_study_guide(test_name=test_name, max_items=payload.max_items)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Study guide unavailable: {exc}. request_id={request_id}") from exc
