from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_bridge_settings
from .service import create_default_service


settings = get_bridge_settings()
service = None
service_error: str | None = None
try:
    service = create_default_service(settings)
except Exception as exc:  # pragma: no cover - exercised only when env vars are missing
    service_error = str(exc)

app = FastAPI(title="AvMate Agent Bridge", version="0.2.0")


class CreateSessionRequest(BaseModel):
    title: str | None = None
    project_root: str | None = None
    system_prompt: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    title: str
    project_root: str
    created_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    include_workspace_snapshot: bool = True


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    candidates: dict[str, str]
    warnings: list[str]
    workspace_snapshot: str | None = None


class TaskResponse(BaseModel):
    session_id: str
    execution_brief: str
    proposal: dict[str, object]
    review: dict[str, object]
    warnings: list[str]
    workspace_snapshot: str | None = None



def _require_service():
    if service is None:
        raise HTTPException(status_code=503, detail=service_error or "Agent bridge is not configured.")
    return service


@app.get("/health")
def health() -> dict[str, object]:
    active_service = service
    return {
        "ok": service_error is None,
        "providers": sorted(active_service.providers.keys()) if active_service else [],
        "db_path": str(active_service.store.db_path) if active_service else str(settings.db_path),
        "error": service_error,
    }


@app.get("/sessions")
def list_sessions() -> list[dict[str, object]]:
    active_service = _require_service()
    return [asdict(item) for item in active_service.list_sessions()]


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    active_service = _require_service()
    session = active_service.create_session(
        title=request.title or settings.default_title,
        project_root=request.project_root or str(settings.default_project_root),
        system_prompt=request.system_prompt or settings.default_system_prompt,
    )
    return CreateSessionResponse(
        session_id=session.session_id,
        title=session.title,
        project_root=session.project_root,
        created_at=session.created_at,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    active_service = _require_service()
    try:
        session = active_service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(session)


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, include_candidates: bool = True) -> list[dict[str, object]]:
    active_service = _require_service()
    try:
        messages = active_service.list_messages(session_id, include_candidates=include_candidates)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [asdict(item) for item in messages]


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    active_service = _require_service()
    try:
        turn = active_service.chat(
            session_id=session_id,
            user_message=request.message,
            include_workspace_snapshot=request.include_workspace_snapshot,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(
        session_id=session_id,
        answer=turn.answer,
        candidates=turn.candidates,
        warnings=turn.warnings,
        workspace_snapshot=turn.workspace_snapshot,
    )


@app.post("/sessions/{session_id}/task", response_model=TaskResponse)
def task(session_id: str, request: ChatRequest) -> TaskResponse:
    active_service = _require_service()
    try:
        turn = active_service.task(
            session_id=session_id,
            user_message=request.message,
            include_workspace_snapshot=request.include_workspace_snapshot,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaskResponse(
        session_id=session_id,
        execution_brief=turn.execution_brief,
        proposal=turn.proposal,
        review=turn.review,
        warnings=turn.warnings,
        workspace_snapshot=turn.workspace_snapshot,
    )
