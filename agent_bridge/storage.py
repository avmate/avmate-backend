from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    title: str
    project_root: str
    system_prompt: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MessageRecord:
    id: int
    session_id: str
    role: str
    provider: str | None
    content: str
    created_at: str
    metadata: dict[str, Any]


class BridgeStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'candidate')),
                    provider TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, id);
                """
            )
            conn.commit()

    def create_session(self, *, title: str, project_root: str, system_prompt: str) -> SessionRecord:
        session = SessionRecord(
            session_id=str(uuid.uuid4()),
            title=title,
            project_root=project_root,
            system_prompt=system_prompt,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, project_root, system_prompt, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.project_root,
                    session.system_prompt,
                    session.created_at,
                    session.updated_at,
                ),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> SessionRecord:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT session_id, title, project_root, system_prompt, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return SessionRecord(**dict(row))

    def list_sessions(self) -> list[SessionRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT session_id, title, project_root, system_prompt, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [SessionRecord(**dict(row)) for row in rows]

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageRecord:
        created_at = _utc_now()
        payload = json.dumps(metadata or {}, ensure_ascii=True)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, provider, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, provider, content, created_at, payload),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
            conn.commit()
            message_id = int(cursor.lastrowid)
        return MessageRecord(
            id=message_id,
            session_id=session_id,
            role=role,
            provider=provider,
            content=content,
            created_at=created_at,
            metadata=metadata or {},
        )

    def list_messages(self, session_id: str, *, include_candidates: bool = True) -> list[MessageRecord]:
        query = (
            """
            SELECT id, session_id, role, provider, content, created_at, metadata_json
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """
            if include_candidates
            else """
            SELECT id, session_id, role, provider, content, created_at, metadata_json
            FROM messages
            WHERE session_id = ? AND role != 'candidate'
            ORDER BY id ASC
            """
        )
        with closing(self._connect()) as conn:
            rows = conn.execute(query, (session_id,)).fetchall()

        messages: list[MessageRecord] = []
        for row in rows:
            data = dict(row)
            metadata = json.loads(data.pop("metadata_json") or "{}")
            messages.append(MessageRecord(metadata=metadata, **data))
        return messages

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        messages = self.list_messages(session_id, include_candidates=True)
        return {
            "session": asdict(session),
            "messages": [asdict(message) for message in messages],
        }
