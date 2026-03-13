from __future__ import annotations

import hashlib
import re
from datetime import datetime

from sqlalchemy import func, or_, select, text, update

from app.db import init_db, session_scope
from app.models import Document, IngestionRun, RegulationSection

_FTS5_RESERVED = frozenset({"and", "or", "not"})


class CanonicalStore:
    def __init__(self) -> None:
        init_db()

    def begin_run(self, documents_seen: int) -> str:
        with session_scope() as session:
            run = IngestionRun(documents_seen=documents_seen)
            session.add(run)
            session.flush()
            return run.id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        documents_indexed: int,
        documents_failed: int,
        chunks_indexed: int,
        notes: str = "",
    ) -> None:
        with session_scope() as session:
            run = session.get(IngestionRun, run_id)
            if not run:
                return
            run.completed_at = datetime.utcnow()
            run.status = status
            run.documents_indexed = documents_indexed
            run.documents_failed = documents_failed
            run.chunks_indexed = chunks_indexed
            run.notes = notes

    def deactivate_all_documents(self) -> None:
        with session_scope() as session:
            session.execute(update(Document).values(is_active=False))

    def upsert_document(
        self,
        *,
        source_file: str,
        source_url: str,
        title: str,
        regulation_type: str,
        raw_bytes: bytes,
        page_count: int,
    ) -> str:
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        with session_scope() as session:
            document = session.scalar(select(Document).where(Document.source_file == source_file))
            if document is None:
                document = Document(
                    source_file=source_file,
                    source_url=source_url,
                    title=title,
                    regulation_type=regulation_type,
                    sha256=sha256,
                    page_count=page_count,
                    is_active=True,
                    last_ingested_at=datetime.utcnow(),
                )
                session.add(document)
            else:
                document.source_url = source_url
                document.title = title
                document.regulation_type = regulation_type
                document.sha256 = sha256
                document.page_count = page_count
                document.is_active = True
                document.last_ingested_at = datetime.utcnow()
            session.flush()
            return document.id

    def replace_sections(self, document_id: str, sections: list[dict]) -> list[dict]:
        with session_scope() as session:
            session.query(RegulationSection).filter(RegulationSection.document_id == document_id).delete()
            persisted: list[dict] = []
            for order_index, section in enumerate(sections):
                row = RegulationSection(
                    document_id=document_id,
                    regulation_id=section["regulation_id"],
                    citation=section["citation"],
                    part=section.get("part", ""),
                    section_label=section.get("section_label", ""),
                    title=section["title"],
                    text=section["text"],
                    page_ref=section.get("page_ref", ""),
                    table_ref=section.get("table_ref", ""),
                    source_file=section["source_file"],
                    source_url=section["source_url"],
                    regulation_type=section["regulation_type"],
                    section_order=order_index,
                )
                session.add(row)
                session.flush()
                persisted.append(
                    {
                        **section,
                        "section_id": row.id,
                        "section_order": order_index,
                    }
                )
            return persisted

    def get_sections_by_ids(self, section_ids: list[str]) -> list[dict]:
        if not section_ids:
            return []
        with session_scope() as session:
            rows = session.scalars(select(RegulationSection).where(RegulationSection.id.in_(section_ids))).all()
            by_id = {row.id: self._row_to_dict(row) for row in rows}
        return [by_id[section_id] for section_id in section_ids if section_id in by_id]

    def get_sections_by_citation_prefix(self, prefix: str, *, limit: int = 20) -> list[dict]:
        """Return sections whose citation starts with `prefix` (case-insensitive), ordered by citation.

        e.g. prefix='AIP ENR 1.5 6.2' → returns 'AIP ENR 1.5 6.2', 'AIP ENR 1.5 6.2.1', etc.
        """
        pattern = f"{prefix.lower()}%"
        with session_scope() as session:
            rows = session.scalars(
                select(RegulationSection)
                .where(func.lower(RegulationSection.citation).like(pattern))
                .order_by(RegulationSection.citation)
                .limit(limit)
            ).all()
        return [self._row_to_dict(row) for row in rows]

    def search_sections_by_terms(
        self,
        terms: list[str],
        *,
        limit: int = 120,
        regulation_type: str | None = None,
    ) -> list[dict]:
        normalized_terms = [term.strip().lower() for term in terms if len(term.strip()) >= 3]
        if not normalized_terms:
            return []

        like_conditions = []
        for term in normalized_terms:
            pattern = f"%{term}%"
            like_conditions.extend(
                [
                    func.lower(RegulationSection.text).like(pattern),
                    func.lower(RegulationSection.title).like(pattern),
                    func.lower(RegulationSection.citation).like(pattern),
                ]
            )

        stmt = select(RegulationSection).where(or_(*like_conditions))
        if regulation_type:
            stmt = stmt.where(func.lower(RegulationSection.regulation_type) == regulation_type.lower())

        with session_scope() as session:
            rows = session.scalars(stmt.limit(limit)).all()
        return [self._row_to_dict(row) for row in rows]

    def rebuild_fts_index(self) -> None:
        """Truncate and repopulate the FTS5 BM25 index from regulation_sections.

        Call once after a full reindex completes. Fast bulk INSERT via raw SQL.
        """
        with session_scope() as session:
            session.execute(text("DELETE FROM regulation_sections_fts"))
            session.execute(text("""
                INSERT INTO regulation_sections_fts (section_id, regulation_type, citation, title, text)
                SELECT id, regulation_type, citation, title, text
                FROM regulation_sections
            """))

    def search_sections_bm25(
        self,
        query: str,
        *,
        limit: int = 40,
        regulation_type: str | None = None,
    ) -> list[tuple[str, float]]:
        """BM25-ranked full-text search via SQLite FTS5.

        Returns list of (section_id, normalized_score) where score is in [0.0, 0.8].
        bm25() returns negative values (lower = better); we invert to positive ascending.
        """
        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9.]+", query)
            if len(t) >= 2 and t.lower() not in _FTS5_RESERVED
        ]
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)

        try:
            with session_scope() as session:
                params: dict = {"q": fts_query, "lim": limit}
                if regulation_type:
                    params["rt"] = regulation_type
                    rows = session.execute(
                        text("""
                            SELECT section_id, bm25(regulation_sections_fts) AS score
                            FROM regulation_sections_fts
                            WHERE regulation_sections_fts MATCH :q
                              AND regulation_type = :rt
                            ORDER BY score
                            LIMIT :lim
                        """),
                        params,
                    ).fetchall()
                else:
                    rows = session.execute(
                        text("""
                            SELECT section_id, bm25(regulation_sections_fts) AS score
                            FROM regulation_sections_fts
                            WHERE regulation_sections_fts MATCH :q
                            ORDER BY score
                            LIMIT :lim
                        """),
                        params,
                    ).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        # bm25() returns negative values: min (most negative) = best match
        # Normalize to [0.0, 0.8]: best → 0.8, worst → 0.0
        raw_scores = [row[1] for row in rows]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max_s - min_s or 1.0

        return [
            (row[0], 0.8 * (1.0 - (row[1] - min_s) / score_range))
            for row in rows
        ]

    def _row_to_dict(self, row: RegulationSection) -> dict:
        return {
            "section_id": row.id,
            "regulation_id": row.regulation_id,
            "citation": row.citation,
            "part": row.part,
            "section_label": row.section_label,
            "title": row.title,
            "text": row.text,
            "page_ref": row.page_ref,
            "table_ref": row.table_ref,
            "source_file": row.source_file,
            "source_url": row.source_url,
            "regulation_type": row.regulation_type,
            "section_order": row.section_order,
        }
