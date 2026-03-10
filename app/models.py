from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    source_file: Mapped[str] = mapped_column(Text, unique=True, index=True)
    source_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    regulation_type: Mapped[str] = mapped_column(Text, index=True)
    sha256: Mapped[str] = mapped_column(Text, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sections: Mapped[list["RegulationSection"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class RegulationSection(Base):
    __tablename__ = "regulation_sections"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    regulation_id: Mapped[str] = mapped_column(Text, index=True)
    citation: Mapped[str] = mapped_column(Text, index=True)
    part: Mapped[str] = mapped_column(Text, default="", index=True)
    section_label: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    page_ref: Mapped[str] = mapped_column(Text, default="")
    table_ref: Mapped[str] = mapped_column(Text, default="")
    source_file: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    regulation_type: Mapped[str] = mapped_column(Text, index=True)
    section_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    document: Mapped[Document] = relationship(back_populates="sections")


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="running")
    documents_seen: Mapped[int] = mapped_column(Integer, default=0)
    documents_indexed: Mapped[int] = mapped_column(Integer, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_indexed: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
