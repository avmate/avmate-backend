from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Settings
from app.services.canonical_store import CanonicalStore
from app.services.embedding_service import EmbeddingService
from app.services.r2_catalog import RegulationCatalog
from app.services.section_parser import chunk_words, split_into_sections
from app.services.vector_store import VectorStore


class IndexerService:
    def __init__(
        self,
        settings: Settings,
        catalog: RegulationCatalog,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        canonical_store: CanonicalStore,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._canonical_store = canonical_store
        self._session = self._build_session()

    def rebuild_index(self) -> dict[str, int]:
        total_documents = 0
        total_chunks = 0
        failed_documents = 0
        upsert_batch_size = max(128, self._settings.embedding_batch_size * 8)
        documents = self._catalog.load_documents()
        run_id = self._canonical_store.begin_run(documents_seen=len(documents))
        self._canonical_store.deactivate_all_documents()
        self._vector_store.reset()

        try:
            for document in documents:
                try:
                    raw_bytes, page_count, text = self._extract_pdf_text(document["source_url"])
                    if not text.strip():
                        failed_documents += 1
                        continue

                    sections = split_into_sections(text, regulation_type=document.get("type", ""))
                    if not sections:
                        fallback_citation = Path(document["path"]).stem
                        sections = [
                            {
                                "regulation_id": fallback_citation,
                                "citation": fallback_citation,
                                "title": document["title"],
                                "part": "",
                                "section_label": fallback_citation,
                                "page_ref": "",
                                "table_ref": "",
                                "text": text,
                            }
                        ]

                    document_id = self._canonical_store.upsert_document(
                        source_file=document["path"],
                        source_url=document["source_url"],
                        title=document["title"],
                        regulation_type=document.get("type", "UNKNOWN"),
                        raw_bytes=raw_bytes,
                        page_count=page_count,
                    )

                    persisted_sections = self._canonical_store.replace_sections(
                        document_id,
                        [
                            {
                                **section,
                                "source_file": document["path"],
                                "source_url": document["source_url"],
                                "regulation_type": document.get("type", "UNKNOWN"),
                            }
                            for section in sections
                        ],
                    )

                    ids: list[str] = []
                    doc_chunks: list[str] = []
                    metadatas: list[dict[str, Any]] = []
                    for section_index, section in enumerate(persisted_sections):
                        chunks = list(
                            chunk_words(
                                section["text"],
                                chunk_size=self._settings.chunk_size_words,
                                overlap=self._settings.chunk_overlap_words,
                            )
                        )
                        for chunk_index, chunk in enumerate(chunks):
                            ids.append(f"{section['section_id']}::{chunk_index}")
                            doc_chunks.append(chunk)
                            metadatas.append(
                                {
                                    "section_id": section["section_id"],
                                    "citation": section["citation"],
                                    "regulation_type": document.get("type", "UNKNOWN"),
                                    "section_index": section_index,
                                    "chunk_index": chunk_index,
                                }
                            )
                    if not doc_chunks:
                        print(f"Skipping document {document['path']}: no chunks generated after parsing.")
                        failed_documents += 1
                        continue

                    for start in range(0, len(doc_chunks), upsert_batch_size):
                        end = start + upsert_batch_size
                        batch_chunks = doc_chunks[start:end]
                        batch_ids = ids[start:end]
                        batch_metadatas = metadatas[start:end]
                        batch_embeddings = self._embeddings.encode(
                            batch_chunks,
                            batch_size=self._settings.embedding_batch_size,
                        )
                        self._vector_store.upsert(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            documents=batch_chunks,
                            metadatas=batch_metadatas,
                        )

                    total_documents += 1
                    total_chunks += len(doc_chunks)
                except Exception as exc:
                    print(f"Skipping document {document['path']}: {exc}")
                    failed_documents += 1
                    continue
        finally:
            self._canonical_store.finish_run(
                run_id,
                status="completed" if failed_documents == 0 else "completed_with_errors",
                documents_indexed=total_documents,
                documents_failed=failed_documents,
                chunks_indexed=total_chunks,
            )

        return {
            "documents_indexed": total_documents,
            "chunks_indexed": total_chunks,
            "documents_failed": failed_documents,
        }

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _extract_pdf_text(self, url: str) -> tuple[bytes, int, str]:
        response = self._session.get(url, timeout=self._settings.request_timeout_seconds)
        response.raise_for_status()
        raw_bytes = response.content

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(raw_bytes)

        try:
            text_parts: list[str] = []
            page_count = 0
            with pdfplumber.open(temp_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
            return raw_bytes, page_count, "\n".join(text_parts)
        finally:
            temp_path.unlink(missing_ok=True)
