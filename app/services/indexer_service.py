from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Settings
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
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._session = self._build_session()

    def rebuild_index(self) -> dict[str, int]:
        total_documents = 0
        total_chunks = 0
        failed_documents = 0
        documents = self._catalog.load_documents()
        self._vector_store.reset()

        for document in documents:
            try:
                text = self._extract_pdf_text(document["source_url"])
                if not text.strip():
                    failed_documents += 1
                    continue

                sections = split_into_sections(text)
                if not sections:
                    fallback_citation = Path(document["path"]).stem
                    sections = [
                        {
                            "regulation_id": fallback_citation,
                            "citation": fallback_citation,
                            "title": document["title"],
                            "part": "",
                            "text": text,
                        }
                    ]

                ids: list[str] = []
                doc_chunks: list[str] = []
                metadatas: list[dict[str, Any]] = []

                for section_index, section in enumerate(sections):
                    chunks = list(
                        chunk_words(
                            section["text"],
                            chunk_size=self._settings.chunk_size_words,
                            overlap=self._settings.chunk_overlap_words,
                        )
                    )
                    for chunk_index, chunk in enumerate(chunks):
                        ids.append(f"{document['path']}::{section_index}::{chunk_index}")
                        doc_chunks.append(chunk)
                        metadatas.append(
                            {
                                "regulation_id": section["regulation_id"],
                                "citation": section["citation"],
                                "title": section["title"],
                                "part": section.get("part", ""),
                                "regulation_type": document.get("type", "UNKNOWN"),
                                "source_file": document["path"],
                                "source_url": document["source_url"],
                                "section_index": section_index,
                                "chunk_index": chunk_index,
                            }
                        )
            except Exception as exc:
                print(f"Skipping document {document['path']}: {exc}")
                failed_documents += 1
                continue

            embeddings = self._embeddings.encode(doc_chunks, batch_size=self._settings.embedding_batch_size)
            self._vector_store.upsert(ids=ids, embeddings=embeddings, documents=doc_chunks, metadatas=metadatas)
            total_documents += 1
            total_chunks += len(doc_chunks)

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

    def _extract_pdf_text(self, url: str) -> str:
        response = self._session.get(url, timeout=self._settings.request_timeout_seconds)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(response.content)

        try:
            text_parts: list[str] = []
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        finally:
            temp_path.unlink(missing_ok=True)
