from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        where: dict | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    @property
    def available(self) -> bool:
        return True

    @property
    def error(self) -> str | None:
        return None


class UnavailableVectorStore:
    def __init__(self, error: Exception) -> None:
        self._error = str(error)

    def count(self) -> int:
        return 0

    def reset(self) -> None:
        raise RuntimeError(self._error)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        raise RuntimeError(self._error)

    def query(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        where: dict | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(self._error)

    @property
    def available(self) -> bool:
        return False

    @property
    def error(self) -> str:
        return self._error
