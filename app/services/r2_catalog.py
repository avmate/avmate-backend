from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

from app.config import load_manifest_file


class RegulationCatalog:
    def __init__(self, base_url: str, manifest_path: Path, manifest_url: str | None, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._manifest_path = manifest_path
        self._manifest_url = manifest_url
        self._timeout_seconds = timeout_seconds

    def load_documents(self) -> list[dict]:
        if self._manifest_url:
            response = requests.get(self._manifest_url, timeout=self._timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("Remote manifest must return a JSON list.")
            return [self._normalize_item(item) for item in data]
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found at {self._manifest_path}. "
                "Create it or set R2_MANIFEST_URL."
            )
        return [self._normalize_item(item) for item in load_manifest_file(self._manifest_path)]

    def source_label(self) -> str:
        return self._manifest_url or str(self._manifest_path)

    def build_file_url(self, path: str) -> str:
        encoded_path = quote(path.lstrip("/"), safe="/")
        return f"{self._base_url}/{encoded_path}"

    def iter_download_targets(self, documents: Iterable[dict]) -> Iterable[tuple[dict, str]]:
        for document in documents:
            yield document, self.build_file_url(document["path"])

    def _normalize_item(self, item: dict) -> dict:
        path = item.get("path")
        if not path:
            raise ValueError("Each manifest record must include 'path'.")
        normalized = dict(item)
        normalized["path"] = path.replace("\\", "/")
        normalized.setdefault("type", normalized["path"].split("/", 1)[0])
        normalized.setdefault("title", Path(normalized["path"]).stem.replace("_", " "))
        normalized.setdefault("source_url", self.build_file_url(normalized["path"]))
        return normalized
