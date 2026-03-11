from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AvMate API")
    app_version: str = os.getenv("APP_VERSION", "3.0.0")
    app_env: str = os.getenv("APP_ENV", "production")
    port: int = int(os.getenv("PORT", "8000"))
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'avmate.db').as_posix()}")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "avmate_regulations")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    preload_model: bool = _as_bool(os.getenv("PRELOAD_EMBEDDINGS"), default=False)
    auto_index_on_startup: bool = _as_bool(os.getenv("AUTO_INDEX_ON_STARTUP"), default=False)
    force_reindex_on_startup: bool = _as_bool(os.getenv("FORCE_REINDEX_ON_STARTUP"), default=False)
    r2_base_url: str = os.getenv(
        "R2_BASE_URL",
        "https://pub-a32237578ade418f9375e48bb3f1982a.r2.dev",
    ).rstrip("/")
    r2_manifest_url: str | None = os.getenv("R2_MANIFEST_URL")
    local_manifest_path: Path = Path(
        os.getenv("REGULATION_MANIFEST_PATH", str(BASE_DIR / "data" / "regulations_manifest.json"))
    )
    search_result_limit: int = int(os.getenv("SEARCH_RESULT_LIMIT", "5"))
    chunk_size_words: int = int(os.getenv("CHUNK_SIZE_WORDS", "220"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
    rate_limit_enabled: bool = _as_bool(os.getenv("RATE_LIMIT_ENABLED"), default=True)
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    cors_allow_origins: list[str] = field(
        default_factory=lambda: _as_csv_list(
            os.getenv("CORS_ALLOW_ORIGINS"),
            ["https://beta.avmate.com.au", "http://localhost:3000", "http://127.0.0.1:3000"],
        )
    )
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    enable_llm_answers: bool = _as_bool(os.getenv("ENABLE_LLM_ANSWERS"), default=False)
    llm_model: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1100"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def describe_manifest(self) -> str:
        if self.r2_manifest_url:
            return self.r2_manifest_url
        return str(self.local_manifest_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_manifest_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Manifest JSON must be a list of document records.")
    return data
