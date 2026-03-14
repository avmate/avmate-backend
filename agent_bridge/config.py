from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


DEFAULT_SYSTEM_PROMPT = (
    "You are part of a two-model engineering cell working on a shared codebase. "
    "Respond as a pragmatic software engineer. Be concise, factual, and specific. "
    "Do not claim to have run commands, inspected files, or validated behavior unless "
    "the provided conversation context explicitly says that happened. Prefer actionable "
    "next steps over abstract discussion."
)


@dataclass(frozen=True)
class BridgeSettings:
    db_path: Path = Path(
        os.getenv("AGENT_BRIDGE_DB_PATH", str(BASE_DIR / "agent_bridge.db"))
    )
    default_project_root: Path = Path(
        os.getenv("AGENT_BRIDGE_PROJECT_ROOT", str(BASE_DIR))
    )
    default_title: str = os.getenv(
        "AGENT_BRIDGE_DEFAULT_TITLE",
        "AvMate Shared Agent Session",
    )
    default_system_prompt: str = os.getenv(
        "AGENT_BRIDGE_SYSTEM_PROMPT",
        DEFAULT_SYSTEM_PROMPT,
    )
    openai_api_key: str | None = os.getenv("AGENT_BRIDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("AGENT_BRIDGE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    openai_model: str = os.getenv("AGENT_BRIDGE_OPENAI_MODEL", "gpt-4.1")
    anthropic_model: str = os.getenv(
        "AGENT_BRIDGE_ANTHROPIC_MODEL",
        "claude-sonnet-4-20250514",
    )
    merge_provider: str = os.getenv("AGENT_BRIDGE_MERGE_PROVIDER", "codex").strip().lower()
    proposer_provider: str = os.getenv("AGENT_BRIDGE_PROPOSER_PROVIDER", "codex").strip().lower()
    reviewer_provider: str = os.getenv("AGENT_BRIDGE_REVIEWER_PROVIDER", "claude").strip().lower()
    timeout_seconds: int = int(os.getenv("AGENT_BRIDGE_TIMEOUT_SECONDS", "60"))
    anthropic_max_tokens: int = int(os.getenv("AGENT_BRIDGE_ANTHROPIC_MAX_TOKENS", "1400"))
    anthropic_temperature: float = float(os.getenv("AGENT_BRIDGE_ANTHROPIC_TEMPERATURE", "0.1"))
    conversation_window: int = int(os.getenv("AGENT_BRIDGE_CONVERSATION_WINDOW", "16"))
    git_log_limit: int = int(os.getenv("AGENT_BRIDGE_GIT_LOG_LIMIT", "5"))


@lru_cache(maxsize=1)
def get_bridge_settings() -> BridgeSettings:
    return BridgeSettings()
