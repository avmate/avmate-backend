from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderReply:
    provider: str
    model: str
    content: str
    raw: dict[str, Any]


class BaseProvider:
    name: str
    model: str

    def generate(self, *, system_prompt: str, messages: list[dict[str, str]]) -> ProviderReply:
        raise NotImplementedError


class OpenAIResponsesProvider(BaseProvider):
    def __init__(self, *, api_key: str, model: str, timeout_seconds: int, provider_name: str = "codex") -> None:
        self.name = provider_name
        self.model = model
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._http = requests.Session()

    def generate(self, *, system_prompt: str, messages: list[dict[str, str]]) -> ProviderReply:
        if not self._api_key:
            raise ProviderError("OpenAI API key is not configured.")

        body = {
            "model": self.model,
            "instructions": system_prompt,
            "input": [
                {"role": item["role"], "content": item["content"]}
                for item in messages
            ],
            "store": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = self._http.post(
            OPENAI_RESPONSES_URL,
            headers=headers,
            json=body,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = self._extract_text(payload)
        if not content:
            raise ProviderError("OpenAI response did not include text output.")
        return ProviderReply(provider=self.name, model=self.model, content=content, raw=payload)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = payload.get("output") or []
        text_parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            for content_block in item.get("content") or []:
                if not isinstance(content_block, dict):
                    continue
                text = content_block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n\n".join(text_parts).strip()


class AnthropicMessagesProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_tokens: int,
        temperature: float,
        provider_name: str = "claude",
    ) -> None:
        self.name = provider_name
        self.model = model
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http = requests.Session()

    def generate(self, *, system_prompt: str, messages: list[dict[str, str]]) -> ProviderReply:
        if not self._api_key:
            raise ProviderError("Anthropic API key is not configured.")

        body = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in messages
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        response = self._http.post(
            ANTHROPIC_MESSAGES_URL,
            headers=headers,
            json=body,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text_parts: list[str] = []
        for item in payload.get("content") or []:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        content = "\n\n".join(text_parts).strip()
        if not content:
            raise ProviderError("Anthropic response did not include text output.")
        return ProviderReply(provider=self.name, model=self.model, content=content, raw=payload)
