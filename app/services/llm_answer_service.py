from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.schemas import ReferenceItem


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class LLMAnswerService:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        *,
        query: str,
        references: list[ReferenceItem],
        fallback_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.enabled or not references:
            return None

        reference_blocks = []
        for idx, ref in enumerate(references[:5], start=1):
            excerpt = " ".join(ref.text.split())
            excerpt = excerpt[:1800]
            reference_blocks.append(
                (
                    f"[{idx}] citation={ref.citation}\n"
                    f"title={ref.title}\n"
                    f"page_ref={ref.page_ref}\n"
                    f"table_ref={ref.table_ref}\n"
                    f"source_file={ref.source_file}\n"
                    f"text_excerpt={excerpt}"
                )
            )

        allowed_citations = [ref.citation for ref in references[:5]]
        system_prompt = (
            "You are a legal aviation assistant. Use only the provided references. "
            "Do not invent any regulation content, citation, or number. "
            "If evidence is weak, say so explicitly and stay conservative. "
            "Return only valid JSON."
        )
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Allowed citations:\n{json.dumps(allowed_citations)}\n\n"
            f"References:\n{chr(10).join(reference_blocks)}\n\n"
            "Return JSON object with keys:\n"
            "- answer (short, direct)\n"
            "- legal_explanation (short legal framing)\n"
            "- plain_english (plain language interpretation of the regulatory meaning)\n"
            "- example (short operational story grounded in the references)\n"
            "- study_questions (exactly 5 items)\n"
            "- study_answers (exactly 5 items)\n"
            "Constraints:\n"
            "- In answer, include the primary citation verbatim.\n"
            "- Use only allowed citations.\n"
            "- plain_english must explain what the rule means in practical terms without adding facts not in references.\n"
            "- example must be scenario-based, practical, and explicitly consistent with reference conditions.\n"
            "- Keep output concise and practical.\n"
        )

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
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

        content_blocks = payload.get("content") or []
        text_parts = [block.get("text", "") for block in content_blocks if isinstance(block, dict)]
        raw_text = "\n".join(part for part in text_parts if part).strip()
        if not raw_text:
            return None

        parsed = self._parse_json_block(raw_text)
        if not parsed:
            return None

        sanitized = self._sanitize_output(parsed, fallback_payload, allowed_citations)
        return sanitized

    def interpret_query(self, query: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        query_clean = self._clean_text(query)
        if not query_clean:
            return None

        system_prompt = (
            "You classify aviation regulation queries for retrieval. "
            "Do not answer the query. Extract intent and search wording only. "
            "Return only valid JSON."
        )
        user_prompt = (
            f"User query:\n{query_clean}\n\n"
            "Return JSON object with keys:\n"
            "- intent (short label)\n"
            "- regulation_type (one of: AIP, CASR, CAR, CAO, MOS, CAA — or null)\n"
            "- rewritten_query (clear retrieval-focused rewrite preserving original meaning, using official regulatory terminology)\n"
            "- keywords (5 to 10 short search terms/phrases grounded in the query)\n"
            "Constraints:\n"
            "- Do not add facts not implied by the query.\n"
            "- Keep rewritten_query under 240 characters.\n"
            "- keywords must be plain strings.\n"
            "- regulation_type must be a single string or null, not a list.\n"
            "- Set regulation_type ONLY when the query explicitly names a regulation family (e.g. 'CASR 61', 'AIP ENR'). Set null for topic-only queries like 'QNH', 'fuel reserve', 'circling radius'.\n"
            "- For 'stable approach' or 'stabilised approach' queries without explicit family, set regulation_type to null.\n"
        )

        body = {
            "model": self._model,
            "max_tokens": min(420, self._max_tokens),
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
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
        content_blocks = payload.get("content") or []
        text_parts = [block.get("text", "") for block in content_blocks if isinstance(block, dict)]
        raw_text = "\n".join(part for part in text_parts if part).strip()
        if not raw_text:
            return None

        parsed = self._parse_json_block(raw_text)
        if not parsed:
            return None

        rewritten_query = self._clean_text(parsed.get("rewritten_query")) or query_clean
        intent = self._clean_text(parsed.get("intent"))
        raw_keywords = parsed.get("keywords") if isinstance(parsed.get("keywords"), list) else []
        keywords = [
            self._clean_text(item).lower()
            for item in raw_keywords
            if isinstance(item, str) and len(self._clean_text(item)) >= 3
        ]
        keywords = list(dict.fromkeys(keywords))[:10]

        raw_reg_type = parsed.get("regulation_type")
        regulation_type = (
            raw_reg_type.strip().upper()
            if isinstance(raw_reg_type, str) and raw_reg_type.strip()
            else None
        )

        return {
            "intent": intent,
            "regulation_type": regulation_type,
            "rewritten_query": rewritten_query[:240],
            "keywords": keywords,
        }

    def _parse_json_block(self, raw_text: str) -> dict[str, Any] | None:
        candidate = raw_text.strip()

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _sanitize_output(
        self,
        raw: dict[str, Any],
        fallback_payload: dict[str, Any],
        allowed_citations: list[str],
    ) -> dict[str, Any]:
        answer = self._clean_text(raw.get("answer")) or fallback_payload["answer"]
        legal_explanation = self._clean_text(raw.get("legal_explanation")) or fallback_payload["legal_explanation"]
        plain_english = self._clean_text(raw.get("plain_english")) or fallback_payload["plain_english"]
        example = self._clean_text(raw.get("example")) or fallback_payload["example"]

        if self._mentions_unknown_citation(answer, allowed_citations):
            answer = fallback_payload["answer"]
        if self._mentions_unknown_citation(legal_explanation, allowed_citations):
            legal_explanation = fallback_payload["legal_explanation"]
        if self._mentions_unknown_citation(plain_english, allowed_citations):
            plain_english = fallback_payload["plain_english"]
        if self._mentions_unknown_citation(example, allowed_citations):
            example = fallback_payload["example"]

        study_questions = self._normalize_list(raw.get("study_questions"), fallback_payload["study_questions"])
        study_answers = self._normalize_list(raw.get("study_answers"), fallback_payload["study_answers"])

        return {
            "answer": answer,
            "legal_explanation": legal_explanation,
            "plain_english": plain_english,
            "example": example,
            "study_questions": study_questions,
            "study_answers": study_answers,
        }

    def _normalize_list(self, value: Any, fallback: list[str]) -> list[str]:
        if not isinstance(value, list):
            return fallback
        cleaned = [self._clean_text(item) for item in value if self._clean_text(item)]
        if len(cleaned) < 5:
            return fallback
        return cleaned[:5]

    def _clean_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs: list[str] = []
        for paragraph in re.split(r"\n{2,}", normalized):
            cleaned = " ".join(paragraph.split()).strip()
            if cleaned:
                paragraphs.append(cleaned)
        return "\n\n".join(paragraphs).strip()

    def _mentions_unknown_citation(self, text: str, allowed_citations: list[str]) -> bool:
        if not text:
            return False
        allowed = {citation.lower() for citation in allowed_citations}
        matches = re.findall(r"\b(?:AIP|CASR|CAR|CAO|MOS|CAA)\b[^.;,\n]*", text, flags=re.IGNORECASE)
        for match in matches:
            normalized = " ".join(match.split()).lower()
            if not any(citation in normalized for citation in allowed):
                return True
        return False
