from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_bridge.providers import ProviderReply
from agent_bridge.service import AgentBridgeService
from agent_bridge.storage import BridgeStore


class StubProvider:
    def __init__(self, *, name: str, model: str, reply: str) -> None:
        self.name = name
        self.model = model
        self.reply = reply

    def generate(self, *, system_prompt: str, messages: list[dict[str, str]]) -> ProviderReply:
        return ProviderReply(
            provider=self.name,
            model=self.model,
            content=self.reply,
            raw={"messages_seen": len(messages), "system_prompt": system_prompt},
        )



def build_service(tmp_path: Path) -> AgentBridgeService:
    store = BridgeStore(tmp_path / "agent-bridge.db")
    providers = {
        "codex": StubProvider(name="codex", model="stub-openai", reply="Codex says check search_service.py."),
        "claude": StubProvider(name="claude", model="stub-anthropic", reply="Claude says verify regression_queries.py."),
    }
    return AgentBridgeService(
        store=store,
        providers=providers,
        merge_provider_name="missing",
        conversation_window=10,
        git_log_limit=3,
    )


class AgentBridgeTests(unittest.TestCase):
    def test_agent_bridge_persists_transcript_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = build_service(tmp_path)
            session = service.create_session(
                title="test",
                project_root=str(tmp_path),
                system_prompt="Be precise.",
            )

            turn = service.chat(
                session_id=session.session_id,
                user_message="What changed in the repo?",
                include_workspace_snapshot=False,
            )

            self.assertIn("Two candidate replies are available.", turn.answer)
            self.assertEqual(set(turn.candidates), {"codex", "claude"})
            self.assertTrue(
                any("configured merge provider unavailable" in warning for warning in turn.warnings)
            )

            messages = service.list_messages(session.session_id, include_candidates=True)
            self.assertEqual(
                [message.role for message in messages],
                ["user", "candidate", "candidate", "assistant"],
            )

    def test_workspace_snapshot_handles_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = build_service(tmp_path)
            snapshot = service._build_workspace_snapshot(tmp_path / "missing")  # noqa: SLF001
            self.assertIn("workspace_status=missing", snapshot)


if __name__ == "__main__":
    unittest.main()
