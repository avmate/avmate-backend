from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_bridge.providers import ProviderReply
from agent_bridge.service import AgentBridgeService
from agent_bridge.storage import BridgeStore


class StubProvider:
    def __init__(self, *, name: str, model: str) -> None:
        self.name = name
        self.model = model

    def generate(self, *, system_prompt: str, messages: list[dict[str, str]]) -> ProviderReply:
        if "execution planner" in system_prompt:
            content = json.dumps(
                {
                    "summary": "Implement BM25 hybrid retrieval.",
                    "assumptions": ["Existing search_service.py owns retrieval orchestration."],
                    "steps": [
                        "Add lexical scoring alongside semantic search.",
                        "Blend and rerank the result set.",
                    ],
                    "files_to_touch": ["app/services/search_service.py"],
                    "commands_to_run": ["python -m unittest discover -s tests -p test_*.py -v"],
                    "tests_to_run": ["python regression_queries.py"],
                    "risks": ["Ranking changes may break exact-citation regressions."],
                }
            )
        elif "You are the reviewer" in system_prompt:
            content = json.dumps(
                {
                    "verdict": "approve",
                    "critical_issues": [],
                    "suggested_changes": ["Retain exact-citation boosts during reranking."],
                    "required_tests": ["python regression_queries.py", "python regression_test.py"],
                    "approved_steps": [
                        "Add lexical scoring alongside semantic retrieval.",
                        "Preserve exact-citation boosts in reranking.",
                        "Run both regression suites.",
                    ],
                }
            )
        else:
            content = f"{self.name} says check search_service.py."
        return ProviderReply(
            provider=self.name,
            model=self.model,
            content=content,
            raw={"messages_seen": len(messages), "system_prompt": system_prompt},
        )



def build_service(tmp_path: Path) -> AgentBridgeService:
    store = BridgeStore(tmp_path / "agent-bridge.db")
    providers = {
        "codex": StubProvider(name="codex", model="stub-openai"),
        "claude": StubProvider(name="claude", model="stub-anthropic"),
    }
    return AgentBridgeService(
        store=store,
        providers=providers,
        merge_provider_name="missing",
        proposer_provider_name="codex",
        reviewer_provider_name="claude",
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

    def test_task_mode_creates_proposal_review_and_execution_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = build_service(tmp_path)
            session = service.create_session(
                title="task",
                project_root=str(tmp_path),
                system_prompt="Be precise.",
            )

            turn = service.task(
                session_id=session.session_id,
                user_message="Implement BM25 and validate retrieval quality.",
                include_workspace_snapshot=False,
            )

            self.assertEqual(turn.proposal["summary"], "Implement BM25 hybrid retrieval.")
            self.assertEqual(turn.review["verdict"], "approve")
            self.assertIn("**Solution**", turn.execution_brief)
            self.assertIn("Run both regression suites.", turn.execution_brief)

            messages = service.list_messages(session.session_id, include_candidates=True)
            self.assertEqual(
                [message.role for message in messages],
                ["user", "candidate", "candidate", "assistant"],
            )
            self.assertEqual(messages[1].metadata.get("stage"), "proposal")
            self.assertEqual(messages[2].metadata.get("stage"), "review")
            self.assertEqual(messages[3].metadata.get("mode"), "task")

    def test_workspace_snapshot_handles_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = build_service(tmp_path)
            snapshot = service._build_workspace_snapshot(tmp_path / "missing")  # noqa: SLF001
            self.assertIn("workspace_status=missing", snapshot)


if __name__ == "__main__":
    unittest.main()
