from __future__ import annotations

import argparse

from .config import get_bridge_settings
from .service import create_default_service



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared Codex + Claude project chat.")
    parser.add_argument("--session-id", help="Existing session id to resume.")
    parser.add_argument("--title", help="Title for a new session.")
    parser.add_argument("--project-root", help="Project root to snapshot on each turn.")
    parser.add_argument(
        "--mode",
        choices=["chat", "task"],
        default="chat",
        help="chat returns a merged answer; task returns a proposer/reviewer execution brief.",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print raw Codex/Claude candidate replies after the merged answer.",
    )
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_bridge_settings()
    service = create_default_service(settings)

    if args.session_id:
        session = service.get_session(args.session_id)
    else:
        session = service.create_session(
            title=args.title or settings.default_title,
            project_root=args.project_root or str(settings.default_project_root),
            system_prompt=settings.default_system_prompt,
        )

    print(f"session_id={session.session_id}")
    print(f"project_root={session.project_root}")
    print(f"mode={args.mode}")
    print("Type 'exit' to quit. Prefix a message with '/task ' or '/chat ' to override mode for one turn.\n")

    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        turn_mode = args.mode
        if user_message.startswith("/task "):
            turn_mode = "task"
            user_message = user_message[6:].strip()
        elif user_message.startswith("/chat "):
            turn_mode = "chat"
            user_message = user_message[6:].strip()

        if turn_mode == "task":
            turn = service.task(
                session_id=session.session_id,
                user_message=user_message,
                include_workspace_snapshot=True,
            )
            print(f"\nexecution_brief>\n{turn.execution_brief}\n")
            if args.show_candidates:
                print("proposal>")
                print(turn.proposal)
                print()
                print("review>")
                print(turn.review)
                print()
            if turn.warnings:
                print("warnings:")
                for warning in turn.warnings:
                    print(f"- {warning}")
                print()
            continue

        turn = service.chat(
            session_id=session.session_id,
            user_message=user_message,
            include_workspace_snapshot=True,
        )
        print(f"\nmerged>\n{turn.answer}\n")
        if turn.warnings:
            print("warnings:")
            for warning in turn.warnings:
                print(f"- {warning}")
            print()
        if args.show_candidates:
            for name, candidate in sorted(turn.candidates.items()):
                print(f"{name}>\n{candidate}\n")


if __name__ == "__main__":
    main()
