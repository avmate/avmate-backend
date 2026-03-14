# Agent Bridge

`agent_bridge` is a local sidecar that gives you one shared project session across two model providers.

What it does:
- stores a single session transcript in SQLite
- sends each new user turn to both OpenAI and Anthropic for normal chat mode
- supports a task mode where one provider proposes repo actions and the other reviews them
- merges the outputs into one answer or one execution brief
- snapshots local git state so both models see the same repo context

What it does not do:
- it does not attach to the proprietary Codex or Claude desktop/web chat session state
- it does not magically merge existing vendor chats
- it does not auto-edit files or auto-run commands yet

Instead, it creates a new shared session that behaves like one project chat loop.

## Security

Do not keep API keys in checked-in YAML or JSON files.

Use environment variables instead:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

If you already pasted a real API key into local config, rotate it before using this bridge.

## Environment

Optional settings:
- `AGENT_BRIDGE_DB_PATH`
- `AGENT_BRIDGE_PROJECT_ROOT`
- `AGENT_BRIDGE_OPENAI_MODEL`
- `AGENT_BRIDGE_ANTHROPIC_MODEL`
- `AGENT_BRIDGE_MERGE_PROVIDER`
- `AGENT_BRIDGE_PROPOSER_PROVIDER`
- `AGENT_BRIDGE_REVIEWER_PROVIDER`

Defaults:
- OpenAI model: `gpt-4.1`
- Anthropic model: `claude-sonnet-4-20250514`
- merge provider: `codex`
- proposer: `codex`
- reviewer: `claude`

## CLI

From `backend_refactor`:

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
.\.venv\Scripts\python.exe -m agent_bridge.cli --project-root . --mode task
```

Resume an existing session:

```powershell
.\.venv\Scripts\python.exe -m agent_bridge.cli --session-id "<session-id>" --show-candidates
```

In the CLI:
- `/task ...` runs a proposer/reviewer execution brief for one turn
- `/chat ...` runs normal merged chat for one turn

## API

Run the local API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn agent_bridge.api:app --reload --port 8011
```

Create a session:

```powershell
$body = @{ title = "AvMate fusion"; project_root = "C:\path\to\backend_refactor" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8011/sessions" -Method Post -ContentType "application/json" -Body $body
```

Send a normal chat turn:

```powershell
$body = @{ message = "Review the latest retrieval changes and tell me the next step." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8011/sessions/<session-id>/chat" -Method Post -ContentType "application/json" -Body $body
```

Send a task turn:

```powershell
$body = @{ message = "Implement BM25 retrieval and outline the test plan." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8011/sessions/<session-id>/task" -Method Post -ContentType "application/json" -Body $body
```

## Task mode output

Task mode returns:
- `proposal`: structured plan from the proposer model
- `review`: structured critique from the reviewer model
- `execution_brief`: merged final brief with solution, steps, validation, and risks

This is intended to be the decision layer before later automation of file edits or shell execution.

## Design notes

- OpenAI uses the Responses API, which OpenAI recommends for new projects.
- Anthropic uses the Messages API.
- The bridge stores conversation state locally instead of depending on provider-specific statefulness.
- Candidate replies are stored separately from the merged assistant reply, so the main transcript stays coherent.
- Task mode keeps the proposer/reviewer split explicit so execution planning is inspectable.
