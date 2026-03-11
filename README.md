# AvMate Backend

FastAPI backend for Australian aviation regulation search. The backend is structured around:

- `app.main`: lightweight API startup for Railway
- `indexer.py`: explicit indexing command for R2-hosted PDFs
- `data/regulations_manifest.json`: document manifest used to build the vector index
- `chroma_db/`: local Chroma persistence directory

## Local development

1. Install dependencies:
   `pip install -r requirements.txt`
2. Build the local index from R2:
   `python indexer.py`
3. Run the API:
   `uvicorn app.main:app --reload`

## Railway deployment

- Use the included `Dockerfile`.
- Railway should start `app.main:app`, not the legacy `server.py` logic.
- Set the following variables as needed:
  - `PORT`
  - `R2_BASE_URL`
  - `R2_MANIFEST_URL` if you want the manifest hosted remotely
  - `PRELOAD_EMBEDDINGS=true` if you want the model loaded shortly after startup
  - `AUTO_INDEX_ON_STARTUP=true` if you want Railway to build the Chroma index in a background thread after boot
  - `FORCE_REINDEX_ON_STARTUP=true` for one deploy when you need to rebuild an existing index with new parsing/ranking logic
  - `CORS_ALLOW_ORIGINS=https://beta.avmate.com.au,https://avmate.com.au,http://localhost:3000`
  - `RATE_LIMIT_ENABLED=true`
  - `RATE_LIMIT_REQUESTS=120`
  - `RATE_LIMIT_WINDOW_SECONDS=60`
  - `ENABLE_LLM_ANSWERS=true` to enable grounded LLM synthesis
  - `ANTHROPIC_API_KEY=...` required when `ENABLE_LLM_ANSWERS=true`
  - `LLM_MODEL=claude-3-5-sonnet-latest`
  - `LLM_TIMEOUT_SECONDS=45`
  - `LLM_MAX_TOKENS=1100`
  - `LLM_TEMPERATURE=0.1`

## Search behavior

- `/health` remains lightweight and should stay responsive during cold starts.
- `/ready` returns `200` only when the service is ready to answer search requests.
- `/search` returns a `503` if the index is empty instead of crashing the container.
- Results are sourced from indexed regulation text and include citations, references, and study questions.
- Numeric queries (for example circling minima by aircraft category) are re-ranked with lexical and numeric evidence checks.
- Strict numeric queries use stronger evidence checks (table signature + unit evidence) before a reference is accepted.
- AIP citations include page and subsection when page markers are present in the extracted text.
- Optional grounded LLM synthesis runs only after retrieval and only using the selected references.
- `X-Request-ID` is returned on API responses for easier tracing in logs.

## Indexing workflow

The indexer downloads each PDF from R2, extracts text with `pdfplumber`, splits the text into regulation sections when possible, chunks those sections, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and upserts them into Chroma.

Update `data/regulations_manifest.json` as you add new CASA source files or move to a hosted manifest in R2.

If you change parsing or citation logic, rebuild the index:
- `python indexer.py`

## Quick validation

1. Check readiness:
   `Invoke-WebRequest http://127.0.0.1:8000/ready | Select-Object -Expand Content`
2. Validate circling minima query:
   `@'`
   `import requests, json`
   `resp = requests.post("http://127.0.0.1:8000/search", json={"query":"what is the circling radius for a cat C aircraft","top_k":5}, timeout=60)`
   `print(resp.status_code)`
   `print(json.dumps(resp.json(), indent=2)[:4000])`
   `'@ | python -`
3. Validate study guide source lookup:
   `Invoke-WebRequest http://127.0.0.1:8000/study-guide -Method Post -ContentType 'application/json' -Body '{"test_name":"instrument rating - aeroplane","max_items":10}' | Select-Object -Expand Content`
