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
  - `CORS_ALLOW_ORIGINS=https://beta.avmate.com.au,https://avmate.com.au,http://localhost:3000`

## Search behavior

- `/health` remains lightweight and should stay responsive during cold starts.
- `/search` returns a `503` if the index is empty instead of crashing the container.
- Results are sourced from indexed regulation text and include citations, references, and study questions.

## Indexing workflow

The indexer downloads each PDF from R2, extracts text with `pdfplumber`, splits the text into regulation sections when possible, chunks those sections, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and upserts them into Chroma.

Update `data/regulations_manifest.json` as you add new CASA source files or move to a hosted manifest in R2.
