# AvMate Backend Deployment Guide
## Overview
This is the FastAPI backend for AvMate, providing vector search on aviation regulations stored in ChromaDB.

## Prerequisites
- Python 3.8+
- Anthropic API key (for AI explanations)
- ChromaDB database (built via `index_new.py`)

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run indexer: `python index_new.py` (downloads PDFs from R2, builds DB)
3. Set API key: `$env:ANTHROPIC_API_KEY = "your-key"`
4. Run server: `py -m uvicorn server:app --reload`

## Deployment to Railway
1. Push code to GitHub (this repo).
2. Go to https://railway.app > New Project > Deploy from GitHub repo.
3. Connect this repo.
4. In Railway dashboard > Variables: Add `ANTHROPIC_API_KEY` with your key.
5. Railway will build using the Dockerfile and deploy.
6. Get the URL (e.g., `https://avmate-backend.up.railway.app`).
7. Update your frontend to point to the Railway URL.

## Troubleshooting
- **AI Errors**: Ensure your Anthropic key is set in Railway Variables.
- **No Results**: The indexer must run successfully during Docker build.
- **CORS**: If frontend can't call backend, add CORS middleware in `server.py`:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
  ```

## Security
- Never commit API keys to GitHub.
- Use HTTPS in production.
