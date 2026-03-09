# AvMate Backend Deployment Guide
## Overview
This is the FastAPI backend for AvMate, providing vector search on aviation regulations stored in ChromaDB.

## Prerequisites
- Python 3.8+
- Anthropic API key (for AI explanations)
- ChromaDB database (built via `index.py`)

## Local Setup (Already Done)
1. Install dependencies: `pip install -r requirements.txt`
2. Run indexer: `python index.py` (downloads PDFs from R2, builds DB)
3. Set API key: `$env:ANTHROPIC_API_KEY = "your-key"`
4. Run server: `py -m uvicorn server:app --reload`

## Deployment to Railway (Recommended)
Railway is free for small apps and supports Python/FastAPI.

1. **Create Railway Account**: Go to https://railway.app and sign up.
2. **Create Project**: Click "New Project" > "Deploy from GitHub repo".
3. **Push Code to GitHub**:
   - Create a new GitHub repo (e.g., `avmate-backend`).
   - Upload `server.py`, `requirements.txt`, `chroma_db/` folder, and this README.
   - Commit and push.
4. **Connect Repo**: In Railway, select your repo and deploy.
5. **Set Environment Variables**:
   - In Railway dashboard > Variables: Add `ANTHROPIC_API_KEY` with your key.
6. **Deploy**: Railway will build and run the app. Get the URL (e.g., `https://avmate-backend.up.railway.app`).
7. **Update Frontend**: In your AvMate HTML (on Cloudflare), change the API endpoint to the Railway URL.

## Alternative: Deploy to Render
1. Go to https://render.com, sign up.
2. Create "Web Service" from GitHub repo.
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add env var: `ANTHROPIC_API_KEY`
6. Deploy and get the URL.

## Troubleshooting
- **AI Errors**: Ensure your Anthropic key has access to models (check console.anthropic.com).
- **No Results**: Re-run `index.py` if DB is missing.
- **CORS**: If frontend can't call backend, add CORS middleware in `server.py`:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
  ```

## Security
- Never commit API keys to GitHub.
- Use HTTPS in production.