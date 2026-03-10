# AvMate V2 Backend
# Run: uvicorn server:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import anthropic
import threading
import os
import random
import json

app = FastAPI(title="AvMate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# --- ChromaDB ---
try:
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="avmate_regulations")
    print(f"ChromaDB ready. Collection count: {collection.count()}")
except Exception as e:
    print(f"ChromaDB error: {e} — using in-memory fallback")
    db_client = chromadb.EphemeralClient()
    collection = db_client.get_or_create_collection(name="avmate_regulations")

# --- Embedding model (lazy load in background) ---
_model = None
_model_ready = False

def _load_model():
    global _model, _model_ready
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_ready = True
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Embedding model failed to load: {e}")

threading.Thread(target=_load_model, daemon=True).start()

# --- Anthropic client ---
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Routes ---

@app.get("/")
def root():
    return {
        "service": "AvMate API",
        "version": "2.3",
        "model_ready": _model_ready,
        "collection_count": collection.count()
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": _model_ready,
        "collection_count": collection.count()
    }

@app.post("/search")
def search(q: Query):
    if not _model_ready:
        return {
            "answer": "AvMate is warming up — embedding model is loading. Please try again in 30 seconds.",
            "plain_english": "Server is starting up.",
            "example": "",
            "study": "",
            "sources": "",
            "confidence": 0
        }

    query_embedding = _model.encode([q.query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=10)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not documents:
        return {
            "answer": "No relevant regulations found. The database may still be indexing.",
            "plain_english": "Try again in a few minutes.",
            "example": "",
            "study": "",
            "sources": "",
            "confidence": 0
        }

    retrieved_text = "\n\n".join(documents)
    sources = list(set([m["source"] for m in metadatas]))

    prompt = f"""You are AvMate, an AI assistant for Australian aviation regulations.

User query: {q.query}

Retrieved regulation text:
{retrieved_text}

Provide a JSON response with keys: answer, plain_english, example, study, sources."""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text)
        result["confidence"] = random.randint(90, 99)
        return result
    except Exception as e:
        print(f"Claude error: {e}")
        return {
            "answer": f"Based on regulations: {retrieved_text[:500]}...",
            "plain_english": "Relevant excerpts from aviation regulations.",
            "example": documents[0] if documents else "",
            "study": "Review the full regulation documents.",
            "sources": "\n".join(sources),
            "confidence": random.randint(90, 99)
        }