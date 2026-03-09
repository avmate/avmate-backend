#!/bin/bash
set -e

echo "=== AvMate Startup ==="

# Check collection count
COUNT=$(python -c "
import chromadb
try:
    c = chromadb.PersistentClient(path='./chroma_db')
    col = c.get_or_create_collection('avmate_regulations')
    print(col.count())
except:
    print(0)
")

echo "Current collection count: $COUNT"

if [ "$COUNT" -eq "0" ]; then
    echo "Database empty — starting indexer in background..."
    python index_new.py &
    echo "Indexer running in background. Server starting now."
else
    echo "Database populated — skipping indexer."
fi

echo "Starting server..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
