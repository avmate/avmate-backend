#!/bin/bash
set -e

echo "=== AvMate Startup ==="

# Check if DB already has data
COUNT=$(python -c "
import chromadb, sys
try:
    c = chromadb.PersistentClient(path='./chroma_db')
    col = c.get_or_create_collection('avmate_regulations')
    print(col.count())
except:
    print(0)
")

echo "Current collection count: $COUNT"

if [ "$COUNT" -eq "0" ]; then
    echo "Database empty — running indexer..."
    python index_new.py
    echo "Indexing complete."
else
    echo "Database already populated — skipping indexer."
fi

echo "Starting server..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
