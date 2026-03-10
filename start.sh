#!/bin/bash
echo "=== AvMate Starting ==="

COUNT=$(python -c "
import chromadb
try:
    c = chromadb.PersistentClient(path='./chroma_db')
    col = c.get_or_create_collection('avmate_regulations')
    print(col.count())
except:
    print(0)
")

echo "Collection count: $COUNT"

if [ "$COUNT" -eq "0" ]; then
    echo "Database empty — running indexer in background..."
    python index_new.py &
fi

echo "Starting server..."
exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
