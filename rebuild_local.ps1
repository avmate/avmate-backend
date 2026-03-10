if (Test-Path ".\chroma_db") {
    Remove-Item -Recurse -Force ".\chroma_db"
}

& ".\.venv\Scripts\python.exe" indexer.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload
