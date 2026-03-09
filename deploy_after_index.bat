@echo off
echo ===================================
echo AvMate - Post-Index Deploy Script
echo ===================================

cd /d "c:\Users\Taylz\Documents\AvMate App\Chat GPT\AvMate_V2_Starter\backend"

echo.
echo [1/4] Checking chroma_db was built...
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_or_create_collection('avmate_regulations'); count=col.count(); print(f'Collection count: {count}'); exit(0 if count > 0 else 1)"
if errorlevel 1 (
    echo ERROR: chroma_db is empty. Run index_local.py first.
    pause
    exit /b 1
)

echo.
echo [2/4] Staging chroma_db for git...
git add chroma_db/ server.py main.py Procfile railway.json .gitignore index_local.py

echo.
echo [3/4] Committing...
git commit -m "Add indexed chroma_db and latest server fixes"

echo.
echo [4/4] Pushing to GitHub...
git push origin master

echo.
echo ===================================
echo Done! Railway will auto-redeploy.
echo Test at: https://avmate-backend-production.up.railway.app/health
echo ===================================
pause
