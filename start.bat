@echo off
echo =====================================
echo    TTU CSE Chatbot - Starting...
echo =====================================
echo.
echo Starting Backend Server (with venv)...
cd /d "%~dp0"
start "TTU CSE Backend" cmd /k "cd backend && ..\venv\Scripts\activate.bat && python main.py"
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "TTU CSE Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo =====================================
echo Both servers are starting...
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo =====================================
echo.
echo Press Ctrl+C in each window to stop the servers
pause
