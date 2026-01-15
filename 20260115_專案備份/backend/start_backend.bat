@echo off
echo ======================================
echo TTU CSE Chatbot - Backend Server
echo ======================================
cd /d %~dp0
echo.
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat
echo.
echo Starting backend server...
python main.py
pause
