@echo off
echo 🎓 HSC Bangla RAG System - Windows Launcher
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the setup script
echo 🚀 Starting system setup...
python run.py

pause 