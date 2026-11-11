@echo off
cd /d "%~dp0"

python -c "import streamlit" >nul 2>nul
if %errorlevel%==0 (
    echo Using existing Python environment...
    python -m streamlit run app.py
    pause
    exit /b
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing dependencies. This can take several minutes the first time...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Dependency installation failed. Please run: python -m pip install -r requirements.txt
    pause
    exit /b
)

echo Starting PoisonShield...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
