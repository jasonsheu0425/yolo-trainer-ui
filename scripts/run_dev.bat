@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found. Create it and install requirements first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python main.py
