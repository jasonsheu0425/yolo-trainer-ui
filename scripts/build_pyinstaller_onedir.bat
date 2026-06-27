@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found.
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name "YOLO-Trainer-UI" main.py
if errorlevel 1 exit /b %errorlevel%
echo Build prepared in dist\YOLO-Trainer-UI.
