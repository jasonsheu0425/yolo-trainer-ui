@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found.
  exit /b 1
)
set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --name "YOLO-Trainer-UI" ^
  --add-data "configs\app_settings.example.json;configs" ^
  --add-data "docs;docs" ^
  --add-data "assets;assets" ^
  --collect-data matplotlib ^
  --hidden-import yaml ^
  --hidden-import psutil ^
  --hidden-import matplotlib.backends.backend_qtagg ^
  --exclude-module torch ^
  --exclude-module torchvision ^
  --exclude-module ultralytics ^
  --exclude-module scipy ^
  --exclude-module pytest ^
  main.py
if errorlevel 1 exit /b 1

set "APP_DIR=dist\YOLO-Trainer-UI"
if not exist "%APP_DIR%\YOLO-Trainer-UI.exe" (
  echo ERROR: PyInstaller completed without creating YOLO-Trainer-UI.exe.
  exit /b 1
)
copy /y "packaging\README_FIRST.txt" "%APP_DIR%\README_FIRST.txt" >nul || exit /b 1
copy /y "LICENSE" "%APP_DIR%\LICENSE.txt" >nul || exit /b 1
if not exist "%APP_DIR%\configs" mkdir "%APP_DIR%\configs" || exit /b 1
copy /y "configs\app_settings.example.json" "%APP_DIR%\configs\app_settings.example.json" >nul || exit /b 1
if exist "%APP_DIR%\configs\app_settings.json" del /q "%APP_DIR%\configs\app_settings.json"
if exist "%APP_DIR%\docs" rmdir /s /q "%APP_DIR%\docs"
xcopy "docs" "%APP_DIR%\docs\" /e /i /y >nul || exit /b 1
if exist "assets" (
  if exist "%APP_DIR%\assets" rmdir /s /q "%APP_DIR%\assets"
  xcopy "assets" "%APP_DIR%\assets\" /e /i /y >nul || exit /b 1
)
echo Portable onedir build created at %APP_DIR%.
