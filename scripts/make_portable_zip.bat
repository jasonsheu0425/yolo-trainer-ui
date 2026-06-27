@echo off
setlocal
cd /d "%~dp0\.."
if not exist "dist\YOLO-Trainer-UI" (
  echo ERROR: dist\YOLO-Trainer-UI does not exist. Run build_pyinstaller_onedir.bat first.
  exit /b 1
)
if exist "dist\YOLO-Trainer-UI-portable.zip" del /q "dist\YOLO-Trainer-UI-portable.zip"
powershell -NoProfile -Command "Compress-Archive -Path 'dist\YOLO-Trainer-UI\*' -DestinationPath 'dist\YOLO-Trainer-UI-portable.zip' -CompressionLevel Optimal"
if errorlevel 1 exit /b %errorlevel%
echo Portable ZIP created at dist\YOLO-Trainer-UI-portable.zip.
