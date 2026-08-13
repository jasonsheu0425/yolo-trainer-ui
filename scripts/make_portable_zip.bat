@echo off
setlocal
cd /d "%~dp0\.."
set "APP_DIR=dist\YOLO-Trainer-UI"
set "ARTIFACT_DIR=release_artifacts"
set "ZIP_PATH=%ARTIFACT_DIR%\YOLO-Trainer-UI-v0.11.1-windows-portable.zip"
if not exist "%APP_DIR%" (
  echo ERROR: dist\YOLO-Trainer-UI does not exist. Run build_pyinstaller_onedir.bat first.
  exit /b 1
)
for %%F in ("YOLO-Trainer-UI.exe" "README_FIRST.txt" "LICENSE.txt" "configs\app_settings.example.json") do (
  if not exist "%APP_DIR%\%%~F" (
    echo ERROR: Required portable file is missing: %%~F
    exit /b 1
  )
)
if not exist "%APP_DIR%\_internal" (
  echo ERROR: Required PyInstaller _internal folder is missing.
  exit /b 1
)
if exist "%APP_DIR%\configs\app_settings.json" del /q "%APP_DIR%\configs\app_settings.json"
if not exist "%ARTIFACT_DIR%" mkdir "%ARTIFACT_DIR%" || exit /b 1
if exist "%ZIP_PATH%" del /q "%ZIP_PATH%" || exit /b 1
powershell -NoProfile -Command "$badDirs = Get-ChildItem -LiteralPath '%APP_DIR%' -Recurse -Directory | Where-Object { $_.Name -in 'runs','datasets','build','dist','runtime','.venv','venv','models','weights' }; $badFiles = Get-ChildItem -LiteralPath '%APP_DIR%' -Recurse -File | Where-Object { $_.Name -eq 'app_settings.json' -or $_.Extension -in '.pt','.onnx','.engine' }; $bad = @($badDirs) + @($badFiles); if ($bad) { $bad.FullName | Write-Error; exit 1 }; Compress-Archive -Path '%APP_DIR%\*' -DestinationPath '%ZIP_PATH%' -CompressionLevel Optimal"
if errorlevel 1 exit /b 1
echo Portable ZIP created at %ZIP_PATH%.
