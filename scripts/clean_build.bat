@echo off
setlocal
cd /d "%~dp0\.."
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
powershell -NoProfile -Command "Get-ChildItem -LiteralPath . -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike '*\.venv\*' } | Remove-Item -Recurse -Force"
echo Build output and project Python caches removed. runs and datasets were preserved.
