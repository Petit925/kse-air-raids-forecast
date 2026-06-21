@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo.
echo ============================================================
echo   Building static report.html (this takes ~30 seconds)
echo ============================================================
echo.
".venv\Scripts\python.exe" -m src.build_report
echo.
echo Done. Opening report.html in your default browser...
start "" "report.html"
