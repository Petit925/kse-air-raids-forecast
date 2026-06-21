@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo.
echo ============================================================
echo   Air Raid Workforce Planner - starting Streamlit
echo   Browser will open at http://localhost:8501
echo   To stop: close this window or press Ctrl+C
echo ============================================================
echo.
".venv\Scripts\streamlit.exe" run app.py --server.headless=false --browser.gatherUsageStats=false
pause
