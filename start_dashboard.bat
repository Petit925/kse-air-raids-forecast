@echo off
REM 🛡️ Air Raid Workforce Planner — one-click launcher
REM Double-click this file. It will open the dashboard in your default browser.
cd /d %~dp0
echo.
echo ============================================================
echo   Air Raid Workforce Planner — starting Streamlit
echo   When ready, your browser will open automatically.
echo   To stop: close this window or press Ctrl+C.
echo ============================================================
echo.
.venv\Scripts\streamlit.exe run app.py --server.headless false --browser.gatherUsageStats false
pause
