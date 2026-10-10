@echo off
REM Launch Kronos Web UI using the project virtual environment
cd /d "%~dp0"
".venv\Scripts\python.exe" "webui\run.py"
