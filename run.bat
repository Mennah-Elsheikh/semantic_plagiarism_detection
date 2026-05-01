@echo off
title Plagiarism Detection System
color 0A

echo ================================================
echo   Plagiarism Detection System - Launcher
echo ================================================
echo.

set PYTHON=D:\programs\envs\tensorflow\python.exe
set DIR=%~dp0

echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI - Plagiarism API" cmd /k "cd /d %DIR% && %PYTHON% -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for API to initialize...
timeout /t 5 /nobreak > nul

echo [2/2] Starting Streamlit frontend on http://localhost:8501 ...
start "Streamlit - Plagiarism UI" cmd /k "cd /d %DIR% && %PYTHON% -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless false"

echo.
echo ================================================
echo   Both services are starting up!
echo ------------------------------------------------
echo   API Docs  : http://localhost:8000/docs
echo   Frontend  : http://localhost:8501
echo ================================================
echo.
timeout /t 4 /nobreak > nul

start "" "http://localhost:8501"

echo Press any key to exit this launcher window...
pause > nul
