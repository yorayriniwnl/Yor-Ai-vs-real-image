@echo off
title AI Image Detector Launcher

echo Starting AI Image Detector...
echo.

cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

pause
