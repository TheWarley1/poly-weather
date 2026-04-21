@echo off
REM ============================================
REM Polymarket Weather Edge - Automated Pipeline
REM ============================================
REM Runs forecast_logger.py + paper_trader.py.
REM Designed to be scheduled every 6 hours by Windows Task Scheduler.
REM
REM TROUBLESHOOTING:
REM   - Check pipeline.log in this folder for error messages
REM   - If python not found: Task Scheduler doesn't inherit your PATH.
REM     Edit PYTHON_EXE below to point to your full python.exe path.
REM     Find it by running "where python" in a fresh cmd window.
REM ============================================

cd /d "%~dp0"

REM Use explicit Python path if "python" isn't found by Task Scheduler.
REM Change this to your Python install if the default fails.
set PYTHON_EXE=python

set LOG_FILE=pipeline.log

echo. >> %LOG_FILE%
echo ============================================ >> %LOG_FILE%
echo Run started: %date% %time% >> %LOG_FILE%
echo ============================================ >> %LOG_FILE%

REM Verify python is reachable before doing anything
%PYTHON_EXE% --version >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo CRITICAL: Python not found. Edit PYTHON_EXE in this .bat to full path. >> %LOG_FILE%
    echo Run "where python" in cmd to find it. >> %LOG_FILE%
    exit /b 1
)

REM Step 1: Log forecasts + backfill actuals
echo. >> %LOG_FILE%
echo [1/2] forecast_logger.py >> %LOG_FILE%
%PYTHON_EXE% forecast_logger.py >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo ERROR: forecast_logger failed with exit code %errorlevel% >> %LOG_FILE%
) else (
    echo OK: forecast_logger completed >> %LOG_FILE%
)

REM Step 2: Paper trader - resolve open trades + scan for new edges
echo. >> %LOG_FILE%
echo [2/2] paper_trader.py >> %LOG_FILE%
%PYTHON_EXE% paper_trader.py >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo ERROR: paper_trader failed with exit code %errorlevel% >> %LOG_FILE%
) else (
    echo OK: paper_trader completed >> %LOG_FILE%
)

echo. >> %LOG_FILE%
echo Run finished: %date% %time% >> %LOG_FILE%
