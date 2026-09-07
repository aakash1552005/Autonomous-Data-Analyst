@echo off
setlocal enabledelayedexpansion

:: ==============================================================================
:: Autonomous Data Analyst — Windows One-Command Launcher
:: ==============================================================================
:: Determines project root from script location, detects or provisions a virtual
:: environment, verifies dependencies idempotently, and launches Streamlit.
::
:: Usage:
::   run.bat               - Standard launch (provisions if needed, runs Streamlit)
::   run.bat --check-only  - Non-interactive validation (verifies environment & exits)
::   run.bat --help        - Displays usage guidance
:: ==============================================================================

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

if /i "%~1"=="--help" (
    echo Usage: run.bat [--check-only ^| --help]
    echo.
    echo Options:
    echo   --check-only  Verify Python virtual environment and dependencies without launching UI.
    echo   --help        Display this help message.
    echo.
    exit /b 0
)

set "VENV_DIR=%PROJECT_ROOT%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe"

:: 1. Detect or create virtual environment
if not exist "%VENV_PYTHON%" (
    echo [INFO] Virtual environment not detected at %VENV_DIR%.
    echo [INFO] Searching for system Python installation...

    set "SYSTEM_PYTHON="
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "SYSTEM_PYTHON=python"
    ) else (
        where py >nul 2>&1
        if !errorlevel! equ 0 (
            set "SYSTEM_PYTHON=py"
        )
    )

    if not defined SYSTEM_PYTHON (
        echo [ERROR] Python 3.10+ was not found in your system PATH.
        echo [ERROR] Please install Python 3.10+ from https://www.python.org and ensure
        echo         "Add python.exe to PATH" is checked during installation.
        if /i not "%~1"=="--check-only" pause
        exit /b 1
    )

    echo [INFO] Creating virtual environment using !SYSTEM_PYTHON!...
    !SYSTEM_PYTHON! -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment at %VENV_DIR%.
        if /i not "%~1"=="--check-only" pause
        exit /b !errorlevel!
    )
    echo [INFO] Virtual environment successfully created at %VENV_DIR%.
)

:: 2. Verify required dependencies idempotently
echo [INFO] Verifying installed dependencies...
"%VENV_PYTHON%" -c "import streamlit, pandas, yaml, plotly, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Required dependencies missing. Installing from requirements.txt...
    if exist "%PROJECT_ROOT%requirements.txt" (
        "%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to install dependencies from requirements.txt.
            if /i not "%~1"=="--check-only" pause
            exit /b !errorlevel!
        )
        echo [INFO] Dependencies installed successfully.
    ) else (
        echo [ERROR] requirements.txt not found at %PROJECT_ROOT%.
        if /i not "%~1"=="--check-only" pause
        exit /b 1
    )
) else (
    echo [INFO] Environment and dependencies verified.
)

:: 3. Support non-interactive validation mode
if /i "%~1"=="--check-only" (
    echo [INFO] Check-only validation passed.
    exit /b 0
)

:: 4. Launch Streamlit user interface
echo [INFO] Launching Autonomous Data Analyst on Streamlit...
if exist "%VENV_STREAMLIT%" (
    "%VENV_STREAMLIT%" run app.py
) else (
    "%VENV_PYTHON%" -m streamlit run app.py
)

set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% neq 0 (
    echo [ERROR] Streamlit exited with error code %EXIT_CODE%.
    pause
    exit /b %EXIT_CODE%
)

exit /b 0
