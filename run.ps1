<#
.SYNOPSIS
    Autonomous Data Analyst -- Windows PowerShell Launcher

.DESCRIPTION
    Determines project root from script location, detects or provisions a local virtual
    environment, verifies dependencies idempotently, and launches Streamlit.

.NOTES
    PowerShell Execution Policy:
    If running this script is blocked by execution policy on your system, do NOT modify
    machine-wide policy. Instead, invoke PowerShell for this session only:
        powershell -ExecutionPolicy Bypass -File .\run.ps1
    Or run the accompanying batch launcher:
        .\run.bat

.PARAMETER CheckOnly
    Verifies the virtual environment and dependencies without launching Streamlit.

.PARAMETER ShowHelp
    Displays usage and execution policy guidance.
#>

[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$ShowHelp
)

if ($ShowHelp) {
    Write-Host "Autonomous Data Analyst -- PowerShell Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\run.ps1 [-CheckOnly] [-ShowHelp]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -CheckOnly  Verify environment and dependencies without launching UI."
    Write-Host "  -ShowHelp   Display this help message."
    Write-Host ""
    Write-Host "Execution Policy Guidance:"
    Write-Host "  If execution is blocked by system policy, run without altering persistent settings:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File .\run.ps1"
    Write-Host "  Or simply execute the batch wrapper:"
    Write-Host "    .\run.bat"
    return
}

# Determine project directory from script location
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = (Get-Location).Path
}

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvStreamlit = Join-Path $VenvDir "Scripts\streamlit.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$AppFile = Join-Path $ProjectRoot "app.py"

# 1. Detect or create virtual environment
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "[INFO] Virtual environment not detected at $VenvDir" -ForegroundColor Cyan
    Write-Host "[INFO] Searching for system Python..." -ForegroundColor Cyan

    $systemPython = $null
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $systemPython = "python"
    } else {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            $systemPython = "py"
        }
    }

    if (-not $systemPython) {
        Write-Error "[ERROR] Python 3.10+ was not found in PATH."
        Write-Host "Please install Python 3.10+ from https://www.python.org and ensure it is added to your PATH." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "[INFO] Creating virtual environment using $systemPython..." -ForegroundColor Cyan
    & $systemPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        Write-Error "[ERROR] Failed to create virtual environment at $VenvDir"
        exit 1
    }
    Write-Host "[INFO] Virtual environment created successfully." -ForegroundColor Green
}

# 2. Verify dependencies idempotently
Write-Host "[INFO] Verifying installed dependencies..." -ForegroundColor Cyan
& $VenvPython -c "import streamlit, pandas, yaml, plotly, sklearn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Missing required dependencies. Installing from requirements.txt..." -ForegroundColor Yellow
    if (Test-Path -LiteralPath $RequirementsFile) {
        & $VenvPython -m pip install -r $RequirementsFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "[ERROR] Failed to install dependencies from requirements.txt"
            exit $LASTEXITCODE
        }
        Write-Host "[INFO] Dependencies installed successfully." -ForegroundColor Green
    } else {
        Write-Error "[ERROR] requirements.txt not found at $RequirementsFile"
        exit 1
    }
} else {
    Write-Host "[INFO] Environment and dependencies verified." -ForegroundColor Green
}

# 3. Support non-interactive validation mode
if ($CheckOnly) {
    Write-Host "[INFO] Check-only validation passed." -ForegroundColor Green
    return
}

# 4. Launch Streamlit
Write-Host "[INFO] Launching Autonomous Data Analyst on Streamlit..." -ForegroundColor Green
if (Test-Path -LiteralPath $VenvStreamlit) {
    & $VenvStreamlit run $AppFile
} else {
    & $VenvPython -m streamlit run $AppFile
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] Streamlit exited with error code $LASTEXITCODE"
    exit $LASTEXITCODE
}
