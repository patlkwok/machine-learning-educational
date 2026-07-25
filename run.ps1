<#
.SYNOPSIS
    Start the ML Playground on Windows. PowerShell equivalent of run.sh.

.EXAMPLE
    .\run.ps1
.EXAMPLE
    .\run.ps1 -Dev
.EXAMPLE
    .\run.ps1 -Port 9000
#>
[CmdletBinding()]
param(
    # Auto-reload the server when backend or frontend files change.
    [switch]$Dev,

    [int]$Port = 8000,

    # Named BindAddress rather than Host: $Host is a reserved PowerShell variable.
    [string]$BindAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Fail legibly if the script has been moved out of the project root; otherwise
# uvicorn reports this as a bare ModuleNotFoundError several frames deep.
if (-not (Test-Path (Join-Path $PSScriptRoot "backend\app\main.py"))) {
    Write-Host "run.ps1 must sit in the project root; backend\app\main.py was not found next to it." -ForegroundColor Red
    exit 1
}

# Prefer the project venv so the script works whether or not the shell activated it.
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $fallback = Get-Command python -ErrorAction SilentlyContinue
    if (-not $fallback) {
        Write-Host "No Python found. Create the virtual environment first:" -ForegroundColor Red
        Write-Host "    py -m venv .venv"
        Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
        exit 1
    }
    $python = $fallback.Source
    Write-Warning "No .venv found; falling back to $python"
}

# find_spec keeps this quiet: no traceback when a dependency is simply absent.
& $python -c "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ('fastapi', 'sklearn')) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing dependencies. Install them with:" -ForegroundColor Yellow
    Write-Host "    $python -m pip install -r requirements.txt"
    exit 1
}

$uvicornArgs = @(
    "-m", "uvicorn", "backend.app.main:app",
    "--host", $BindAddress,
    "--port", $Port
)
if ($Dev) {
    $uvicornArgs += @("--reload", "--reload-dir", "backend", "--reload-dir", "frontend")
}

Write-Host "ML Playground -> http://${BindAddress}:${Port}" -ForegroundColor Cyan
& $python @uvicornArgs
