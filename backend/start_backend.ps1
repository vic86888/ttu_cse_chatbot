Write-Host "======================================" -ForegroundColor Cyan
Write-Host "TTU CSE Chatbot - Backend Server" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $scriptPath

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$projectRoot\venv\Scripts\Activate.ps1"

# Install requirements if needed
if (-not (python -c "import fastapi" 2>$null)) {
    Write-Host "Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Starting backend server..." -ForegroundColor Green
python main.py
