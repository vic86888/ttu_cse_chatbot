Write-Host "======================================" -ForegroundColor Cyan
Write-Host "TTU CSE Chatbot - Frontend Server" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

npm run dev
