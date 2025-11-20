# TTU CSE Chatbot Startup Script

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   TTU CSE Chatbot - Starting..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Start backend with venv
Write-Host "Starting Backend Server (with venv)..." -ForegroundColor Green
$backendScript = @"
Set-Location 'd:\畢業專題\ttu_cse_chatbot\backend'
& 'd:\畢業專題\ttu_cse_chatbot\venv\Scripts\Activate.ps1'
python main.py
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript

Start-Sleep -Seconds 3

# Start frontend
Write-Host "Starting Frontend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location 'd:\畢業專題\ttu_cse_chatbot\frontend'; npm run dev"

Write-Host ""
Write-Host "=====================================" -ForegroundColor Yellow
Write-Host "Both servers are starting..." -ForegroundColor Yellow
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C in each terminal to stop the servers" -ForegroundColor Gray
