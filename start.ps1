# ARIA ver.2 - Start Backend + Frontend
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPort = 8200
$frontendPort = 5274

# Force UTF-8 encoding for Python and Powershell Console to prevent corrupted characters on Windows
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $root

Write-Host "=== ARIA ver.2 Patent Report Generator ===" -ForegroundColor Cyan

Write-Host "[1/3] Checking Python dependencies..." -ForegroundColor Yellow
python -m pip install -r backend/requirements.txt --quiet

Write-Host "[2/3] Starting FastAPI backend (port $backendPort)..." -ForegroundColor Yellow
$backend = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "$backendPort", "--reload" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Minimized
Write-Host "  Backend PID: $($backend.Id)" -ForegroundColor Green

Write-Host "  Waiting for backend to become ready..." -ForegroundColor Gray
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (-not (Get-Process -Id $backend.Id -ErrorAction SilentlyContinue)) {
        throw "Backend process exited before it became ready. Run this command to see the error: python -m uvicorn backend.main:app --host 0.0.0.0 --port $backendPort"
    }
    try {
        Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$backendPort/" -TimeoutSec 2 | Out-Null
        $backendReady = $true
        break
    } catch {
        # Keep waiting; uvicorn reload can take a few seconds on first startup.
    }
}

if (-not $backendReady) {
    throw "Backend did not respond on http://127.0.0.1:$backendPort within 30 seconds."
}

Write-Host "[3/3] Starting React frontend (port $frontendPort)..." -ForegroundColor Yellow

$nodeCheck = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

Set-Location "$root\frontend"
if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

$frontend = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory "$root\frontend" `
    -PassThru -WindowStyle Minimized
Write-Host "  Frontend PID: $($frontend.Id)" -ForegroundColor Green

Set-Location $root

Write-Host ""
Write-Host "=== Started ===" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:$frontendPort" -ForegroundColor White
Write-Host "  Backend:  http://localhost:$backendPort" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop..." -ForegroundColor Gray

try {
    while ($true) {
        Start-Sleep -Seconds 2
        $bAlive = Get-Process -Id $backend.Id -ErrorAction SilentlyContinue
        $fAlive = Get-Process -Id $frontend.Id -ErrorAction SilentlyContinue
        if (-not $bAlive -and -not $fAlive) { break }
    }
} finally {
    Write-Host "Stopping processes..." -ForegroundColor Yellow
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $frontend.Id -ErrorAction SilentlyContinue
    Write-Host "Stopped." -ForegroundColor Green
}
