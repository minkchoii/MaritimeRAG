# Opens two PowerShell windows: live batch log + progress summary.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$log = Join-Path $root "data\processed\logs\pilot_batch.log"
if (-not (Test-Path $log)) {
    New-Item -ItemType File -Path $log -Force | Out-Null
    "Waiting for batch output..." | Set-Content $log -Encoding utf8
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; Write-Host '=== Pilot batch log (tail) ===' -ForegroundColor Cyan; Get-Content '$log' -Wait -Tail 35"
)

Start-Sleep -Milliseconds 400

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; python scripts/watch_pilot_progress.py --interval 5"
)

Write-Host "Opened 2 monitor windows (log tail + progress summary)."
