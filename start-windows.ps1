[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipSetup,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$RunDir = Join-Path $Root ".run"
$PidFile = Join-Path $RunDir "windows-processes.json"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BackendPort = 8000

function Test-PortAvailable([int]$Port) {
    $Listener = [System.Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    try {
        $Listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        try { $Listener.Stop() } catch { }
    }
}

function Test-HttpEndpoint([string]$Url) {
    try {
        $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Wait-Http([string]$Url, [System.Diagnostics.Process]$Process, [int]$Seconds) {
    for ($Attempt = 0; $Attempt -lt $Seconds; $Attempt++) {
        $Process.Refresh()
        if ($Process.HasExited) { return $false }
        try {
            $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) { return $true }
        }
        catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Repair-DuplicateProcessPath {
    $Variables = [System.Environment]::GetEnvironmentVariables()
    $PathKeys = @($Variables.Keys | Where-Object { $_ -match "^(?i)path$" })
    if ($PathKeys.Count -le 1) { return }

    $Entries = [System.Collections.Generic.List[string]]::new()
    $Seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($PathKey in @($PathKeys | Sort-Object { if ($_ -ceq "Path") { 0 } else { 1 } })) {
        foreach ($Entry in ([string]$Variables[$PathKey]).Split(";")) {
            $Trimmed = $Entry.Trim()
            if ($Trimmed -and $Seen.Add($Trimmed)) {
                $Entries.Add($Trimmed)
            }
        }
    }

    [System.Environment]::SetEnvironmentVariable("PATH", $null, [System.EnvironmentVariableTarget]::Process)
    [System.Environment]::SetEnvironmentVariable("Path", ($Entries -join ";"), [System.EnvironmentVariableTarget]::Process)
}

Repair-DuplicateProcessPath

if ((-not (Test-Path $VenvPython)) -or (-not (Test-Path (Join-Path $Frontend "node_modules")))) {
    if ($SkipSetup) {
        throw "Dependencies are missing. Run setup-windows.cmd first."
    }
    & (Join-Path $Root "setup-windows.ps1")
}

$BackendPortAvailable = Test-PortAvailable $BackendPort
$FrontendPortAvailable = Test-PortAvailable $FrontendPort
if (-not $BackendPortAvailable -or -not $FrontendPortAvailable) {
    $ExistingBackendReady = Test-HttpEndpoint "http://127.0.0.1:$BackendPort/health/live"
    $ExistingFrontendReady = Test-HttpEndpoint "http://127.0.0.1:$FrontendPort/"
    if ($ExistingBackendReady -and $ExistingFrontendReady) {
        $Url = "http://127.0.0.1:$FrontendPort/"
        Write-Host "TuXun is already running: $Url" -ForegroundColor Green
        Write-Host "Stop with: stop-windows.cmd"
        if (-not $NoBrowser) {
            Start-Process $Url
        }
        exit 0
    }
    if (-not $BackendPortAvailable) {
        throw "Port $BackendPort is already in use by another or unhealthy service. Stop it before starting TuXun."
    }
    throw "Port $FrontendPort is already in use by another or unhealthy service. Stop it or pass -FrontendPort."
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackendOut = Join-Path $RunDir "backend-$Timestamp.log"
$BackendErr = Join-Path $RunDir "backend-$Timestamp.error.log"
$FrontendOut = Join-Path $RunDir "frontend-$Timestamp.log"
$FrontendErr = Join-Path $RunDir "frontend-$Timestamp.error.log"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Starting backend on http://127.0.0.1:$BackendPort ..." -ForegroundColor Cyan
$BackendProcess = Start-Process -FilePath $VenvPython `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
    -WorkingDirectory $Backend -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr

Write-Host "Starting frontend on http://127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
$Npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$FrontendProcess = Start-Process -FilePath $Npm `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
    -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $FrontendOut -RedirectStandardError $FrontendErr

@{
    backend = $BackendProcess.Id
    frontend = $FrontendProcess.Id
    started_at = (Get-Date).ToString("o")
    backend_port = $BackendPort
    frontend_port = $FrontendPort
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

try {
    if (-not (Wait-Http "http://127.0.0.1:$BackendPort/health/live" $BackendProcess 180)) {
        throw "Backend did not become ready. See $BackendErr"
    }
    if (-not (Wait-Http "http://127.0.0.1:$FrontendPort/" $FrontendProcess 60)) {
        throw "Frontend did not become ready. See $FrontendErr"
    }
}
catch {
    foreach ($Process in @($BackendProcess, $FrontendProcess)) {
        try { Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue } catch { }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    throw
}

$Url = "http://127.0.0.1:$FrontendPort/"
Write-Host ""
Write-Host "TuXun is running: $Url" -ForegroundColor Green
Write-Host "Logs: $RunDir"
Write-Host "Stop with: stop-windows.cmd"
if (-not $NoBrowser) {
    Start-Process $Url
}
