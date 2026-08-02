[CmdletBinding()]
param(
    [switch]$CoreOnly,
    [switch]$SkipFrontend,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $Root
    )
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Stop-ProjectFrontendProcesses {
    $ProcessIds = [System.Collections.Generic.HashSet[int]]::new()

    foreach ($Process in @(Get-Process -Name "esbuild" -ErrorAction SilentlyContinue)) {
        try {
            if ($Process.Path -and $Process.Path.StartsWith($Frontend, [StringComparison]::OrdinalIgnoreCase)) {
                [void]$ProcessIds.Add($Process.Id)
            }
        }
        catch { }
    }

    try {
        foreach ($Process in @(Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction Stop)) {
            if ($Process.CommandLine -and $Process.CommandLine.IndexOf($Frontend, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                [void]$ProcessIds.Add([int]$Process.ProcessId)
            }
        }
    }
    catch {
        Write-Verbose "Could not inspect node.exe command lines: $($_.Exception.Message)"
    }

    foreach ($ProcessId in $ProcessIds) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { continue }
        Write-Host "Stopping project frontend process PID $ProcessId..." -ForegroundColor DarkGray
        & taskkill.exe /PID $ProcessId /T /F *> $null
        if ($LASTEXITCODE -ne 0 -and (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            Write-Warning "Could not stop PID $ProcessId. Close the running frontend before continuing."
        }
    }
}

function Invoke-NpmCi {
    param([Parameter(Mandatory = $true)][string]$NpmPath)

    Push-Location $Frontend
    try {
        for ($Attempt = 1; $Attempt -le 2; $Attempt++) {
            $NpmOutput = @(& $NpmPath ci 2>&1)
            $ExitCode = $LASTEXITCODE
            $NpmOutput | ForEach-Object { Write-Host $_ }
            if ($ExitCode -eq 0) { return }

            $FailureText = $NpmOutput -join [Environment]::NewLine
            if ($Attempt -eq 1 -and $FailureText -match "(?i)\\bEPERM\\b") {
                Write-Warning "npm found a locked file. Stopping this project's frontend processes and retrying once..."
                Stop-ProjectFrontendProcesses
                Start-Sleep -Seconds 2
                continue
            }

            if ($FailureText -match "(?i)\\bEPERM\\b") {
                throw "npm ci failed because Windows is still using a file in frontend\\node_modules. Close terminals or editors running this frontend, temporarily allow the project folder in antivirus if needed, then run setup-windows.cmd again."
            }
            throw "Command failed ($ExitCode): $NpmPath ci"
        }
    }
    finally {
        Pop-Location
    }
}

function Resolve-Python311 {
    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Launcher) {
        & $Launcher.Source -3.11 -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return @{ File = $Launcher.Source; Prefix = @("-3.11") }
        }
    }

    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Python) {
        $Version = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $Version.Trim() -eq "3.11") {
            return @{ File = $Python.Source; Prefix = @() }
        }
    }

    throw "Python 3.11 x64 was not found. Install it from python.org and enable the py launcher."
}

Write-Host "[1/5] Checking Windows prerequisites..." -ForegroundColor Cyan
$Python311 = Resolve-Python311
$Node = Get-Command node.exe -ErrorAction SilentlyContinue
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Node -or -not $Npm) {
    throw "Node.js 20+ was not found. Install the Node.js LTS x64 package first."
}
$NodeMajor = [int]((& $Node.Source --version).TrimStart("v").Split(".")[0])
if ($NodeMajor -lt 20) {
    throw "Node.js 20+ is required. Current version: $(& $Node.Source --version)"
}
if ($CheckOnly) {
    Write-Host "Python 3.11 and Node.js $NodeMajor are available." -ForegroundColor Green
    exit 0
}

$PidFile = Join-Path $Root ".run\windows-processes.json"
if (Test-Path $PidFile) {
    Write-Host "Stopping services from the previous Windows launch..." -ForegroundColor DarkGray
    & (Join-Path $Root "stop-windows.ps1")
}
Stop-ProjectFrontendProcesses

if (-not (Test-Path $VenvPython)) {
    Write-Host "[2/5] Creating .venv with Python 3.11..." -ForegroundColor Cyan
    $VenvArgs = @() + $Python311.Prefix + @("-m", "venv", $Venv)
    Invoke-Checked -FilePath $Python311.File -Arguments $VenvArgs
}
else {
    Write-Host "[2/5] Reusing existing .venv." -ForegroundColor DarkGray
}

Write-Host "[3/5] Installing core backend dependencies..." -ForegroundColor Cyan
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements-windows.txt") -WorkingDirectory $Backend

$OptionalFailures = @()
if (-not $CoreOnly) {
    Write-Host "[4/5] Installing optional AI/search capabilities..." -ForegroundColor Cyan
    $OptionalFile = Join-Path $Backend "requirements-optional.txt"
    $ConstraintsFile = Join-Path $Backend "requirements-constraints.txt"
    foreach ($Line in [IO.File]::ReadAllLines($OptionalFile, [Text.Encoding]::UTF8)) {
        $Package = $Line.Trim()
        if (-not $Package -or $Package.StartsWith("#")) { continue }
        Write-Host "  Installing $Package"
        & $VenvPython -m pip install --constraint $ConstraintsFile $Package
        if ($LASTEXITCODE -ne 0) {
            $OptionalFailures += $Package
            Write-Warning "$Package is unavailable on this Windows/Python build. Its feature will be disabled at runtime."
        }
    }
}
else {
    Write-Host "[4/5] Optional AI dependencies skipped (-CoreOnly)." -ForegroundColor DarkGray
}

if (-not $SkipFrontend) {
    Write-Host "[5/5] Installing frontend dependencies..." -ForegroundColor Cyan
    Invoke-NpmCi -NpmPath $Npm.Source
}
else {
    Write-Host "[5/5] Frontend dependencies skipped." -ForegroundColor DarkGray
}

foreach ($Directory in @(
    (Join-Path $Backend "uploads"),
    (Join-Path $Backend "data\geo_image_db_v2"),
    (Join-Path $Root ".run")
)) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

$EnvFile = Join-Path $Backend ".env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Backend ".env.example") $EnvFile
    Write-Warning "Created backend\.env. Add QWEN_API_KEY and map/search keys before analysis."
}

Write-Host ""
Write-Host "Windows setup completed." -ForegroundColor Green
if ($OptionalFailures.Count -gt 0) {
    Write-Warning "Optional packages not installed: $($OptionalFailures -join ', ')"
    Write-Host "The application can start, but those model/search stages will report unavailable."
}
Write-Host "Next: edit backend\.env, then run start-windows.cmd"
