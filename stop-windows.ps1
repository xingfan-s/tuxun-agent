[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$PidFile = Join-Path $PSScriptRoot ".run\windows-processes.json"
if (-not (Test-Path $PidFile)) {
    Write-Host "No Windows launcher processes are recorded."
    exit 0
}

$Processes = Get-Content $PidFile -Raw | ConvertFrom-Json
function Get-DescendantProcessIds([int]$RootId) {
    try {
        $Snapshot = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    }
    catch {
        return @()
    }

    $Found = [System.Collections.Generic.HashSet[int]]::new()
    $Pending = [System.Collections.Generic.Queue[int]]::new()
    $Pending.Enqueue($RootId)
    while ($Pending.Count -gt 0) {
        $ParentId = $Pending.Dequeue()
        foreach ($Child in @($Snapshot | Where-Object { $_.ParentProcessId -eq $ParentId })) {
            $ChildId = [int]$Child.ProcessId
            if ($Found.Add($ChildId)) {
                $Pending.Enqueue($ChildId)
            }
        }
    }
    return @($Found)
}

$FailedIds = [System.Collections.Generic.List[int]]::new()
foreach ($Id in @($Processes.frontend, $Processes.backend)) {
    if (-not $Id) { continue }
    $Process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if (-not $Process) { continue }

    $TreeIds = @((Get-DescendantProcessIds -RootId $Id) + @([int]$Id) | Select-Object -Unique)

    # taskkill can fail on one child in a Node process tree. Do not let its
    # stderr terminate this script; fall back to Stop-Process for every PID.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & taskkill.exe /PID $Id /T /F 2>&1
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    foreach ($TreeId in $TreeIds) {
        if (Get-Process -Id $TreeId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $TreeId -Force -ErrorAction SilentlyContinue
        }
    }

    $RemainingIds = @($TreeIds | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if ($RemainingIds.Count -gt 0) {
        foreach ($RemainingId in $RemainingIds) { $FailedIds.Add([int]$RemainingId) }
        Write-Warning "Could not stop process tree rooted at PID $Id."
    }
    else {
        Write-Host "Stopped PID $Id"
    }
}

if ($FailedIds.Count -eq 0) {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "TuXun Windows services stopped." -ForegroundColor Green
    exit 0
}

try {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $Principal = [Security.Principal.WindowsPrincipal]::new($Identity)
    $IsAdministrator = $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
catch {
    $IsAdministrator = $false
}

if (-not $IsAdministrator) {
    Write-Warning "Some processes require Administrator permission. Requesting a one-time UAC confirmation..."
    try {
        $EnvironmentVariables = [System.Environment]::GetEnvironmentVariables()
        $PathKeys = @($EnvironmentVariables.Keys | Where-Object { $_ -match "^(?i)path$" })
        if ($PathKeys.Count -gt 1) {
            $PathEntries = [System.Collections.Generic.List[string]]::new()
            $SeenPathEntries = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            foreach ($PathKey in $PathKeys) {
                foreach ($Entry in ([string]$EnvironmentVariables[$PathKey]).Split(";")) {
                    $TrimmedEntry = $Entry.Trim()
                    if ($TrimmedEntry -and $SeenPathEntries.Add($TrimmedEntry)) {
                        $PathEntries.Add($TrimmedEntry)
                    }
                }
            }
            [System.Environment]::SetEnvironmentVariable("PATH", $null, [System.EnvironmentVariableTarget]::Process)
            [System.Environment]::SetEnvironmentVariable("Path", ($PathEntries -join ";"), [System.EnvironmentVariableTarget]::Process)
        }
        $Elevated = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath
        )
        if ($Elevated.ExitCode -eq 0) {
            exit 0
        }
    }
    catch {
        Write-Warning "The Administrator retry was not started: $($_.Exception.Message)"
    }
}

throw "Some TuXun processes are still running (PIDs: $($FailedIds -join ', ')). Run stop-windows.cmd once from an Administrator PowerShell window."
