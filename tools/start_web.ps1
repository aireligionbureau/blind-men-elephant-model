param(
    [switch]$NoBrowser,
    [ValidateSet("baseline", "candidate")]
    [string]$ExecutionProfile = "candidate"
)

$ErrorActionPreference = "Stop"

function Repair-PathEnvironment {
    $variables = [Environment]::GetEnvironmentVariables("Process")
    $pathKeys = @(
        $variables.Keys |
            Where-Object {
                [string]::Equals(
                    [string]$_,
                    "Path",
                    [StringComparison]::OrdinalIgnoreCase
                )
            }
    )
    if ($pathKeys.Count -le 1) {
        return
    }

    $pathValue = $pathKeys |
        ForEach-Object { [string]$variables[$_] } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Sort-Object Length -Descending |
        Select-Object -First 1
    foreach ($pathKey in $pathKeys) {
        [Environment]::SetEnvironmentVariable(
            [string]$pathKey,
            $null,
            "Process"
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($pathValue)) {
        [Environment]::SetEnvironmentVariable(
            "Path",
            $pathValue,
            "Process"
        )
    }
}

Repair-PathEnvironment

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"
$hostAddress = "127.0.0.1"
$candidatePorts = 8766..8775
$requiredRuntimeRevision = "2026-09-17-model-choice-v20"

foreach ($name in @("BME_LLM_API_KEY", "DEEPSEEK_API_KEY")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
        $storedValue = [Environment]::GetEnvironmentVariable($name, "User")
        if (-not [string]::IsNullOrWhiteSpace($storedValue)) {
            [Environment]::SetEnvironmentVariable($name, $storedValue, "Process")
        }
    }
}
$requestedPipelineMode = if ($ExecutionProfile -eq "candidate") {
    "streaming_v2"
}
else {
    "sequential"
}
$requestedSynthesisMode = if ($ExecutionProfile -eq "candidate") {
    "optimized_v2"
}
else {
    "legacy_sequential"
}
$requestedRunsNamespace = if ($ExecutionProfile -eq "candidate") {
    "runs-candidate"
}
else {
    "runs"
}

Write-Host "Starting Blind Men Elephant Model. Please wait..." -ForegroundColor Cyan

function Get-BmeHealth([int]$Port) {
    try {
        $health = Invoke-RestMethod `
            -Uri "http://${hostAddress}:$Port/api/health" `
            -TimeoutSec 6
        if (
            $health.status -eq "ok" -and
            $health.service -eq "blind-men-elephant-model"
        ) {
            return $health
        }
    }
    catch {
        return $null
    }
    return $null
}

function Test-LocalPortFree([int]$Port) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $Port
        )
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

$port = $null
foreach ($candidate in $candidatePorts) {
    $health = Get-BmeHealth $candidate
    if (
        $null -ne $health -and
        $health.runtime_revision -eq $requiredRuntimeRevision -and
        [string]::Equals(
            [string]$health.project_root,
            [string]$projectRoot,
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        $health.pipeline_mode -eq $requestedPipelineMode -and
        $health.synthesis_mode -eq $requestedSynthesisMode -and
        $health.runs_namespace -eq $requestedRunsNamespace
    ) {
        $port = $candidate
        break
    }
}

if ($null -eq $port) {
    foreach ($candidate in $candidatePorts) {
        if (Test-LocalPortFree $candidate) {
            $port = $candidate
            break
        }
    }
}

if ($null -eq $port) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Ports 8766-8775 are occupied. The application cannot start.",
        "Blind Men Elephant Model"
    ) | Out-Null
    exit 1
}

if ($null -eq (Get-BmeHealth $port)) {
    $pythonCandidates = @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe")
    )
    $programsPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path $programsPythonRoot) {
        $pythonCandidates += @(
            Get-ChildItem `
                -Path $programsPythonRoot `
                -Filter "python.exe" `
                -Recurse `
                -File `
                -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending |
                Select-Object -ExpandProperty FullName
        )
    }
    $localPythonRoot = Join-Path $env:LOCALAPPDATA "Python"
    if (Test-Path $localPythonRoot) {
        $pythonCandidates += @(
            Get-ChildItem `
                -Path $localPythonRoot `
                -Filter "python.exe" `
                -Recurse `
                -File `
                -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending |
                Select-Object -ExpandProperty FullName
        )
    }
    $pythonCandidates += Join-Path `
        $env:USERPROFILE `
        ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (
        $null -ne $pythonCommand -and
        $pythonCommand.Source -notlike "*\WindowsApps\*"
    ) {
        $pythonCandidates += $pythonCommand.Source
    }
    $pythonPath = $pythonCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if ($null -eq $pythonPath) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            "A real Python installation could not be found. Run install-windows.cmd first.",
            "Blind Men Elephant Model"
        ) | Out-Null
        exit 1
    }
    $arguments = @(
        "-m",
        "bme_model",
        "serve",
        "--host",
        $hostAddress,
        "--port",
        "$port",
        "--provider",
        "three-layer",
        "--runs-dir",
        $requestedRunsNamespace,
        "--pipeline-mode",
        $requestedPipelineMode,
        "--synthesis-mode",
        $requestedSynthesisMode
    )
    $logDirectory = Join-Path $projectRoot "logs"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stdoutLog = Join-Path $logDirectory "service.$port.stdout.log"
    $stderrLog = Join-Path $logDirectory "service.$port.stderr.log"
    Write-Host "Starting the local analysis service. A cold start usually takes 10-20 seconds..." -ForegroundColor DarkGray
    Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Milliseconds 300
        $health = Get-BmeHealth $port
    } while ($null -eq $health -and (Get-Date) -lt $deadline)

    if ($null -eq $health) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            "The local service did not become ready within 60 seconds. Error log: " + $stderrLog,
            "Blind Men Elephant Model"
        ) | Out-Null
        exit 1
    }
}

$url = "http://${hostAddress}:$port/"
if ($NoBrowser) {
    Write-Output $url
}
else {
    Write-Host "Ready. Opening $url" -ForegroundColor Green
    Start-Process $url
}
