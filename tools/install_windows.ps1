param(
    [switch]$NoLaunch
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
        [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
    }
}

function Show-InstallerMessage(
    [string]$Message,
    [System.Windows.Forms.MessageBoxIcon]$Icon
) {
    [System.Windows.Forms.MessageBox]::Show(
        $Message,
        "Blind Men Elephant Model",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        $Icon
    ) | Out-Null
}

function Stop-Installer([string]$Message) {
    Show-InstallerMessage $Message ([System.Windows.Forms.MessageBoxIcon]::Error)
    Write-Error $Message
    exit 1
}

function Test-CompatiblePython([string]$Candidate) {
    if (
        [string]::IsNullOrWhiteSpace($Candidate) -or
        -not (Test-Path -LiteralPath $Candidate)
    ) {
        return $false
    }
    try {
        & $Candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-CompatiblePython([string]$ProjectRoot) {
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    )
    foreach ($root in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        (Join-Path $env:LOCALAPPDATA "Python"),
        (Join-Path $env:ProgramFiles "Python")
    )) {
        if (Test-Path -LiteralPath $root) {
            $candidates += @(
                Get-ChildItem `
                    -LiteralPath $root `
                    -Filter "python.exe" `
                    -Recurse `
                    -File `
                    -ErrorAction SilentlyContinue |
                    Sort-Object FullName -Descending |
                    Select-Object -ExpandProperty FullName
            )
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (
        $null -ne $pythonCommand -and
        $pythonCommand.Source -notlike "*\WindowsApps\*"
    ) {
        $candidates += $pythonCommand.Source
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        try {
            $launcherPython = & $launcher.Source -3 -c "import sys; print(sys.executable)"
            if ($LASTEXITCODE -eq 0) {
                $candidates += [string]$launcherPython
            }
        }
        catch {
            # Continue through the explicit installation paths.
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-CompatiblePython ([string]$candidate)) {
            return [string]$candidate
        }
    }
    return $null
}

Repair-PathEnvironment
Add-Type -AssemblyName System.Windows.Forms

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$virtualEnvironment = Join-Path $projectRoot ".venv"
$virtualPython = Join-Path $virtualEnvironment "Scripts\python.exe"
$productName = -join @(
    [char]0x76F2,
    [char]0x4EBA,
    [char]0x6478,
    [char]0x8C61,
    [char]0x6A21,
    [char]0x578B
)

Write-Host "Blind Men Elephant Model setup" -ForegroundColor Cyan
Write-Host "Checking Python 3.10 or newer..." -ForegroundColor DarkGray

$pythonPath = Find-CompatiblePython $projectRoot
if ($null -eq $pythonPath) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        Stop-Installer "Python 3.10 or newer is required. Install Python from python.org, then run install-windows.cmd again."
    }
    $choice = [System.Windows.Forms.MessageBox]::Show(
        "Python 3.10 or newer is required. Install Python 3.12 for the current Windows user now?",
        "Blind Men Elephant Model",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if ($choice -ne [System.Windows.Forms.DialogResult]::Yes) {
        exit 1
    }
    Write-Host "Installing Python 3.12 with Windows Package Manager..." -ForegroundColor Cyan
    & $winget.Source install `
        --id Python.Python.3.12 `
        --exact `
        --scope user `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        Stop-Installer "Windows Package Manager could not install Python. Install Python 3.10 or newer manually, then run this installer again."
    }
    Start-Sleep -Seconds 2
    $pythonPath = Find-CompatiblePython $projectRoot
    if ($null -eq $pythonPath) {
        Stop-Installer "Python was installed but could not yet be found. Sign out and back in, then run install-windows.cmd again."
    }
}

if (-not (Test-CompatiblePython $virtualPython)) {
    Write-Host "Creating the private Python runtime..." -ForegroundColor Cyan
    & $pythonPath -m venv $virtualEnvironment
    if ($LASTEXITCODE -ne 0 -or -not (Test-CompatiblePython $virtualPython)) {
        Stop-Installer "The private Python runtime could not be created."
    }
}

try {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "$productName.lnk"
    $powershellPath = Join-Path $PSHOME "powershell.exe"
    $startScript = Join-Path $projectRoot "tools\start_web.ps1"
    $iconPath = Join-Path $projectRoot "homepage\assets\blind-men-elephant-model.ico"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powershellPath
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`""
    $shortcut.WorkingDirectory = $projectRoot
    if (Test-Path -LiteralPath $iconPath) {
        $shortcut.IconLocation = "$iconPath,0"
    }
    else {
        $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,14"
    }
    $shortcut.Description = "Start $productName"
    $shortcut.Save()
}
catch {
    Stop-Installer "The model is ready, but the desktop shortcut could not be created: $($_.Exception.Message)"
}

Write-Host "Installation complete." -ForegroundColor Green
Show-InstallerMessage "Installation complete. A desktop shortcut named $productName has been created. The browser will ask only for your API Key." ([System.Windows.Forms.MessageBoxIcon]::Information)

if (-not $NoLaunch) {
    & (Join-Path $PSHOME "powershell.exe") `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File (Join-Path $projectRoot "tools\start_web.ps1")
}
