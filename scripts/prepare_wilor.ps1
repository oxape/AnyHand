# prepare_wilor.ps1
# Windows setup for WiLoR submodule, YOLO detector, and AnyHand WiLoR checkpoint.
# Run from repo root in PowerShell:
#   .\scripts\prepare_wilor.ps1
#
# Required even for HaMeR backend: AnyHand uses WiLoR's YOLO hand detector.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Info($msg)  { Write-Host "[INFO]  $msg" }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Die($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

$Python = $env:PYTHON
if (-not $Python) {
    $candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\bin\python")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $Python = $c; break }
    }
}
if (-not $Python) {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $Python) {
    Die "Python not found. Create a venv first:  python -m venv .venv"
}
Info "Using Python: $Python"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Die "git not found. Install Git for Windows first."
}

function Invoke-PythonCli {
    param([string[]]$CliArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & $Python @CliArgs 2>&1
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prev
    return [PSCustomObject]@{
        ExitCode = $exit
        Output   = $output
    }
}

function Ensure-Pip {
    $check = Invoke-PythonCli -CliArgs @('-m', 'pip', '--version')
    if ($check.ExitCode -ne 0) {
        Info "Bootstrapping pip..."
        $boot = Invoke-PythonCli -CliArgs @('-m', 'ensurepip', '--upgrade')
        if ($boot.ExitCode -ne 0) { Die "pip is not available for $Python" }
    }
}

function Pip-Install {
    param([string[]]$PipArgs)
    Ensure-Pip
    $result = Invoke-PythonCli -CliArgs (@('-m', 'pip', 'install') + $PipArgs)
    if ($result.ExitCode -ne 0) {
        if ($result.Output) { Write-Host ($result.Output | Out-String) }
        Die "pip install failed: $($PipArgs -join ' ')"
    }
}

function Download-File {
    param([string]$Url, [string]$Dest)
    if (Test-Path $Dest) {
        Info "Already exists, skipping: $Dest"
        return
    }
    $parent = Split-Path -Parent $Dest
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Info "Downloading: $Dest"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    } catch {
        Die "Download failed: $Url — $_"
    }
}

# --- [1/4] WiLoR submodule ---
Info "=== [1/4] Initialising WiLoR submodule ==="

$WilorDir = Join-Path $RepoRoot "WiLoR"
$gitMeta  = Join-Path $WilorDir ".git"
if ((Test-Path $gitMeta) -or (Test-Path $WilorDir)) {
    Info "WiLoR submodule path present."
} else {
    Die "WiLoR missing. Run:  git submodule update --init WiLoR"
}

$reqFile = Join-Path $WilorDir "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Die "WiLoR directory appears empty. Run:  git submodule update --init WiLoR"
}

# --- [2/4] WiLoR dependencies (includes ultralytics) ---
Info "=== [2/4] Installing WiLoR dependencies ==="

Pip-Install -PipArgs @('-r', $reqFile)
Info "WiLoR dependencies installed."

# --- [3/4] YOLO hand detector ---
Info "=== [3/4] Downloading WiLoR hand detector ==="

$PretrainedDir = Join-Path $RepoRoot "pretrained_models"
$WilorHf = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models"
Download-File "$WilorHf/detector.pt" (Join-Path $PretrainedDir "detector.pt")

# --- [4/4] AnyHand WiLoR checkpoint ---
Info "=== [4/4] Downloading AnyHand WiLoR checkpoint ==="

$HfBase = "https://huggingface.co/chen-si-02/AnyHand-Models/resolve/main"
Download-File "$HfBase/anyhand_wilor.ckpt"      (Join-Path $PretrainedDir "anyhand_wilor.ckpt")
Download-File "$HfBase/model_config_wilor.yaml" (Join-Path $PretrainedDir "model_config_wilor.yaml")

Write-Host ""
Write-Host "============================================================"
Write-Host "  WiLoR setup complete."
Write-Host ""
Write-Host "  pretrained_models\"
Write-Host "  ├── detector.pt              <- YOLO hand detector (required for HaMeR too)"
Write-Host "  ├── anyhand_wilor.ckpt       <- AnyHand fine-tuned WiLoR"
Write-Host "  └── model_config_wilor.yaml  <- matching config"
Write-Host ""
Write-Host "  ACTION: place MANO_RIGHT.pkl in mano_data\ (see README)"
Write-Host "============================================================"
