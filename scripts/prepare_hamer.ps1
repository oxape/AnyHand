# prepare_hamer.ps1
# Windows setup for HaMeR (third_party/hamer) + AnyHand checkpoint.
# Run from repo root in PowerShell:
#   .\scripts\prepare_hamer.ps1
#
# Prerequisites: Git, Python 3.10 venv at .venv, PyTorch (see requirements-oxape-cu128.txt)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Info($msg)  { Write-Host "[INFO]  $msg" }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Die($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# --- resolve Python (.venv on Windows) ---
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

function Ensure-Pip {
    & $Python -m pip --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Info "Bootstrapping pip..."
        & $Python -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { Die "pip is not available for $Python" }
    }
}

function Pip-Install {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Ensure-Pip
    & $Python -m pip install @Args
    if ($LASTEXITCODE -ne 0) { Die "pip install failed: $Args" }
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

# --- [1/5] HaMeR submodule ---
Info "=== [1/5] Initialising HaMeR submodule ==="

$HamerDir = Join-Path $RepoRoot "third_party\hamer"
$gitMeta  = Join-Path $HamerDir ".git"
if ((Test-Path $gitMeta) -or (Test-Path $HamerDir)) {
    Info "HaMeR submodule path present."
} else {
    Die "third_party/hamer missing. Run:  git submodule update --init --recursive"
}

$hasSetup = @("setup.py", "setup.cfg", "pyproject.toml") | ForEach-Object {
    Test-Path (Join-Path $HamerDir $_)
} | Where-Object { $_ }
if (-not $hasSetup) {
    Die "HaMeR directory appears empty. Run:  git submodule update --init third_party/hamer"
}

# --- [2/5] Install HaMeR package ---
Info "=== [2/5] Installing HaMeR Python package ==="

& $Python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Die @"
PyTorch is not installed. Install first (CUDA 12.8):
  .\.venv\Scripts\python.exe -m pip install -r requirements-oxape-cu128.txt `
    --index-url https://download.pytorch.org/whl/cu128
"@
}

Pip-Install -e "$HamerDir" --no-deps
Pip-Install @(
    "gdown", "numpy", "opencv-python", "pyrender", "pytorch-lightning", "scikit-image",
    "smplx==0.1.28", "yacs", "timm", "einops", "xtcocotools", "pandas",
    "chumpy @ git+https://github.com/mattloper/chumpy"
)
Info "HaMeR core installed."

# --- [3/5] ViTPose ---
Info "=== [3/5] Installing ViTPose backbone ==="

$VitposeDir = Join-Path $HamerDir "third-party\ViTPose"
if (-not (Test-Path $VitposeDir)) {
    Info "Initialising ViTPose nested submodule..."
    git -C $HamerDir submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { Die "git submodule update failed inside hamer" }
}

if ((Test-Path (Join-Path $VitposeDir "setup.py")) -or (Test-Path (Join-Path $VitposeDir "setup.cfg"))) {
    Pip-Install -v -e $VitposeDir
    Info "ViTPose installed."
} else {
    Warn "ViTPose not found at $VitposeDir"
    Warn "Try manually: $Python -m pip install -v -e $VitposeDir"
}

# --- [4/5] Checkpoints ---
Info "=== [4/5] Downloading AnyHand HaMeR checkpoint ==="

$CkptDir = Join-Path $RepoRoot "pretrained_models\hamer_ckpts\checkpoints"
$HfBase  = "https://huggingface.co/chen-si-02/AnyHand-Models/resolve/main"

Download-File "$HfBase/anyhand_hamer.ckpt"       (Join-Path $CkptDir "anyhand_hamer.ckpt")
Download-File "$HfBase/model_config_hamer.yaml"  (Join-Path $CkptDir "model_config.yaml")

# --- [5/5] mano_mean_params ---
Info "=== [5/5] Downloading HaMeR auxiliary data ==="

$DataDir = Join-Path $RepoRoot "pretrained_models\hamer_ckpts\data"
$ManoUrl = "https://huggingface.co/spaces/geopavlakos/hamer/resolve/main/_DATA/data/mano_mean_params.npz"
try {
    Download-File $ManoUrl (Join-Path $DataDir "mano_mean_params.npz")
} catch {
    Warn "Could not download mano_mean_params.npz — fetch manually if needed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  HaMeR setup complete."
Write-Host ""
Write-Host "  third_party\hamer\           <- HaMeR submodule"
Write-Host "  pretrained_models\hamer_ckpts\"
Write-Host "  ACTION: place MANO_RIGHT.pkl in mano_data\ (see README)"
Write-Host "============================================================"
