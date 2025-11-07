param(
  [int]$MaxImages = 200,
  [int]$BatchSizeDet = 16,
  [string]$DetectorName = "yolov11_combo",
  [string]$PlateModelPath = "",
  [string]$OrigDir = "",
  [string]$GtAnnotations = "",
  [string]$RepoDir = ""
)

$ErrorActionPreference = 'Stop'

# Resolve repo dir
if (-not $RepoDir) { $RepoDir = (Get-Location).Path }
Write-Host "[INFO] REPO_DIR = $RepoDir" -ForegroundColor Cyan

# Reduce pip disk usage and redirect temp to repo folder (avoid C: full)
$tmpDir = Join-Path $RepoDir ".tmp"
if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null }
$env:PIP_NO_CACHE_DIR = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:TEMP = $tmpDir
$env:TMP = $tmpDir

$venv = Join-Path $RepoDir ".venv"
$py = Join-Path $venv "Scripts/python.exe"
$pip = Join-Path $venv "Scripts/pip.exe"

if (-not (Test-Path $venv)) {
  Write-Host "[INFO] Creating venv ..." -ForegroundColor Cyan
  python -m venv "$venv"
}

Write-Host "[INFO] Upgrading pip/wheel/setuptools ..." -ForegroundColor Cyan
& "$py" -m pip install -U pip wheel setuptools | Out-Host

Write-Host "[INFO] Pin numpy/opencv and install core deps ..." -ForegroundColor Cyan
& "$pip" uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless | Out-Null
& "$pip" install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78" | Out-Host

# Try Torch CUDA first, fallback to CPU if fails
try {
  Write-Host "[INFO] Installing Torch (CUDA 12.1) ..." -ForegroundColor Cyan
  & "$pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 | Out-Host
} catch {
  Write-Warning "CUDA Torch install failed, fallback to CPU ..."
  & "$pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu | Out-Host
}

Write-Host "[INFO] Installing Python deps ..." -ForegroundColor Cyan
if ($DetectorName -eq 'yolov8_face') {
  # Faces-only: minimal deps
  & "$pip" install insightface onnxruntime pillow pandas tqdm | Out-Host
} else {
  & "$pip" install ultralytics insightface onnxruntime-gpu onnxruntime pillow pandas tqdm | Out-Host
}
# Re-pin
& "$pip" install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78" | Out-Host

# Sanity print
& "$py" -c "import numpy as np, importlib; print('NUMPY VERSION:', np.__version__); cv2=importlib.import_module('cv2'); print('CV2 VERSION:', cv2.__version__)"

# Prepare env and run
$env:REPO_DIR = $RepoDir
$env:MAX_IMAGES = $MaxImages
$env:BATCH_SIZE_DET = $BatchSizeDet
$env:DETECTOR_NAME = $DetectorName
if ($PlateModelPath) { $env:PLATE_MODEL_PATH = $PlateModelPath }
if ($OrigDir) { $env:ORIG_DIR = $OrigDir }
if ($GtAnnotations) { $env:GT_ANNOTATIONS = $GtAnnotations }

# Prepare anonymized dirs (keep minimal set; non-existing will be skipped)
$anon = @(
  Join-Path $RepoDir 'outputs\baselines\pixelation_16x16')
$anon += (Join-Path $RepoDir 'outputs\baselines\gaussian_k15')
$env:ANON_DIRS_JOINED = ($anon -join '|')

Write-Host "[INFO] Running local quick eval ..." -ForegroundColor Cyan
& "$py" (Join-Path $RepoDir 'scripts\local_eval.py') | Tee-Object -FilePath (Join-Path $RepoDir 'outputs\local_run.log')

Write-Host "[INFO] Done. Artifacts at outputs/preds/* and outputs/metrics_all.csv" -ForegroundColor Green
