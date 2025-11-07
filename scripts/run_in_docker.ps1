param(
    [int]$MaxImages = 2000,
    [int]$BatchSizeDet = 16,
    [string]$DetectorName = "yolov8_combo",
    [int]$EnableDeepPrivacy2 = 0,
    [int]$EnableLDFA = 0
)

$ErrorActionPreference = 'Stop'

Write-Host "[INFO] Checking Docker..." -ForegroundColor Cyan
$dockerVersion = (& docker --version) 2>$null
if (-not $dockerVersion) {
    Write-Error "Docker 未安装或未加入 PATH，请先安装 Docker Desktop 并启用 GPU 支持。"
}

# Optional: quick NVIDIA check inside docker
try {
    Write-Host "[INFO] Checking GPU availability in Docker (nvidia-smi) ..." -ForegroundColor Cyan
    & docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04 nvidia-smi | Out-Host
} catch {
    Write-Warning "无法在 docker 中运行 nvidia-smi。请确保：Docker Desktop 已启用 GPU，加速器驱动与 WSL2 驱动安装正确。"
}

# Build image
Write-Host "[INFO] Building image anony-eval:cu121 ..." -ForegroundColor Cyan
& docker compose build | Out-Host

# Run compose service with env overrides
$env:MAX_IMAGES = $MaxImages
$env:BATCH_SIZE_DET = $BatchSizeDet
$env:DETECTOR_NAME = $DetectorName
$env:ENABLE_DEEPPRIVACY2 = $EnableDeepPrivacy2
$env:ENABLE_LDFA = $EnableLDFA

Write-Host "[INFO] Running eval in container..." -ForegroundColor Cyan
& docker compose run --rm eval | Tee-Object -FilePath "outputs\docker_run.log"

Write-Host "[INFO] Done. Artifacts should be under outputs/ and outputs/eval_runs/* ." -ForegroundColor Green
