param(
  [string]$CocoJson = "data/ccpd_coco.json",
  [string]$ImagesRoot = "../CCPD2020",
  [string]$OutDir = "outputs/datasets/plate_yolo11",
  [int]$Epochs = 50,
  [int]$Batch = 16,
  [int]$ImgSize = 640,
  [string]$Device = "0",
  [string]$RepoDir = ""
)

$ErrorActionPreference = 'Stop'
if (-not $RepoDir) { $RepoDir = (Get-Location).Path }
Write-Host "[INFO] REPO_DIR = $RepoDir" -ForegroundColor Cyan

# Resolve paths
$cocoPath = Join-Path $RepoDir $CocoJson
$imagesPath = Join-Path $RepoDir $ImagesRoot
Write-Host "[INFO] COCO_JSON   = $cocoPath" -ForegroundColor Cyan
Write-Host "[INFO] IMAGES_ROOT = $imagesPath" -ForegroundColor Cyan
if ($imagesPath -match "CCPD2020\\CCPD2020") {
  Write-Warning "IMAGES_ROOT seems to include CCPD2020 twice; if your COCO file_name starts with 'CCPD2020/...', IMAGES_ROOT should be the parent directory that contains the 'CCPD2020' folder (e.g., ..\\CCPD2020)."
}

# Reuse .venv from run_local.ps1 if exists; otherwise create
$venv = Join-Path $RepoDir ".venv"
$py = Join-Path $venv "Scripts/python.exe"
$pip = Join-Path $venv "Scripts/pip.exe"
if (-not (Test-Path $venv)) {
  Write-Host "[INFO] Creating venv ..." -ForegroundColor Cyan
  python -m venv "$venv"
}
& "$py" -m pip install -U pip wheel setuptools | Out-Host
# Ensure ultralytics and deps
& "$pip" install ultralytics numpy "opencv-python-headless==4.8.1.78" | Out-Host
# Re-pin ABI-safe versions to avoid NumPy 2.x / OpenCV ABI issues
& "$pip" install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78" | Out-Host

# Export COCO -> YOLO dataset
& "$py" (Join-Path $RepoDir 'scripts/export_coco_to_yolo.py') --coco_json $cocoPath --images_root $imagesPath --out_dir (Join-Path $RepoDir $OutDir)

# Train YOLOv11
& "$py" (Join-Path $RepoDir 'scripts/train_plate_yolo11.py') --data_yaml (Join-Path $RepoDir 'data/plate_yolo11.yaml') --epochs $Epochs --imgsz $ImgSize --batch $Batch --device $Device

$best = Join-Path $RepoDir 'models/plate_yolo11_best.pt'
if (Test-Path $best) {
  Write-Host "[INFO] Training finished. Best weights: $best" -ForegroundColor Green
  Write-Host "[INFO] You can run evaluation with: ./scripts/run_local.ps1 -DetectorName yolov11_combo -PlateModelPath $best" -ForegroundColor Green
} else {
  Write-Warning "Training finished but best weight not found. Check runs/detect/plate_yolo11/weights/best.pt"
}
