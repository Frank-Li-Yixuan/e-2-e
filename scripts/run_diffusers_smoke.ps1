$ErrorActionPreference = 'Stop'
Set-Location -Path 'F:\Datasets\anony-project'
if (-not (Test-Path -Path 'logs')) { New-Item -ItemType Directory -Path 'logs' | Out-Null }

$py = '.\.venv\Scripts\python.exe'

# Face smoke on WIDER_val (recursive)
$wider = 'F:\Datasets\WiderFace\WIDER_val\WIDER_val\images'
if (Test-Path $wider) {
  & $py -m src.infer_e2e --config 'configs\e2e_full.yaml' --det_backend face_plate_hybrid --images $wider --recursive --output 'outputs\anonymized_e2e\smoke_wider_diff_2' --max_images 2 --gen_backend diffusers *> 'logs\infer_e2e_diff_faces.log'
  "=== wider outputs ==="
  Get-ChildItem .\outputs\anonymized_e2e\smoke_wider_diff_2 -File -ErrorAction SilentlyContinue | Select-Object -First 5 Name,Length
}

# Plate smoke on CCPD2020
$ccpd = 'F:\Datasets\CCPD2020\CCPD2020'
if (Test-Path $ccpd) {
  & $py -m src.infer_e2e --config 'configs\e2e_full.yaml' --det_backend face_plate_hybrid --images $ccpd --output 'outputs\anonymized_e2e\smoke_ccpd_diff_2' --max_images 2 --recursive --gen_backend diffusers *> 'logs\infer_e2e_diff_plate.log'
  "=== ccpd outputs ==="
  Get-ChildItem .\outputs\anonymized_e2e\smoke_ccpd_diff_2 -File -ErrorAction SilentlyContinue | Select-Object -First 5 Name,Length
}
