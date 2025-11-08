param(
  [string]$Source = (Get-Location).Path,
  [string]$Target = "..\anony-project-clean",
  [switch]$Force = $false
)

$ErrorActionPreference = 'Stop'
function Write-Section($m){ Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Step($m){ Write-Host "[>] $m" -ForegroundColor Yellow }
function Ensure-Dir($p){ if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }

Write-Section "Prepare clean repo"
$absTarget = Resolve-Path -Path $Target -ErrorAction SilentlyContinue
if ($absTarget) { $Target = $absTarget.Path }
if ((Test-Path $Target) -and -not $Force) { Write-Error "Target exists: $Target (use -Force to overwrite)" }
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
Ensure-Dir $Target

# 需要保留的相对路径（按需增减）
$keep = @(
  'configs','scripts','src','requirements.txt','README.md','README_planB_colab.md','Makefile','docker','docker-compose.yml','.gitignore','.gitattributes'
)

foreach ($item in $keep) {
  $src = Join-Path $Source $item
  if (Test-Path $src) {
    Write-Step "Copy $item"
    Copy-Item $src -Destination (Join-Path $Target $item) -Recurse -Force
  }
}

# 覆盖式写.gitignore（干净上传）
$gitignore = @"
# Python
__pycache__/
*.pyc
*.pyo

# Virtual env
.venv*/
venv/
env/

# Datasets / large raw data
Datasets/
CCPD2020/
PP4AV/
WiderFace/

data/images/
data/masks/
data/unified/

# Outputs / logs / runs
outputs/
runs/
logs/
status/

# Checkpoints / weights
*.pt
*.onnx
*.ckpt
*.safetensors

# Jupyter
.ipynb_checkpoints/

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Cache
.cache/
wandb/
"@
$gitignore | Set-Content -Path (Join-Path $Target ".gitignore") -Encoding UTF8

# 可选 LFS 配置占位（用户在新目录中执行 git lfs install 后生效）
$gitattributes = @"
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
"@
$gitattributes | Set-Content -Path (Join-Path $Target ".gitattributes") -Encoding UTF8

Write-Section "Done"
Write-Host "Clean repo prepared at: $Target" -ForegroundColor Green
