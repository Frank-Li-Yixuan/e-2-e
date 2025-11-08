param(
  [string]$Remote = "https://github.com/Frank-Li-Yixuan/E2E.git",
  [string]$Branch = "main",
  [switch]$UseLFS = $true,
  [switch]$Force = $false,
  [switch]$DryRun = $false,
  [switch]$SkipInit = $false,
  [switch]$SetAutoCRLF = $true
)

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Step($msg) { Write-Host "[>] $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor DarkYellow }
function Write-Err($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red }

function Test-Cmd($name) {
  try { & $name --version > $null 2>&1; return $true } catch { return $false }
}

function Ensure-File(
  [string]$Path,
  [string]$Content,
  [switch]$Force
) {
  $dir = Split-Path -Parent $Path
    Git-Run "config user.name AutoUser"
    Git-Run "config user.email auto@example.com"
  }
  $Content | Set-Content -Path $Path -Encoding UTF8
  Write-Ok "Wrote: $Path"
}

function Git-Run([string]$Cmd) {
  # Robust git invoker preserving quoted arguments (commit messages, URLs, etc.)
  Write-Step "git $Cmd"
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "git"
  $psi.Arguments = $Cmd
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $proc = [System.Diagnostics.Process]::Start($psi)
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  if ($stdout) { $stdout.TrimEnd().Split("`n") | ForEach-Object { Write-Host $_ } }
  if ($stderr) { $stderr.TrimEnd().Split("`n") | ForEach-Object { Write-Host $_ } }
  if ($proc.ExitCode -ne 0) { throw "git $Cmd failed with code $($proc.ExitCode)" }
}

Write-Section "Preflight checks"
if (-not (Test-Cmd git)) { Write-Err "Git not found in PATH"; exit 1 }
if ($UseLFS -and -not (Test-Cmd git-lfs)) { Write-Warn "Git LFS not found; will try 'git lfs install' anyway" }

$repoRoot = Get-Location
Write-Step "Repo root: $repoRoot"

Write-Section "Init & Settings"
if (-not $SkipInit) {
  # If running under a parent Git repo (detected via rev-parse), force a local repo using GIT_DIR/WORK_TREE
  try {
    $top = (& git rev-parse --show-toplevel 2>$null).Trim()
    if ($top -and ($top -ne (Get-Location).Path)) {
      Write-Warn "Detected parent Git repo at $top; forcing local repo in current folder"
      $env:GIT_DIR = (Join-Path (Get-Location).Path ".git")
      $env:GIT_WORK_TREE = (Get-Location).Path
    }
  } catch {}
  if (-not (Test-Path ".git")) {
    Git-Run "init"
  } else { Write-Step ".git exists (skip init)" }
}
try { Git-Run "config credential.helper manager" } catch { Write-Warn "credential.helper manager config failed (non-fatal)" }
if ($SetAutoCRLF) {
  try { Git-Run "config core.autocrlf true" } catch { Write-Warn "core.autocrlf setup failed (non-fatal)" }
}

Write-Section "Write ignore & LFS (idempotent)"
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
Ensure-File -Path ".gitignore" -Content $gitignore -Force:$Force

if ($UseLFS) {
  try { Git-Run "lfs install" } catch { Write-Warn "git lfs install failed (continuing)" }
  $gitattributes = @"
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
"@
  Ensure-File -Path ".gitattributes" -Content $gitattributes -Force:$Force
}

Write-Section "Untrack heavy folders already added"
$heavyPaths = @(
  "Datasets","CCPD2020","PP4AV","WiderFace",
  "data/images","data/masks","data/unified",
  "outputs","runs","logs","status"
)
foreach ($p in $heavyPaths) {
  if (Test-Path $p) {
  try { Git-Run "rm -r --cached --ignore-unmatch $p" } catch { Write-Warn "skip rm cached for $p (non-fatal)" }
  }
}

Write-Section "Stage & commit"
try { Git-Run "add ." } catch { Write-Err "git add failed"; exit 1 }

if ($DryRun) {
  Write-Section "DryRun status"
  & git status
  Write-Warn "DryRun enabled; stop before commit/push"
  exit 0
}

# Create an initial commit if no commits exist
$hasCommit = $false
try {
  & git rev-parse --verify HEAD > $null 2>&1; if ($LASTEXITCODE -eq 0) { $hasCommit = $true }
} catch { $hasCommit = $false }

if ($hasCommit) {
  Write-Step "Amend or new commit"
  try { Git-Run "commit -m `"Repo cleanup: ignore datasets/outputs, LFS for weights, EOL normalize`"" } catch { Write-Warn "Nothing to commit" }
} else {
  Write-Step "Initial commit"
  try {
    $name = (& git config user.name 2>$null)
    $email = (& git config user.email 2>$null)
    if (-not $name) { Git-Run "config user.name AutoUser" }
    if (-not $email) { Git-Run "config user.email auto@example.com" }
  Git-Run 'commit -m "Initial commit: anonymization Plan A/Plan B pipeline"'
  } catch { Write-Warn "Nothing to commit" }
}

Write-Section "Branch & remote"
try { Git-Run "branch -M $Branch" } catch { Write-Warn "branch rename failed (non-fatal)" }

# Ensure remote origin
$remoteExists = & git remote 2>$null | Select-String -Pattern "^origin$"
if (-not $remoteExists) {
  try { Git-Run "remote add origin $Remote" } catch { Write-Warn "remote add failed (maybe exists)" }
} else {
  Write-Step "Remote 'origin' exists"
}

# If URL mismatch, set-url
try {
  $curUrl = (& git remote get-url origin).Trim()
  if ($curUrl -ne $Remote) {
    Write-Step "Updating origin URL: $curUrl -> $Remote"
    Git-Run "remote set-url origin $Remote"
  }
} catch { Write-Warn "get-url origin failed (non-fatal)" }

Write-Section "Push"
try {
  Git-Run "push -u origin $Branch"
  Write-Ok "Pushed to $Remote ($Branch)"
} catch {
  Write-Warn "Push failed. If this is a private repo or first push, credentials may be required. If branch refspec missing, ensure at least one commit was created."
  throw
}

Write-Ok "Done."
