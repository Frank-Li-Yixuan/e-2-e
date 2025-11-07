param(
  [string]$StartTime = "01:50",      # HH:mm (24h)
  [int]$Epochs = 24,                  # ~12h 估算（按当前吞吐量 2 epochs ≈ 1 小时）
  [int]$SaveEvery = 1000              # 减少磁盘 IO
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

# 项目根目录
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# 若脚本在 scripts/ 目录下，项目根是上一层
if (Split-Path -Leaf $ProjectDir -ne 'anony-project') { $ProjectDir = Split-Path -Parent $ProjectDir }
Set-Location $ProjectDir

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { Write-Err "找不到 venv Python: $VenvPython"; exit 1 }

# 计算等待到 StartTime 的秒数（若已过该时间，则加一天）
$now = Get-Date
$today = $now.ToString('yyyy-MM-dd')
try {
  $target = [datetime]::ParseExact("$today $StartTime", 'yyyy-MM-dd HH:mm', $null)
} catch {
  Write-Err "StartTime 格式应为 HH:mm，例如 01:50"; exit 1
}
if ($target -lt $now) { $target = $target.AddDays(1) }
$waitSec = [int][Math]::Max(0, ($target - $now).TotalSeconds)
Write-Info "当前时间 $($now.ToString('HH:mm:ss'))，将等待至 $($target.ToString('yyyy-MM-dd HH:mm:ss'))，约 $waitSec 秒"
Start-Sleep -Seconds $waitSec

# 等待可能仍在运行的上一段训练结束，避免冲突
function Is-TrainRunning {
  $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*src.train_joint*' }
  return ($procs -ne $null -and $procs.Count -gt 0)
}

$checkCount = 0
while (Is-TrainRunning) {
  if ($checkCount -eq 0) { Write-Warn "检测到已有 train_joint 进程，等待其结束…" }
  Start-Sleep -Seconds 30
  $checkCount++
}
if ($checkCount -gt 0) { Write-Info "先前训练已结束，开始新的长跑。" }

# 选择最近的 generator 检查点，若没有则回退到预训练权重
$latestGen = Get-ChildItem "$ProjectDir\outputs\joint_small" -Filter "generator_step*.pt" -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $latestGen) {
  $latestGenPath = Join-Path $ProjectDir "outputs\joint_small\pretrain_checkpoint.pt"
  if (-not (Test-Path $latestGenPath)) { Write-Err "没有找到可用的 resume 检查点"; exit 1 }
  Write-Warn "未找到 generator_step*.pt，回退到预训练检查点"
} else {
  $latestGenPath = $latestGen.FullName
}
Write-Info "使用 resume: $latestGenPath"

# 生成输出与日志路径
$ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
$outDir = Join-Path $ProjectDir "outputs\joint_long_$ts"
$logsDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
$stdoutLog = Join-Path $logsDir "longrun_${ts}_stdout.txt"
$stderrLog = Join-Path $logsDir "longrun_${ts}_stderr.txt"

# 生成新的配置
$baseCfg = Join-Path $ProjectDir "configs\joint_small.yaml"
$genCfg  = Join-Path $ProjectDir "configs\joint_long_${ts}.yaml"

& $VenvPython "$ProjectDir\scripts\generate_config.py" \
  --base "$baseCfg" \
  --out "$genCfg" \
  --outputs_dir "$outDir" \
  --epochs $Epochs \
  --save_every $SaveEvery \
  --resume "$latestGenPath"
if ($LASTEXITCODE -ne 0) { Write-Err "生成配置失败"; exit 1 }

Write-Info "启动长跑训练，输出目录：$outDir"
Start-Process -FilePath $VenvPython \
  -ArgumentList "-m","src.train_joint","--config","$genCfg","--mode","joint" \
  -WorkingDirectory $ProjectDir \
  -RedirectStandardOutput "$stdoutLog" \
  -RedirectStandardError  "$stderrLog"

Write-Info "已在后台启动。日志：`n  STDOUT: $stdoutLog`n  STDERR: $stderrLog"
