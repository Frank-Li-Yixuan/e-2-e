param(
  [int[]]$BatchCandidates = @(16, 14, 12, 10, 8, 6, 4, 3, 2),
  [int[]]$SizeCandidates = @(768, 640, 512, 448, 384, 320),
  [int[]]$GenChannels = @(96, 80, 64, 56, 48),
  [int[]]$DiscChannels = @(96, 80, 64, 56, 48),
  [int]$Epochs = 24,
  [int]$SaveEvery = 2000
)

$ErrorActionPreference = 'Stop'
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Leaf $proj) -ne 'anony-project') { $proj = Split-Path -Parent $proj }
Set-Location $proj
$py = Join-Path $proj '.venv\Scripts\python.exe'
$base = Join-Path $proj 'configs\joint_small.yaml'
$env:PYTHONPATH = $proj

# kill existing train processes (safe)
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*src.train_joint*' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }

# pick resume
$res = (Get-ChildItem "$proj\outputs\joint_small" -Filter 'generator_step*.pt' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
if (-not $res) { $res = Join-Path $proj 'outputs\joint_small\pretrain_checkpoint.pt' }

$chosen = $null
foreach ($ch in $GenChannels) {
  foreach ($sz in $SizeCandidates) {
    foreach ($bs in $BatchCandidates) {
      Write-Host "[TRY] gen=$ch, disc=$ch, size=$sz, batch=$bs" -ForegroundColor Cyan
      & $py "$proj\scripts\probe_train_fit.py" --config "$base" --batch $bs --image_size $sz --gen_channels $ch --disc_channels $ch
      if ($LASTEXITCODE -eq 0) {
        $chosen = @{ gen=$ch; disc=$ch; size=$sz; batch=$bs }
        break
      }
    }
    if ($chosen) { break }
  }
  if ($chosen) { break }
}

if (-not $chosen) {
  Write-Host "[ERR] No configuration fit in memory. Falling back to batch=2,size=320,gen=48,disc=48" -ForegroundColor Red
  $chosen = @{ gen=48; disc=48; size=320; batch=2 }
}

$ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
$out = Join-Path $proj "outputs\joint_high_${ts}"
$genCfg = Join-Path $proj "configs\joint_high_${ts}.yaml"

& $py "$proj\scripts\generate_config.py" `
  --base "$base" `
  --out "$genCfg" `
  --outputs_dir "$out" `
  --epochs $Epochs `
  --save_every $SaveEvery `
  --resume "$res" `
  --batch_size $($chosen.batch) `
  --num_workers 4 `
  --precision fp16 `
  --image_size $($chosen.size) `
  --gen_base_channels $($chosen.gen) `
  --disc_base_channels $($chosen.disc) `
  --gen_torch_compile `
  --disc_torch_compile
if ($LASTEXITCODE -ne 0) { throw 'generate_config failed' }

Write-Host "[LAUNCH] size=$($chosen.size) batch=$($chosen.batch) gen=$($chosen.gen) disc=$($chosen.disc)" -ForegroundColor Green
$stdout = Join-Path $proj "logs\longrun_${ts}_stdout.txt"
$stderr = Join-Path $proj "logs\longrun_${ts}_stderr.txt"
Start-Process -FilePath $py -ArgumentList "-m","src.train_joint","--config","$genCfg","--mode","joint" -WorkingDirectory $proj -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Write-Host "Logs: `n STDOUT: $stdout`n STDERR: $stderr"
