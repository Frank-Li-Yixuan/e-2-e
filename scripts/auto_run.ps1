#requires -version 5.1
# Auto-run E2E pipeline on Windows PowerShell with logging and error reporting
# Usage: powershell -ExecutionPolicy Bypass -File scripts/auto_run.ps1

$ErrorActionPreference = 'Stop'

function New-LogEnv {
  New-Item -ItemType Directory -Force -Path logs | Out-Null
  New-Item -ItemType Directory -Force -Path status | Out-Null
  New-Item -ItemType Directory -Force -Path tests\reports | Out-Null
}

function Write-StatusError($Title, $Step, $ErrLogPath) {
  function _SafePy([string]$code) {
    try {
      return (python -c $code 2>$null)
    } catch {
      return 'N/A'
    }
  }
  $ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmssZ')
  $issue = "issues/${Title}_${ts}.md"
  New-Item -ItemType Directory -Force -Path issues | Out-Null
  $pyver = _SafePy "import sys; print(sys.version)"
  $torchver = _SafePy "import torch; print(getattr(torch,'__version__','N/A'))"
  $cudaver = _SafePy "import torch; print(getattr(getattr(torch,'version',None),'cuda','N/A'))"
  $gpu = _SafePy "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  $tail = if (Test-Path $ErrLogPath) { Get-Content $ErrLogPath -Tail 100 } else { @('NO STDERR') }
  @(
    "# $Title",
    "",
    "## When",
    "- UTC: $ts",
    "",
    "## Step",
    "- $Step",
    "",
    "## Error Snippet (last 100 lines)",
    '```',
    ($tail -join "`n"),
    '```',
    "",
    "## Environment",
    "- Python: $pyver",
    "- Torch: $torchver",
    "- CUDA: $cudaver",
    "- GPU: $gpu",
    "",
    "## Attempted Fixes",
    "1. Retry install",
    "2. Reduce batch size / switch to CPU",
    "3. Defer to Colab",
    "",
    "## Request",
    "- Continue with workaround / rollback / change parameter?"
  ) | Set-Content -Encoding UTF8 $issue
  @("# ERROR STOP","","Issue: $issue") | Set-Content -Encoding UTF8 status\ERROR_STOP.md
}

try {
  New-LogEnv

  # 1) Venv + pip
  if (!(Test-Path '.venv\Scripts\Activate.ps1')) {
    $venvCreated = $false
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
      try {
        py -3.11 -m venv .venv
        $venvCreated = $true
        Write-Host '[INFO] Created venv with Python 3.11 via py launcher.'
      } catch {
        Write-Host '[WARN] py -3.11 not available; falling back to current python for venv.'
      }
    }
    if (-not $venvCreated) {
      python -m venv .venv
    }
  }
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip

  # 2) Install requirements
  try {
    python -m pip install -r requirements.txt *> logs/install_stdout.txt 2*> logs/install_stderr.txt
    if ($LASTEXITCODE -ne 0) {
      "INSTALL_FAIL" | Set-Content status/install_fail.flag
    }
  } catch {
    "INSTALL_FAIL" | Set-Content status/install_fail.flag
  }
  if (Test-Path status/install_fail.flag) {
    Write-StatusError -Title 'install_failure' -Step 'pip install requirements' -ErrLogPath 'logs/install_stderr.txt'
    exit 2
  }
  $installErr = Select-String -Path logs/install_stdout.txt -Pattern 'ERROR' -SimpleMatch -Quiet
  if ($installErr) {
    Write-StatusError -Title 'install_error_keyword' -Step 'pip install requirements' -ErrLogPath 'logs/install_stdout.txt'
    exit 2
  }

  # 3) Merge COCO if needed
  $unifiedTrain = 'data/unified/unified_train.json'
  if (Test-Path $unifiedTrain -and ((Get-Item $unifiedTrain).Length -gt 1MB)) {
    Write-Host '[INFO] Unified train exists; skip merge.'
  } else {
    python scripts/merge_and_sample_coco.py --inputs data/*.json --out_dir data/unified --min_side 8 --train_ratio 0.7 --val_ratio 0.15 --seed 42 *> logs/merge_stdout.txt 2*> logs/merge_stderr.txt
  }
  $train = 'data/unified/unified_train.json'
  $val = 'data/unified/unified_val.json'
  $test = 'data/unified/unified_test.json'
  if (!(Test-Path $train) -or !(Test-Path $val) -or !(Test-Path $test)) {
    Write-StatusError -Title 'merge_failure' -Step 'merge_and_sample_coco' -ErrLogPath 'logs/merge_stderr.txt'
    exit 2
  }
  if ( (Get-Item $train).Length -lt 100KB -or (Get-Item $val).Length -lt 100KB -or (Get-Item $test).Length -lt 100KB ) {
    Write-StatusError -Title 'merge_too_small' -Step 'merge_and_sample_coco size check' -ErrLogPath 'logs/merge_stdout.txt'
    exit 2
  }
  # write status/01_data_prep.md details via helper
  python scripts/report_coco_status.py --train $train --val $val --test $test --out status/01_data_prep.md

  # 4) Pseudotargets bootstrap (local stub)
  if (Test-Path 'scripts/generate_pseudotargets.py') {
    python scripts/generate_pseudotargets.py *> logs/pseudo_stdout.txt 2*> logs/pseudo_stderr.txt
  } else {
    @('# 02 Pseudotargets QC Report','','- deferred-to-colab: run DeepPrivacy2/LDFA on top-2000 train images and save to paths.pseudotargets','- see README for commands') | Set-Content -Encoding UTF8 status/02_pseudotargets.md
  }

  # 5) Smoke Test
  try {
    python -m tests.smoke_test *> logs/smoke_stdout.txt 2*> logs/smoke_stderr.txt
  } catch {
    'SMOKE_FAIL' | Set-Content status/smoke_failed.flag
  }
  if (Test-Path status/smoke_failed.flag) {
    Write-StatusError -Title 'smoke_failure' -Step 'python -m tests.smoke_test' -ErrLogPath 'logs/smoke_stderr.txt'
    Copy-Item logs/smoke_* tests/reports/ -Force
    exit 2
  }
  Copy-Item logs/smoke_* tests/reports/ -Force

  # 6) Pretrain (short)
  try {
    python -m src.train_joint --config configs/joint_small.yaml --mode pretrain --max_steps 200 *> logs/pretrain_stdout.txt 2*> logs/pretrain_stderr.txt
  } catch {
    Write-StatusError -Title 'pretrain_failure' -Step 'train_joint pretrain' -ErrLogPath 'logs/pretrain_stderr.txt'
    exit 2
  }
  if (!(Test-Path 'outputs/pretrain_samples')) {
    Write-Host '[WARN] Pretrain samples not found.'
  }

  # 7) Joint short run (may fallback to Colab)
  try {
    python -m src.train_joint --config configs/joint_small.yaml --mode joint --max_steps 800 *> logs/joint_stdout.txt 2*> logs/joint_stderr.txt
  } catch {
    @('# 04 Colab Run Report','','- Fallback to Colab: see notebooks/colab_train.ipynb cells for execution','- Reason: local joint run error; check logs/joint_stderr.txt') | Set-Content -Encoding UTF8 status/04_colab_run.md
  }

  # 8) Evaluate latest val_samples
  $latest = Get-ChildItem -Path runs\joint_small\val_samples -Recurse -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($null -eq $latest) { $latest = Get-ChildItem -Path outputs\val_samples -Recurse -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 }
  if ($null -ne $latest) {
    $orig = Join-Path $latest.FullName 'orig'
    $anon = Join-Path $latest.FullName 'anon'
    if (Test-Path $orig -and Test-Path $anon) {
      python -m src.eval_utils --orig_dir $orig --anon_dir $anon --out outputs/eval_report.json *> logs/eval_stdout.txt 2*> logs/eval_stderr.txt
    }
  }

  # 9) Git branch + PR (instructions only; do not auto-push credentials)
  @(
    '# PR Instructions',
    '',
    'git checkout -b feat/auto-run-{0}' -f (Get-Date -Format yyyyMMdd),
    'git add status logs tests/reports outputs/eval_report.* configs/joint_small.yaml',
    'git commit -m "feat(auto-run): data→pseudotargets→smoke→pretrain→joint (auto-run)"',
    'git push origin HEAD',
    'Open PR with template .github/PULL_REQUEST_TEMPLATE.md'
  ) | Set-Content -Encoding UTF8 status/PR_INSTRUCTIONS.md

  Write-Host 'Auto-run completed. Review status/*.md and logs/*.txt, then follow status/PR_INSTRUCTIONS.md to open PR.'

} catch {
  Write-Host "[FATAL] $($_.Exception.Message)"
  Write-StatusError -Title 'fatal_error' -Step 'auto_run.ps1' -ErrLogPath 'logs/install_stderr.txt'
  exit 2
}
