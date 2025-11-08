$ErrorActionPreference = 'Stop'
Set-Location -Path 'F:\Datasets\anony-project'
if (-not (Test-Path -Path 'logs')) { New-Item -ItemType Directory -Path 'logs' | Out-Null }
$py = '.\\.venv\\Scripts\\python.exe'
$pyArgs = @(
  '-m','src.infer_e2e',
  '--config','configs\e2e_full.yaml',
  '--images','F:\Datasets\PP4AV\images\fisheye',
  '--output','outputs\anonymized_e2e\smoke_pp4av_unet_4',
  '--max_images','4',
  '--gen_backend','unet'
)
& $py @pyArgs *> 'logs\\infer_e2e_unet_smoke.log'
"=== list first 5 ==="
Get-ChildItem .\outputs\anonymized_e2e\smoke_pp4av_unet_4 -File -ErrorAction SilentlyContinue | Select-Object -First 5 Name,Length
"=== total count ==="
(Get-ChildItem .\outputs\anonymized_e2e\smoke_pp4av_unet_4 -File -ErrorAction SilentlyContinue | Measure-Object).Count
