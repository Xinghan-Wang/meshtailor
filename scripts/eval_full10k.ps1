# paper100k full 10k test evaluation (p0 protocol).
# Stages: 0) GT outputs (skipped if already present) -> 1) model inference ->
# 2) ABF++ unwrap evaluation -> 3) finalize summary table.
$ErrorActionPreference = "Stop"
$py = $env:MESHTAILOR_PYTHON; if (-not $py) { $py = "python" }
$root = Split-Path -Parent $PSScriptRoot
$ckpt = "$root\checkpoints\best_paper100k.pt"
$data = "$root\processed_data_seamless_maximal"
$split = "$root\meshtailor\data\split_seamless_128k.json"
$p0out = "$root\test_outputs_full"
$gtout = "$root\gt_outputs_full"
$logd = "$root\checkpoints"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $logd | Out-Null

$ngt = @(Get-ChildItem -Path (Join-Path $gtout "*\seam.json") -ErrorAction SilentlyContinue).Count
if ($ngt -lt 10000) {
    Write-Output ("=== STAGE0 gt_eval " + (Get-Date -Format s) + " ===")
    $argstr0 = ('"' + "$root\eval\gt_eval.py" + '"' +
        " --split test --out_dir `"$gtout`" --unwrap abf --limit 0" +
        " --data_dir `"$data`" --split_file `"$split`"")
    $proc0 = Start-Process -FilePath $py -ArgumentList $argstr0 -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$logd\full10k_gt_log.txt" `
        -RedirectStandardError "$logd\full10k_gt_err.txt"
    Write-Output ("=== STAGE0 exit=" + $proc0.ExitCode + " " + (Get-Date -Format s) + " ===")
    if ($proc0.ExitCode -ne 0) { exit $proc0.ExitCode }
}

Write-Output ("=== STAGE1 inference " + (Get-Date -Format s) + " ===")
$argstr = ('"' + "$root\meshtailor\inference.py" + '"' +
    " --ckpt `"$ckpt`" --split test --out_dir `"$p0out`"" +
    " --limit 0 --temperature 0.1 --seed 20260818 --bf16" +
    " --data_dir `"$data`" --split_file `"$split`" --skip_existing")
$proc = Start-Process -FilePath $py -ArgumentList $argstr -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$logd\full10k_infer_log.txt" `
    -RedirectStandardError "$logd\full10k_infer_err.txt"
Write-Output ("=== STAGE1 exit=" + $proc.ExitCode + " " + (Get-Date -Format s) + " ===")
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

$n = @(Get-ChildItem -Path (Join-Path $p0out "*\seam.json") -ErrorAction SilentlyContinue).Count
Write-Output ("seam_count=" + $n)
if ($n -lt 10000) { Write-Output "INFERENCE INCOMPLETE"; exit 2 }

Write-Output ("=== STAGE2 abf eval " + (Get-Date -Format s) + " ===")
$argstr2 = ('"' + "$root\eval\run_eval.py" + '"' +
    " --ckpt `"$ckpt`" --split test --out_dir `"$p0out`" --skip_inference" +
    " --unwrap abf --limit 0 --data_dir `"$data`" --split_file `"$split`"")
$proc2 = Start-Process -FilePath $py -ArgumentList $argstr2 -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$logd\full10k_eval_log.txt" `
    -RedirectStandardError "$logd\full10k_eval_err.txt"
Write-Output ("=== STAGE2 exit=" + $proc2.ExitCode + " " + (Get-Date -Format s) + " ===")
if ($proc2.ExitCode -ne 0) { exit $proc2.ExitCode }

Write-Output ("=== STAGE3 finalize " + (Get-Date -Format s) + " ===")
$argstr3 = ('"' + "$root\tools\finalize_eval.py" + '"' +
    " --model_dir `"$p0out`" --gt_dir `"$gtout`"" +
    " --data_dir `"$data`" --split_file `"$split`"" +
    " --summary `"$logd\full10k_summary.txt`"")
$proc3 = Start-Process -FilePath $py -ArgumentList $argstr3 -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$logd\full10k_finalize_log.txt" `
    -RedirectStandardError "$logd\full10k_finalize_err.txt"
Write-Output ("=== STAGE3 exit=" + $proc3.ExitCode + " " + (Get-Date -Format s) + " ===")
Write-Output "V13 FULL 10K PIPELINE DONE"
