# Post-train structural quick-check of best_v13paper100k.pt on 150 test GIDs
# (split_seamless_26k test), plus an oracle-start probe.
$ErrorActionPreference = "Stop"
$py = $env:MESHTAILOR_PYTHON; if (-not $py) { $py = "python" }
$root = Split-Path -Parent $PSScriptRoot
$argstr = ('"' + "$root\tools\paired_protocol_eval.py" + '"' +
    " --legacy_ckpt `"$root\checkpoints\best_v13paper100k.pt`"" +
    " --paper_ckpt `"$root\checkpoints\best_v13paper100k.pt`"" +
    " --data_dir `"$root\processed_data_seamless_v13`"" +
    " --split_file `"$root\meshtailor\data\split_seamless_26k.json`"" +
    " --split test --limit 150 --bf16 --seed 20260818" +
    " --out_json `"$root\checkpoints\struct_150_v13paper100k.json`"")
Write-Output ("=== START struct150 " + (Get-Date -Format s) + " ===")
$proc = Start-Process -FilePath $py -ArgumentList $argstr -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$root\checkpoints\struct_150_v13paper100k_log.txt" `
    -RedirectStandardError "$root\checkpoints\struct_150_v13paper100k_err.txt"
Write-Output ("=== DONE struct150 exit=" + $proc.ExitCode + " " + (Get-Date -Format s) + " ===")

# oracle-start probe on the full model (paired free vs forced starts)
$argstr2 = ('"' + "$root\tools\oracle_start.py" + '"' +
    " --ckpt `"$root\checkpoints\best_v13paper100k.pt`"" +
    " --data_dir `"$root\processed_data_seamless_v13`"" +
    " --split_file `"$root\meshtailor\data\split_seamless_26k.json`"" +
    " --n 150 --out `"$root\checkpoints\oracle_start_v13paper100k.txt`"")
Write-Output ("=== START oracle " + (Get-Date -Format s) + " ===")
$proc2 = Start-Process -FilePath $py -ArgumentList $argstr2 -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$root\checkpoints\oracle_start_v13paper100k_log.txt" `
    -RedirectStandardError "$root\checkpoints\oracle_start_v13paper100k_err.txt"
Write-Output ("=== DONE oracle exit=" + $proc2.ExitCode + " " + (Get-Date -Format s) + " ===")
