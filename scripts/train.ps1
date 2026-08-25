# Final model training: maximal (paper B.1) labels, paper protocol, 100k train split.
# Reproduce best_paper100k.pt. Hyperparameters identical to the paper-style run.
$ErrorActionPreference = "Stop"
$py = $env:MESHTAILOR_PYTHON; if (-not $py) { $py = "python" }
$root = Split-Path -Parent $PSScriptRoot
$tag = "paper100k"
$argstr = ('"' + "$root\meshtailor\train.py" + '"' +
    " --epochs 30 --batch_size 4 --accum 8 --num_workers 4" +
    " --t_max 2000 --dropout 0.1 --lr 1e-4 --grad_clip 1.0" +
    " --eoc_weight 1.0 --eos_weight 1.0 --ss_prob 0.0" +
    " --sequence_protocol paper --seed 20260818 --patience 5" +
    " --tag $tag --ckpt_dir `"$root\checkpoints`"" +
    " --data_dir `"$root\processed_data_seamless_maximal`"" +
    " --split_file `"$root\meshtailor\data\split_seamless_128k.json`"")
Write-Output ("=== START $tag " + (Get-Date -Format s) + " ===")
$proc = Start-Process -FilePath $py -ArgumentList $argstr -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$root\checkpoints\paper100k_log.txt" `
    -RedirectStandardError "$root\checkpoints\paper100k_err.txt"
Write-Output ("=== DONE $tag exit=" + $proc.ExitCode + " " + (Get-Date -Format s) + " ===")
