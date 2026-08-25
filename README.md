# MeshTailor Reproduction

An independent reproduction of **MeshTailor: Cutting Seams via Generative Mesh Traversal**
(Ma, Yan, Zhang, Huang, arXiv:2603.27309, 2026; official code TBA).

MeshTailor generates garment sewing seams directly on a triangle mesh as an
autoregressive traversal: a pointer network decodes chains of vertex indices
(seam chains) conditioned on geometry and panel-graph encodings, instead of
operating in UV space.

This repository contains the complete training/evaluation pipeline of our final
model (**v13**, "paper B.1 maximal chain" labels + paper sequence protocol) and
its reported metrics. Pretrained weights are hosted on Hugging Face (see below).

## Final results (10k test split, garment count 10,000)

Best checkpoint: `best_v13paper100k.pt` (epoch with best val NLL, 100k train split).

| Metric | v13 (ours) | GT | v13 / GT | Paper model / GT |
|---|---|---|---|---|
| area std-log ↓ | 0.6957 | 0.5551 | 1.25× | 1.00× |
| area mean\|r−1\| ↓ | 0.8483 | 0.7972 | 1.06× | — |
| compactness ↑ | 0.5232 | 0.5236 | 1.00× | 1.00× |
| convexity ↑ | 0.8578 | 0.8426 | 1.02× | 1.00× |
| seam length / area ↓ | 2.9798 | 3.2846 | 0.91× | 0.94× |
| jaggedness ↓ | 0.0014 | 0.0013 | 1.10× | — |
| chart count ↓ | 9.582 | 10.951 | 0.88× | 0.91× |
| macro edge recall ↑ | 0.8465 | — | — | — |
| macro edge precision ↑ | 0.9314 | — | — | — |
| unique seam edges | 346.6 | 397.4 | 0.87× | — |
| chains per garment | 12.69 | 12.9 | ✓ | — |

UV/area metrics use a real ABF++ unwrap (geogram-based, run headlessly in WSL).

## Model weights

The checkpoint is ~1.14 GB and is **not** stored in this git repository.
Download it from Hugging Face into `checkpoints/`:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <YOUR_HF_USERNAME>/meshtailor-v13paper100k \
    best_v13paper100k.pt --local-dir checkpoints
```

(Replace `<YOUR_HF_USERNAME>` with the repository owner's Hugging Face user name;
see `huggingface/README.md` for the model card used for that upload.)

## Setup

Environment used for all results:

| Item | Value |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| GPU | NVIDIA RTX 5080 16 GB |
| Python | 3.10 (conda env `meshtailor`) |
| PyTorch / PyG | 2.13.0+cu130 / 2.8.0 |
| trimesh / PyMeshLab | 4.12.2 / 2025.7 |

```bash
conda create -n meshtailor python=3.10 -y
conda activate meshtailor
pip install -r requirements.txt   # install the CUDA build of torch first if needed
```

Two external native tools are required (both run inside WSL):

1. **SeamAwareDecimater** (paper ref. [21]) — seam-preserving mesh decimation
   used by the preprocessing stage. Expected at `/root/seam-decimater/build/decimater`
   (override with env var `SEAMLESS_DECIMATER`).
2. **ABF++ unwrap** — build `tools/abf/abf_unwrap.cpp` with CMake against
   [geogram](https://github.com/BrunoLevy/geogram); expected at
   `/root/abf_toolkit/tool/build/abf_unwrap` (override with `ABF_BIN`, and set
   `GEO_LIB` to the geogram release lib dir).

Optional environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `MESHTAILOR_PYTHON` | Python executable used by `scripts/*.ps1` | `python` |
| `SEAMLESS_DECIMATER` | WSL path to the decimater binary | `/root/seam-decimater/build/decimater` |
| `SEAMLESS_TMP` | Windows staging dir for decimation batches | `C:\Temp\seamless_batch` |
| `ABF_BIN` / `GEO_LIB` / `ABF_WSL` | ABF++ unwrap binary / geogram lib / WSL launcher | see `eval/unwrap_abf.py` |
| `BLENDER_BIN` | Blender executable (fallback unwrap only) | `D:\Blender\blender.exe` |

## Data

The dataset is GarmentCodeDataset (v2 archives,
`garments_5000_*/default_body/data.tar.gz`) — obtain it from its official release
and point `--data_root` at it. It is not redistributable from this repo.

Full reproduction flow (PowerShell on Windows; data pipeline shells out to WSL):

```powershell
# 1) seam-preserving decimation -> processed_data_seamless/ (128,974 garments)
python tools/preprocess_seamless.py `
    --data_root GarmentCodeData/GarmentCodeData_v2 `
    --output_dir processed_data_seamless --target 1000 --par 16

# 2) v13 relabel: paper B.1 maximal chain decomposition -> processed_data_seamless_v13/
python tools/repair_seam_chains.py `
    --src_dir processed_data_seamless `
    --out_dir processed_data_seamless_v13 `
    --split_file meshtailor/data/split_seamless_128k.json `
    --mode maximal --workers 12

# 3) train (30 epochs, patience 5; ~24 h on an RTX 5080)
./scripts/run_v13paper100k.ps1

# 4) quick structural checks on 150 test garments + oracle-start probe
./scripts/run_v13_postcheck.ps1

# 5) full 10k test evaluation (GT unwrap -> inference -> ABF eval -> summary)
./scripts/run_v13_full10k.ps1
```

Key hyperparameters (see `scripts/run_v13paper100k.ps1`): batch 4 × grad-accum 8,
lr 1e-4, dropout 0.1, grad clip 1.0, `T_max=2000`, `eoc_weight=eos_weight=1.0`,
`--sequence_protocol paper`, seed 20260818.

Splits (100k train / 10k val / 10k test) are committed under `meshtailor/data/`.

## Quick inference only

With the downloaded checkpoint and the processed test data:

```bash
python meshtailor/inference.py --ckpt checkpoints/best_v13paper100k.pt \
    --split test --out_dir test_outputs --temperature 0.1 --bf16
```

Per-garment outputs: `mesh.obj`, `seam.json` (generated chains), plus unwrap
results when run through `eval/run_eval.py`.

Visualization helpers:

- `tools/export_compare_obj_v13.py` — three-color comparison OBJs
  (green = hit, blue = missed GT, red = spurious prediction).
- `eval/visualize_seams.py` — Blender seam rendering.

## Repository layout

```
meshtailor/            model package (train.py, inference.py, models/, data/, utils/)
  data/split_*.json    committed train/val/test splits
eval/                  ABF++/Blender unwrap + UV & structural metrics, GT evaluation
tools/                 preprocessing, v13 relabel, paired eval, oracle probe, finalize
tools/abf/             ABF++ unwrap C++ source (CMake) + calibration
scripts/               PowerShell drivers for the final v13 runs
huggingface/           model card used for the weight upload
```

## Notes & known gaps vs. the paper

- chart count 0.88×GT vs. paper's 0.91×GT; area std-log 1.25×GT vs. paper's 1.00×GT.
- Suspected causes: duplicate-transition "echo" tails on ~12% high-density garments,
  and the paper's additional TexVerse co-training data (not used here).
- Training uses `ss_prob=0` (no scheduled sampling), decimation target 1000 vertices
  (paper uses 2000-triangle / `T_max=400` budget on a different target distribution).

## License

MIT — see [LICENSE](LICENSE). This is an unofficial reproduction; it is not
affiliated with the MeshTailor authors.
