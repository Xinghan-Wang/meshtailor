# MeshTailor — Reproduction

An unofficial, from-scratch reproduction of

> **MeshTailor: Cutting Seams via Generative Mesh Traversal**
> Ma, Yan, Zhang, Huang — arXiv:2603.27309 (2026, official code TBA)

MeshTailor predicts the sewing seams of a garment directly on its triangle
mesh. Instead of working in UV space, an autoregressive pointer network walks
the mesh vertices and decodes seam chains one by one, conditioned on geometry
and panel-graph encodings. This repository contains the complete pipeline we
used to reproduce the paper — data preprocessing, training, evaluation against
a real ABF++ unwrap — together with our best checkpoint.

**Headline numbers on the 10k test split** (see [Results](#results)): edge
recall **0.85** / precision **0.93**, chart count **0.88×** ground truth
(paper reports 0.91×), UV area distortion **1.06×** ground truth.

---

## Quick start (inference only)

Requirements: Python 3.10, a CUDA GPU, and the packages in
[`requirements.txt`](requirements.txt).

```bash
# 1. install
conda create -n meshtailor python=3.10 -y && conda activate meshtailor
pip install -r requirements.txt          # install a CUDA build of torch first

# 2. download the checkpoint (~1.14 GB) from Hugging Face
pip install -U "huggingface_hub[cli]"
hf download XingHan-WANG/meshtailor best_paper100k.pt --local-dir checkpoints

# 3. generate seams for the test split
python meshtailor/inference.py \
    --ckpt checkpoints/best_paper100k.pt \
    --split test --out_dir test_outputs --temperature 0.1 --bf16
```

Each garment produces `test_outputs/<gid>/mesh.obj` and `seam.json`
(the generated seam chains). Note that inference requires the processed test
data (`processed_data_seamless_maximal/`), which you can build with the
preprocessing steps below.

## Results

Best checkpoint on the 10k test split (10,000 garments), evaluated with a
geogram-based ABF++ unwrap:

| Metric | Ours | GT | Ours / GT | Paper / GT |
|---|---|---|---|---|
| area std-log ↓ | 0.6957 | 0.5551 | 1.25× | 1.00× |
| area mean\|r−1\| ↓ | 0.8483 | 0.7972 | 1.06× | — |
| compactness ↑ | 0.5232 | 0.5236 | 1.00× | 1.00× |
| convexity ↑ | 0.8578 | 0.8426 | 1.02× | 1.00× |
| seam length / area ↓ | 2.9798 | 3.2846 | 0.91× | 0.94× |
| jaggedness ↓ | 0.0014 | 0.0013 | 1.10× | — |
| chart count ↓ | 9.582 | 10.951 | **0.88×** | 0.91× |
| macro edge recall ↑ | **0.8465** | — | — | — |
| macro edge precision ↑ | **0.9314** | — | — | — |
| unique seam edges | 346.6 | 397.4 | 0.87× | — |
| chains per garment | 12.69 | 12.9 | ✓ | — |

Known gaps vs. the paper: chart count 0.88× vs 0.91×, area distortion 1.25×
vs 1.00× — likely caused by duplicate-transition tails on ~12% of
high-density garments and by the paper's additional TexVerse co-training
data, which we did not use.

## Repository layout

```
meshtailor/        model + training/inference entry points
  models/          pointer decoder, geometry & panel encoders, fusion
  data/            dataset loader, seam extraction, committed splits
eval/              ABF++/Blender unwrap, UV & structural metrics, GT evaluation
tools/             preprocessing, chain relabeling, eval summary, visualization
tools/abf/         ABF++ unwrap C++ source (CMake, built against geogram)
scripts/           one-shot PowerShell drivers: train / postcheck / eval
huggingface/       model card used for the checkpoint upload
```

## Full reproduction

### Extra requirements

Everything was run on Windows 11 + WSL2 (Ubuntu), one RTX 5080 16 GB,
PyTorch 2.13 + PyG 2.8. Two native tools live inside WSL:

| Tool | Role | Default location | Override |
|---|---|---|---|
| SeamAwareDecimater | seam-preserving decimation (preprocessing) | `/root/seam-decimater/build/decimater` | `SEAMLESS_DECIMATER` |
| ABF++ unwrap (`tools/abf/`) | UV evaluation | `/root/abf_toolkit/tool/build/abf_unwrap` | `ABF_BIN`, `GEO_LIB` |

The dataset is GarmentCodeDataset (v2 archives,
`garments_5000_*/default_body/data.tar.gz`): obtain it from its official
release — it is not redistributed here.

### Step 1 — Build the data (128,974 garments)

```powershell
# seam-preserving decimation -> processed_data_seamless/
python tools/preprocess_seamless.py `
    --data_root GarmentCodeData/GarmentCodeData_v2 `
    --output_dir processed_data_seamless --target 1000 --par 16

# maximal chain decomposition (paper App. B.1) -> processed_data_seamless_maximal/
python tools/repair_seam_chains.py `
    --src_dir processed_data_seamless `
    --out_dir processed_data_seamless_maximal `
    --split_file meshtailor/data/split_seamless_128k.json `
    --mode maximal --workers 12
```

### Step 2 — Train (~24 h on an RTX 5080)

```powershell
./scripts/train.ps1        # writes checkpoints/best_paper100k.pt
./scripts/postcheck.ps1    # sanity checks on 150 test garments
```

Training config: 100k train split, batch 4 × grad-accum 8, lr 1e-4,
dropout 0.1, grad clip 1.0, `T_max=2000`, paper sequence protocol,
seed 20260818, early stopping patience 5.

### Step 3 — Evaluate on the full 10k test split

```powershell
./scripts/eval_full10k.ps1
```

Runs GT unwrap → model inference → ABF++ evaluation → metric summary
(`checkpoints/full10k_summary.txt`).

### Visualization

- `tools/export_compare_obj.py` — per-garment comparison OBJs:
  green = correctly predicted seams, blue = missed GT seams, red = spurious.
- `eval/visualize_seams.py` — Blender rendering of seams on the mesh.

## License & disclaimer

MIT (see [LICENSE](LICENSE)). This is an independent research reproduction
and is not affiliated with the MeshTailor authors. Dataset rights belong to
the authors of GarmentCodeDataset.
