---
license: mit
library_name: pytorch
tags:
  - mesh-generation
  - garment
  - seam-prediction
  - pytorch
---

# MeshTailor v13 (paper100k) — reproduction checkpoint

Best checkpoint of our reproduction of **MeshTailor: Cutting Seams via
Generative Mesh Traversal** (arXiv:2603.27309).

- File: `best_v13paper100k.pt` (~1.14 GB), a full PyTorch training checkpoint
  (model weights + optimizer state + config, saved by `meshtailor/train.py`).
- Trained on 100k garments from GarmentCodeDataset with paper-style "maximal
  chain" labels (v13 relabel) and the paper sequence protocol.
- 10k test results: macro edge recall 0.846, precision 0.931, chart count
  0.88×GT, area mean|r−1| 1.06×GT.

## Usage

```python
import torch
# The checkpoint contains non-tensor objects (optimizer state/config), so it must
# be loaded with weights_only=False. Only load checkpoints from trusted sources.
ckpt = torch.load("best_v13paper100k.pt", weights_only=False)
# ckpt["model"] holds the state dict; load with meshtailor.models.model.MeshTailor
# (see the companion GitHub repository for the full pipeline).
```

Recommended inference config: `temperature=0.1`, no penalties (p0 protocol),
`--bf16` on modern GPUs.

## License

MIT (code); dataset rights belong to GarmentCodeDataset's authors.
