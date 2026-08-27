---
license: mit
library_name: pytorch
tags:
  - mesh-generation
  - garment
  - seam-prediction
  - pytorch
---

# MeshTailor reproduction checkpoint (paper protocol, 100k train)

Best checkpoint of our reproduction of **MeshTailor: Cutting Seams via
Generative Mesh Traversal** (arXiv:2603.27309). Companion code repository:
[github.com/Xinghan-Wang/meshtailor](https://github.com/Xinghan-Wang/meshtailor).

- File: `best_paper100k.pt` (~1.14 GB), a full PyTorch training checkpoint
  (model weights + optimizer state + config, saved by `meshtailor/train.py`).
- Trained on 100k garments from GarmentCodeDataset with paper-style "maximal
  chain" labels (paper App. B.1) and the paper sequence protocol.
- 10k test results: macro edge recall 0.846, precision 0.931, chart count
  0.88×GT, PartUV overall distortion within 4% of GT (paper reports parity).

## Usage

```python
import torch
# The checkpoint contains non-tensor objects (optimizer state/config), so it must
# be loaded with weights_only=False. Only load checkpoints from trusted sources.
ckpt = torch.load("best_paper100k.pt", weights_only=False)
# ckpt["model"] holds the state dict; load with meshtailor.models.model.MeshTailor
# (see the companion GitHub repository for the full pipeline).
```

Recommended inference config: `temperature=0.1`, no penalties (p0 protocol),
`--bf16` on modern GPUs.

## External encoder

MeshTailor uses the frozen point-cloud encoder from
[NeuralCarver/Michelangelo](https://github.com/NeuralCarver/Michelangelo).
Clone the upstream repository as `Michelangelo/` at the root of the companion
MeshTailor repository, then download the two required weight directories:

```bash
git clone https://github.com/NeuralCarver/Michelangelo.git Michelangelo
git -C Michelangelo checkout 6d83b0b
hf download Maikou/Michelangelo \
    checkpoints/aligned_shape_latents/shapevae-256.ckpt \
    --local-dir Michelangelo
hf download Maikou/Michelangelo \
    --include "checkpoints/clip/clip-vit-large-patch14/*" \
    --local-dir Michelangelo
```

Michelangelo is not included in this model repository. Its code remains under
the upstream GPL-3.0 license, and its pretrained weights retain their upstream
terms. See the companion repository README for the complete setup and data
preprocessing instructions.

## License

MIT (code); dataset rights belong to GarmentCodeDataset's authors.
