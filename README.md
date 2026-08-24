# HEFGNet

Model implementation for **HEFGNet: Hierarchically Enhanced Features-Guided
Network for Fine-Grained Food Image Segmentation**.

This repository contains only the network architecture. Dataset preparation,
training, evaluation, and trained model weights are not included.

## Model components

| Manuscript component | Python class |
| --- | --- |
| Dual-Pooling Squeeze Attention | `DualPoolingSqueezeAttention`, `DPSABlock` |
| Multi-Source Contextual Edge Enhancement | `MSCEE` |
| Detail Feature Path | `DetailFeaturePath` |
| Hierarchical feature fusion | `HierarchicalFeatureFusion` |
| Complete network | `HEFGNet` |

The public names replace experimental identifiers used during model
development. `HEFGNet.load_checkpoint()` can still convert checkpoints saved by
the earlier implementation.

## Installation

```bash
pip install -r requirements.txt
```

The experiments used PyTorch 2.0.0 and CUDA 11.8. The ConvNeXt V2 backbone is
adapted from the
[official implementation](https://github.com/facebookresearch/ConvNeXt-V2).

## Usage

```python
import torch

from hefgnet import HEFGNet

model = HEFGNet(num_classes=104).eval().cuda()
image = torch.randn(1, 3, 768, 768, device="cuda")

with torch.inference_mode():
    logits = model(image)

print(logits.shape)  # torch.Size([1, 104, 768, 768])
```

To initialize the backbone from the official ConvNeXt V2-Base ImageNet-22K
checkpoint:

```python
from hefgnet import build_hefgnet

model = build_hefgnet(
    num_classes=104,
    backbone_weights="convnextv2_base_22k_384_ema.pt",
)
```

Download the checkpoint from the
[official ConvNeXt V2 release](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt).

## Citation

Please cite the HEFGNet paper if this implementation is useful in your
research. Complete publication metadata will be added after publication.
