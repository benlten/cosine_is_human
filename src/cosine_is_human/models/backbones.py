"""Self-supervised ViT backbones (DINOv1, DINOv2, DINOv3).

Returns a callable that maps B×3×224×224 to B×D CLS-token embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    name: str
    hub_repo: str
    hub_name: str
    dim: int
    patch_size: int


BACKBONE_SPECS = {
    "dinov1": BackboneSpec("dinov1", "facebookresearch/dino:main", "dino_vitb16", 768, 16),
    "dinov2": BackboneSpec("dinov2", "facebookresearch/dinov2", "dinov2_vitb14", 768, 14),
    "dinov3": BackboneSpec("dinov3", "facebookresearch/dinov3", "dinov3_vitb16", 768, 16),
}


def load_backbone(name: str, device: str = "cuda") -> nn.Module:
    if name not in BACKBONE_SPECS:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_SPECS)}")
    spec = BACKBONE_SPECS[name]
    model = torch.hub.load(spec.hub_repo, spec.hub_name, pretrained=True)
    model.eval().to(device)
    return model


def cls_forward(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the CLS token embedding from any of the supported ViTs.

    Uses forward_features when available (DINOv2/v3) and falls back to plain
    forward for DINOv1 which returns CLS directly.
    """
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(x)
        if isinstance(out, dict):
            return out.get("x_norm_clstoken", out.get("cls_token", out.get("x")))
        return out
    return backbone(x)
