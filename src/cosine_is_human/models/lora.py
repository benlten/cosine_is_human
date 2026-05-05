"""LoRA wrappers shared by all trained methods.

Applies rank-r LoRA adapters to qkv and proj linear layers in every transformer
block of a ViT backbone. Layer-norm parameters are unfrozen alongside.
"""
from __future__ import annotations

from peft import LoraConfig, get_peft_model
import torch.nn as nn


def apply_lora(
    backbone: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("qkv", "proj"),
) -> nn.Module:
    """Apply LoRA to every block's qkv/proj and unfreeze layer-norm params."""
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    peft_model = get_peft_model(backbone, cfg)

    # Unfreeze all LayerNorm parameters (small, cheap, usually helps)
    for name, param in peft_model.named_parameters():
        if "norm" in name.lower():
            param.requires_grad = True

    return peft_model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
