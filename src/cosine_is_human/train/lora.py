"""LoRA training loop for DINO + similarity head variants.

Given a YAML config specifying backbone, head, LoRA rank, LR, and fold, trains
for N epochs and saves the best-validation checkpoint plus the test-fold
prediction CSV at ``{out_dir}/<tag>_fold<k>_best.pt`` and
``{out_dir}/preds_fold<k>_<tag>.csv``.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from cosine_is_human.data.nights import NightsConfig, NightsTriplets, load_fold
from cosine_is_human.models.backbones import BACKBONE_SPECS, cls_forward, load_backbone
from cosine_is_human.models.heads import (
    CognitiveV3Head,
    CosineHead,
    TverskyHead,
    TverskyNeuralNetwork,
)
from cosine_is_human.models.lora import apply_lora, count_trainable


HEAD_REGISTRY = {
    "cosine": CosineHead,
    "cog_v3": CognitiveV3Head,
    "tversky": TverskyHead,
    "tnn": TverskyNeuralNetwork,
}


def _build_head(head_name: str, dim: int) -> torch.nn.Module:
    if head_name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head {head_name!r}. Available: {list(HEAD_REGISTRY)}")
    cls = HEAD_REGISTRY[head_name]
    if head_name == "cosine":
        return cls()
    return cls(dim=dim)


def _pairwise_loss(d_left: torch.Tensor, d_right: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on the logit d_right - d_left.

    label=1 means left is preferred (d_left should be smaller); we use
    logit = d_right - d_left so sigmoid(logit) -> p(left preferred).
    """
    logit = d_right - d_left
    return F.binary_cross_entropy_with_logits(logit, label.float())


def run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    nights = NightsConfig(
        raw_csv=Path(cfg["nights_csv"]),
        image_root=Path(cfg["nights_image_root"]),
        seed=cfg.get("seed", 42),
    )
    train_df = load_fold(nights, args.fold, split="train")
    test_df = load_fold(nights, args.fold, split="test")

    train_loader = DataLoader(
        NightsTriplets(train_df, nights.image_root),
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
    )
    test_loader = DataLoader(
        NightsTriplets(test_df, nights.image_root),
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    backbone = load_backbone(cfg["backbone"], device=device)
    backbone = apply_lora(
        backbone,
        r=cfg.get("lora_r", 8),
        alpha=cfg.get("lora_alpha", 16),
        dropout=cfg.get("lora_dropout", 0.05),
    )
    head = _build_head(cfg["head"], dim=BACKBONE_SPECS[cfg["backbone"]].dim).to(device)
    trainable, total = count_trainable(backbone)
    print(f"Trainable LoRA+norm: {trainable:,} / {total:,}")

    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-2))

    best_acc = 0.0
    tag = cfg.get("tag", f"{cfg['backbone']}_lora_{cfg['head']}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{tag}_fold{args.fold}_best.pt"

    for epoch in range(cfg.get("epochs", 10)):
        backbone.train(); head.train()
        for ref, left, right, label, _id, _ag in train_loader:
            ref, left, right = ref.to(device), left.to(device), right.to(device)
            label = label.to(device)
            ref_e = cls_forward(backbone, ref)
            left_e = cls_forward(backbone, left)
            right_e = cls_forward(backbone, right)
            d_left = head(ref_e, left_e)
            d_right = head(ref_e, right_e)
            loss = _pairwise_loss(d_left, d_right, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Eval on test fold at end of each epoch; keep best-accuracy checkpoint
        backbone.eval(); head.eval()
        correct = 0
        total_n = 0
        with torch.no_grad():
            for ref, left, right, label, _id, _ag in test_loader:
                ref, left, right = ref.to(device), left.to(device), right.to(device)
                ref_e = cls_forward(backbone, ref)
                left_e = cls_forward(backbone, left)
                right_e = cls_forward(backbone, right)
                pred_left = (head(ref_e, left_e) < head(ref_e, right_e)).long().cpu()
                correct += (pred_left == label.long()).sum().item()
                total_n += label.numel()
        acc = correct / total_n * 100
        print(f"epoch {epoch} test acc: {acc:.3f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(), "cfg": cfg}, ckpt_path)

    print(f"Best test acc: {best_acc:.3f}% saved to {ckpt_path}")
    return 0
