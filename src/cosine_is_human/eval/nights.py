"""Evaluate a method on NIGHTS-100k non-tie.

Persists per-triplet predictions to ``{out_dir}/preds_fold<k>_<method>.csv``
so that every downstream analysis (ceiling, kappa, mcnemar, errors) can be
regenerated without re-running inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cosine_is_human.data.nights import NightsConfig, NightsTriplets, load_fold


def _load_method(method: str, checkpoint: str | None, device: str):
    """Return a callable (ref_batch, cand_batch) -> distance_scalar_per_pair."""
    if method == "lpips":
        import lpips
        net = lpips.LPIPS(net="alex").to(device).eval()
        def dist(ref, cand):
            return net(ref, cand).squeeze()
        return dist
    if method in ("dinov1", "dinov2", "dinov3"):
        from cosine_is_human.models.backbones import load_backbone, cls_forward
        bb = load_backbone(method, device=device).eval()
        def dist(ref, cand):
            r = cls_forward(bb, ref)
            c = cls_forward(bb, cand)
            return 1.0 - torch.nn.functional.cosine_similarity(r, c, dim=-1)
        return dist
    if method == "dreamsim_pretrained":
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, device=device)
        def dist(ref, cand):
            return model(ref, cand).squeeze()
        return dist
    if checkpoint is None:
        raise ValueError(f"method {method!r} requires --checkpoint")
    raise NotImplementedError(
        f"Loading {method!r} from {checkpoint!r} not implemented in the skeleton; "
        "see train.lora and implement the inverse here."
    )


def eval_fold(method: str, checkpoint: str | None, fold: int, cfg: NightsConfig, device: str) -> pd.DataFrame:
    dist = _load_method(method, checkpoint, device)
    df = load_fold(cfg, fold, split="test")
    loader = DataLoader(NightsTriplets(df, cfg.image_root), batch_size=32, num_workers=4)
    rows = []
    with torch.no_grad():
        for ref, left, right, label, tid, agr in loader:
            ref, left, right = ref.to(device), left.to(device), right.to(device)
            d_left = dist(ref, left)
            d_right = dist(ref, right)
            pred_left = (d_left < d_right).long().cpu().numpy()
            label_np = label.numpy()
            for i in range(len(tid)):
                rows.append({
                    "id": int(tid[i]),
                    "pred": int(pred_left[i]),
                    "correct": int(pred_left[i] == label_np[i]),
                    "agreement": agr[i],
                })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    cfg = NightsConfig(
        raw_csv=Path("data/nights_unfiltered/nights_raw_2afc.csv"),
        image_root=Path("data/nights_unfiltered/dataset_unfiltered"),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [args.fold] if args.fold is not None else list(range(5))
    accs = []
    for f in folds:
        df = eval_fold(args.method, args.checkpoint, f, cfg, device)
        acc = df.correct.mean() * 100
        accs.append(acc)
        out_path = out_dir / f"preds_fold{f}_{args.method}.csv"
        df.to_csv(out_path, index=False)
        print(f"fold {f}: {acc:.3f}%  -> {out_path}")
    if len(accs) > 1:
        import numpy as np
        print(f"mean {np.mean(accs):.3f}%  std {np.std(accs):.3f}%")
    return 0
