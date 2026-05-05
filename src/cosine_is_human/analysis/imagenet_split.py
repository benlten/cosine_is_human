"""Accuracy on ImageNet-prompted vs non-ImageNet-prompted triplets."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd


def _key(path: Path) -> str:
    m = re.match(r"preds_fold(\d+)_(.+)\.csv$", path.name)
    return m.group(2) if m else path.stem


def run(args: argparse.Namespace) -> int:
    # Load raw NIGHTS CSV for is_imagenet mapping
    raw = pd.read_csv(Path("data/nights_unfiltered/nights_raw_2afc.csv"))
    imnet = dict(zip(raw.id.astype(int), raw.is_imagenet.astype(bool)))

    preds_dir = Path(args.preds_dir)
    results: dict[str, dict] = {}
    per_method: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(glob.glob(str(preds_dir / "preds_fold*_*.csv"))):
        per_method.setdefault(_key(Path(p)), []).append(pd.read_csv(p))
    for m, frames in per_method.items():
        df = pd.concat(frames, ignore_index=True)
        df = df.assign(is_imagenet=df.id.astype(int).map(imnet))
        imnet_acc = df[df.is_imagenet == True].correct.mean() * 100  # noqa: E712
        non_acc = df[df.is_imagenet == False].correct.mean() * 100  # noqa: E712
        results[m] = {
            "imagenet_pct": float(imnet_acc),
            "non_imagenet_pct": float(non_acc),
            "delta_pp": float(imnet_acc - non_acc),
        }

    payload = json.dumps(results, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload)
        print(f"Saved {args.out}")
    return 0
