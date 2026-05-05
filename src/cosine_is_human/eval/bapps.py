"""BAPPS 2AFC evaluation (zero-shot sanity check)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cosine_is_human.data.bapps import BAPPS_CATEGORIES, BappsTriplets
from .nights import _load_method


def run(args: argparse.Namespace) -> int:
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dist = _load_method(args.method, args.checkpoint, device)
    per_cat = {}
    all_correct, all_total = 0, 0
    for cat in BAPPS_CATEGORIES:
        ds = BappsTriplets(Path("data/bapps"), split="val")
        ds.triplets = [t for t in ds.triplets if t.category == cat]
        loader = DataLoader(ds, batch_size=32, num_workers=4)
        correct, total = 0, 0
        with torch.no_grad():
            for ref, p0, p1, label, _idx, _cat in loader:
                ref, p0, p1 = ref.to(device), p0.to(device), p1.to(device)
                d0 = dist(ref, p0)
                d1 = dist(ref, p1)
                pred = (d1 < d0).long().cpu().numpy()
                correct += int((pred == label.numpy()).sum())
                total += label.numel()
        acc = correct / total * 100 if total else 0
        per_cat[cat] = {"N": total, "accuracy_pct": acc}
        print(f"{cat:>14}  N={total:>6}  acc={acc:5.1f}%")
        all_correct += correct
        all_total += total
    overall = all_correct / all_total * 100
    print(f"{'ALL':>14}  N={all_total:>6}  acc={overall:5.1f}%")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bapps_{args.method}.csv"
    pd.DataFrame(per_cat).T.to_csv(out)
    print(f"Saved {out}")
    return 0
