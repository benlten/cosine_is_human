"""Error distribution: where do the model's errors live?

For each method, decomposes the set of wrong predictions over agreement
patterns. Shows that the best methods' errors concentrate on weak-majority
triplets (2-1, 3-1, 4-1) where humans themselves were close to a coin-flip.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUCKETS = ("Weak (2-1/3-1/4-1)", "Strong X-1 (X>=5)", "Unanimous (X-0)")
BUCKET_COLORS = {
    "Weak (2-1/3-1/4-1)": "#d62728",
    "Strong X-1 (X>=5)": "#ff7f0e",
    "Unanimous (X-0)": "#2ca02c",
}


def bucket_of(pattern: str) -> str:
    large, small = (int(x) for x in pattern.split("-"))
    if small == 0:
        return "Unanimous (X-0)"
    if large >= 5:
        return "Strong X-1 (X>=5)"
    return "Weak (2-1/3-1/4-1)"


def _key(path: Path) -> str:
    m = re.match(r"preds_fold(\d+)_(.+)\.csv$", path.name)
    return m.group(2) if m else path.stem


def load_preds(preds_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(glob.glob(str(preds_dir / "preds_fold*_*.csv"))):
        key = _key(Path(p))
        df = pd.read_csv(p)
        if "agreement" not in df.columns or "correct" not in df.columns:
            continue
        out.setdefault(key, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def run(args: argparse.Namespace) -> int:
    preds = load_preds(Path(args.preds_dir))
    methods = sorted(preds.keys())
    err_weak, err_strong, err_unan, support = [], [], [], []
    for m in methods:
        df = preds[m]
        df = df.assign(bucket=df.agreement.map(bucket_of))
        wrong = df[df.correct.astype(bool) == False]  # noqa: E712
        if len(wrong) == 0:
            err_weak.append(0); err_strong.append(0); err_unan.append(0); support.append(0)
            continue
        share = wrong.bucket.value_counts(normalize=True)
        err_weak.append(share.get("Weak (2-1/3-1/4-1)", 0) * 100)
        err_strong.append(share.get("Strong X-1 (X>=5)", 0) * 100)
        err_unan.append(share.get("Unanimous (X-0)", 0) * 100)
        # Fraction of wrong predictions where at least one human agreed with model
        wrong = wrong.assign(min_share=wrong.agreement.map(
            lambda p: int(p.split("-")[1]) / sum(int(x) for x in p.split("-"))
        ))
        support.append(float((wrong.min_share > 0).mean() * 100))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), width_ratios=[1.2, 1])
    x = np.arange(len(methods))
    axes[0].bar(x, err_weak, color=BUCKET_COLORS["Weak (2-1/3-1/4-1)"], label="Weak (2-1/3-1/4-1)")
    axes[0].bar(x, err_strong, bottom=err_weak, color=BUCKET_COLORS["Strong X-1 (X>=5)"], label="Strong X-1 (X>=5)")
    axes[0].bar(x, err_unan, bottom=np.array(err_weak) + np.array(err_strong),
                color=BUCKET_COLORS["Unanimous (X-0)"], label="Unanimous (X-0)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_ylabel("Share of model errors (%)")
    axes[0].set_title("Where model errors live on NIGHTS-100k")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, support, color="#1f77b4")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].set_ylabel("Errors where ≥1 human sided with model (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("When wrong, did a human agree?")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0
