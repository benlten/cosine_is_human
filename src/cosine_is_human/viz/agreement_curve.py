"""Agreement curve: accuracy vs. human-agreement fraction, per method."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _key(path: Path) -> str:
    m = re.match(r"preds_fold(\d+)_(.+)\.csv$", path.name)
    return m.group(2) if m else path.stem


def _agreement_fraction(pattern: str) -> float:
    L, R = (int(x) for x in pattern.split("-"))
    if L + R == 0:
        return 0.5
    return max(L, R) / (L + R)


def run(args: argparse.Namespace) -> int:
    per_method: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(glob.glob("results/preds_fold*_*.csv")):
        per_method.setdefault(_key(Path(p)), []).append(pd.read_csv(p))
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, frames in per_method.items():
        df = pd.concat(frames, ignore_index=True)
        if "agreement" not in df.columns:
            continue
        df = df.assign(frac=df.agreement.map(_agreement_fraction))
        curve = df.groupby("frac").correct.mean() * 100
        ax.plot(curve.index, curve.values, marker="o", label=m, linewidth=1.3)
    ax.set_xlabel("Fraction of human voters in majority")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy is monotonic in human agreement strength")
    ax.set_ylim(40, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0
