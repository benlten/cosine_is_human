"""Drop chart: X-0 unanimous subset accuracy vs full NIGHTS-100k accuracy."""
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


def run(args: argparse.Namespace) -> int:
    per_method: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(glob.glob("results/preds_fold*_*.csv")):
        per_method.setdefault(_key(Path(p)), []).append(pd.read_csv(p))
    methods, unan_accs, full_accs = [], [], []
    for m, frames in per_method.items():
        df = pd.concat(frames, ignore_index=True)
        if "agreement" not in df.columns:
            continue
        unan = df[df.agreement.str.endswith("-0")]
        methods.append(m)
        unan_accs.append(unan.correct.mean() * 100 if len(unan) else 0)
        full_accs.append(df.correct.mean() * 100)

    import numpy as np
    order = np.argsort(-np.array(full_accs))
    methods = [methods[i] for i in order]
    unan_accs = [unan_accs[i] for i in order]
    full_accs = [full_accs[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    w = 0.4
    ax.bar(x - w/2, unan_accs, w, label="NIGHTS-20k-unan (X-0 subset)", color="#1f77b4")
    ax.bar(x + w/2, full_accs, w, label="Full NIGHTS-100k non-tie", color="#d62728")
    ax.axhline(96.2, ls="--", color="green", alpha=0.7, label="Published DreamSim 96.2%")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Every method loses 10–17 pp on the full NIGHTS-100k corpus")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0
