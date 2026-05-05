"""Pairwise Cohen's kappa across methods, with heatmap output."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _key(path: Path) -> str:
    m = re.match(r"preds_fold(\d+)_(.+)\.csv$", path.name)
    return m.group(2) if m else path.stem


def _load_vote_preds(preds_dir: Path) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for p in sorted(glob.glob(str(preds_dir / "preds_fold*_*.csv"))):
        key = _key(Path(p))
        df = pd.read_csv(p)
        col = "pred" if "pred" in df.columns else "pred_left"
        if col not in df.columns:
            continue
        prev = out.setdefault(key, {})
        for tid, pred in zip(df.id.astype(int), df[col].astype(int)):
            prev[tid] = int(pred)
    return out


def compute(preds_dir: Path) -> tuple[list[str], np.ndarray]:
    preds = _load_vote_preds(preds_dir)
    methods = sorted(preds.keys())
    n = len(methods)
    K = np.eye(n)
    for i, m_a in enumerate(methods):
        for j, m_b in enumerate(methods[i + 1:], start=i + 1):
            shared = sorted(set(preds[m_a]) & set(preds[m_b]))
            va = [preds[m_a][t] for t in shared]
            vb = [preds[m_b][t] for t in shared]
            k = cohen_kappa_score(va, vb)
            K[i, j] = k
            K[j, i] = k
    return methods, K


def run(args: argparse.Namespace) -> int:
    methods, K = compute(Path(args.preds_dir))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(K, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methods, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center",
                    color="black" if K[i, j] > 0.5 else "white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0
