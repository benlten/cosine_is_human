"""Pairwise McNemar tests on persisted per-triplet predictions."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _method_key_from_filename(path: Path) -> str:
    m = re.match(r"preds_fold(\d+)_(.+)\.csv$", path.name)
    if not m:
        return path.stem
    return m.group(2)


def load_all_preds(preds_dir: Path) -> dict[str, pd.DataFrame]:
    """Return a dict method_key -> concatenated DataFrame over folds with columns id, correct."""
    grouped: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(glob.glob(str(preds_dir / "preds_fold*_*.csv"))):
        key = _method_key_from_filename(Path(p))
        grouped.setdefault(key, []).append(pd.read_csv(p))
    return {k: pd.concat(v, ignore_index=True) for k, v in grouped.items()}


def mcnemar_pair(a: pd.Series, b: pd.Series) -> dict:
    """McNemar on boolean per-triplet correctness."""
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    a_only = int(((a) & (~b)).sum())
    b_only = int((~a & b).sum())
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    if a_only + b_only == 0:
        p = 1.0
        stat = 0.0
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
        p = float(1 - stats.chi2.cdf(stat, df=1))
    return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither, "chi2": float(stat), "p": p}


def run(args: argparse.Namespace) -> int:
    preds = load_all_preds(Path(args.preds_dir))
    methods = sorted(preds.keys())
    table = {}
    for i, m_a in enumerate(methods):
        for m_b in methods[i + 1:]:
            shared_ids = sorted(set(preds[m_a].id.astype(int)) & set(preds[m_b].id.astype(int)))
            a = preds[m_a].set_index("id").loc[shared_ids, "correct"]
            b = preds[m_b].set_index("id").loc[shared_ids, "correct"]
            table[f"{m_a}__vs__{m_b}"] = mcnemar_pair(a, b)

    payload = json.dumps({"methods": methods, "pairs": table}, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload)
        print(f"Saved {args.out}")
    return 0
