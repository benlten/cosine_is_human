"""DreamSim-Scratch training shim.

This module documents how to reproduce the DreamSim-Ens-Scratch-100k row of
Table 1: clone the authors' public repo, point it at our fold CSVs, and run
their ``training/train.py`` unchanged. We do not re-implement the ensemble
trainer because doing so would diverge from the authors' released recipe,
defeating the apples-to-apples comparison the paper makes.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    ds_root = Path("third_party/dreamsim")
    if not ds_root.exists():
        print(
            "\n[DreamSim-Scratch] Clone the DreamSim repository first:\n"
            "  git clone https://github.com/ssundaram21/dreamsim third_party/dreamsim\n",
            flush=True,
        )
        return 2
    cfg = Path(args.config).resolve()
    cmd = [
        "python", "training/train.py", "--config", str(cfg),
    ]
    print("Running:", " ".join(cmd), "in", ds_root)
    env_cmd = ["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", "PYTHONPATH=" + str(ds_root)] + cmd
    return subprocess.call(env_cmd, cwd=str(ds_root))
