"""Command-line entry point for cosine_is_human.

Usage:
    cih <group> <command> [args]

Groups:
    train     Train a method on NIGHTS-100k (e.g. LoRA variants, DreamSim-Scratch).
    eval      Evaluate a trained or zero-shot method on NIGHTS or BAPPS.
    analyze   Post-hoc analyses (ceiling, kappa, mcnemar, error distribution).
    viz       Regenerate paper figures from predictions.

Examples:
    cih train lora   --config configs/lora_cog_v3.yaml --fold 0 --gpu 0
    cih eval  nights --checkpoint results/<ckpt>.pt --fold 0
    cih eval  bapps  --method dinov3_zs
    cih analyze ceiling --dataset nights
    cih analyze ceiling --dataset bapps
    cih analyze kappa   --out figures/kappa_matrix.pdf
    cih analyze errors  --out figures/error_distribution.pdf
    cih viz drop-chart  --out figures/drop_chart.pdf
"""
from __future__ import annotations

import argparse
import sys


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train a method on NIGHTS-100k")
    ssub = p.add_subparsers(dest="method", required=True)

    p_lora = ssub.add_parser("lora", help="LoRA head on DINOv1/v2/v3")
    p_lora.add_argument("--config", required=True, help="Path to YAML config")
    p_lora.add_argument("--fold", type=int, required=True, choices=range(5))
    p_lora.add_argument("--gpu", type=int, default=0)
    p_lora.add_argument("--out-dir", default="results")

    p_ds = ssub.add_parser("dreamsim-scratch", help="DreamSim ensemble from scratch")
    p_ds.add_argument("--config", required=True)
    p_ds.add_argument("--fold", type=int, required=True, choices=range(5))
    p_ds.add_argument("--gpu", type=int, default=0)
    p_ds.add_argument("--out-dir", default="results")


def _add_eval(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="Evaluate a method on NIGHTS or BAPPS")
    ssub = p.add_subparsers(dest="dataset", required=True)

    p_n = ssub.add_parser("nights", help="5-fold evaluation on NIGHTS-100k non-tie")
    p_n.add_argument("--method", choices=[
        "lpips", "dinov1", "dinov2", "dinov3",
        "dinov1_lora_cog_v3", "dinov2_lora_cog_v3", "dinov2_lora_cos",
        "dinov2_lora_cog_v3_strat", "dinov2_lora_tnn", "dinov3_lora_cog_v3",
        "dreamsim_pretrained", "dreamsim_scratch", "harmonized_dinov3",
    ], required=True)
    p_n.add_argument("--checkpoint", default=None,
                     help="Path to checkpoint for LoRA methods; ignored for zero-shot")
    p_n.add_argument("--fold", type=int, choices=range(5), default=None,
                     help="If omitted, evaluates all 5 folds")
    p_n.add_argument("--gpu", type=int, default=0)
    p_n.add_argument("--out-dir", default="results")

    p_b = ssub.add_parser("bapps", help="BAPPS 2AFC validation sanity check")
    p_b.add_argument("--method", required=True)
    p_b.add_argument("--checkpoint", default=None)
    p_b.add_argument("--gpu", type=int, default=0)
    p_b.add_argument("--out-dir", default="results")


def _add_analyze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Post-hoc analyses from persisted predictions")
    ssub = p.add_subparsers(dest="analysis", required=True)

    p_c = ssub.add_parser("ceiling", help="Split-half reproducibility ceiling (hypergeometric PMF)")
    p_c.add_argument("--dataset", choices=["nights", "bapps"], required=True)
    p_c.add_argument("--out", default="-", help='"-" for stdout or path to JSON output')

    p_k = ssub.add_parser("kappa", help="Pairwise Cohen's kappa across methods")
    p_k.add_argument("--preds-dir", default="results")
    p_k.add_argument("--out", default="figures/kappa_matrix.pdf")

    p_m = ssub.add_parser("mcnemar", help="Paired McNemar tests across methods")
    p_m.add_argument("--preds-dir", default="results")
    p_m.add_argument("--out", default="-")

    p_e = ssub.add_parser("errors", help="Distribution of model errors over agreement patterns")
    p_e.add_argument("--preds-dir", default="results")
    p_e.add_argument("--out", default="figures/error_distribution.pdf")

    p_i = ssub.add_parser("imagenet-split", help="Accuracy split by ImageNet vs non-ImageNet prompts")
    p_i.add_argument("--preds-dir", default="results")
    p_i.add_argument("--out", default="-")


def _add_viz(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("viz", help="Regenerate paper figures from persisted predictions")
    ssub = p.add_subparsers(dest="figure", required=True)

    ssub.add_parser("drop-chart", help="X-0 subset vs full 100k accuracy per method") \
        .add_argument("--out", default="figures/drop_chart.pdf")
    ssub.add_parser("agreement-curve", help="Accuracy vs. agreement fraction, per method") \
        .add_argument("--out", default="figures/agreement_curve.pdf")
    ssub.add_parser("errors", help="Error-distribution plot (same as analyze errors)") \
        .add_argument("--out", default="figures/error_distribution.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cih",
        description="Cosine is Human: NIGHTS-100k perceptual similarity audit.",
    )
    sub = parser.add_subparsers(dest="group", required=True)
    _add_train(sub)
    _add_eval(sub)
    _add_analyze(sub)
    _add_viz(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.group == "train":
        if args.method == "lora":
            from cosine_is_human.train.lora import run as run_lora
            return run_lora(args)
        if args.method == "dreamsim-scratch":
            from cosine_is_human.train.dreamsim_scratch import run as run_ds
            return run_ds(args)
    if args.group == "eval":
        if args.dataset == "nights":
            from cosine_is_human.eval.nights import run as run_n
            return run_n(args)
        if args.dataset == "bapps":
            from cosine_is_human.eval.bapps import run as run_b
            return run_b(args)
    if args.group == "analyze":
        from cosine_is_human.analysis import ceiling, kappa, mcnemar, errors, imagenet_split
        return {
            "ceiling": ceiling.run,
            "kappa": kappa.run,
            "mcnemar": mcnemar.run,
            "errors": errors.run,
            "imagenet-split": imagenet_split.run,
        }[args.analysis](args)
    if args.group == "viz":
        from cosine_is_human.viz import drop_chart, agreement_curve, errors as viz_errors
        return {
            "drop-chart": drop_chart.run,
            "agreement-curve": agreement_curve.run,
            "errors": viz_errors.run,
        }[args.figure](args)

    print(f"Unknown command: {args.group}", file=sys.stderr)
    return 2
