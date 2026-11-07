"""Reproducibility ceiling from per-triplet vote counts.

Primary estimator (fresh-panel repeatability, used in the camera-ready):
for a triplet with L votes for left and R votes for right (total V=L+R),
draw V new votes with replacement from the observed proportions and ask
whether the new panel's majority matches the observed one; a tied new panel
counts half. This is a closed-form binomial tail, so no simulation is needed.
On NIGHTS-100k non-tie it gives about 89% overall and 74% / 84% / 94% on the
2-1 / 3-1 / 4-1 patterns. No model reaches this ceiling, so gaps smaller than
the distance to it are not separable from label noise.

Secondary estimator (split-half agreement, kept for the appendix): split the
V votes into two disjoint halves and ask whether the halves' majorities
agree (exact via the hypergeometric PMF). This is more conservative because
each half has only half the votes; it is reported in the appendix as a
companion number, not as the paper's ceiling.

Works for any 2AFC benchmark with integer vote counts per triplet.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom


def agreement_probability(L: int, R: int) -> float | None:
    """Exact P(split-half majorities agree) via the hypergeometric PMF.

    Ties (equal L and R in a half) are resolved by a random coin flip, which
    contributes 0.5 to the expected agreement.
    """
    V = L + R
    if V < 2:
        return None
    V1 = V // 2
    V2 = V - V1
    p_agree = 0.0
    for L1 in range(max(0, L - V2), min(V1, L) + 1):
        p_L1 = hypergeom.pmf(L1, V, L, V1)
        R1 = V1 - L1
        L2 = L - L1
        R2 = R - R1
        m1 = 1 if L1 > R1 else (-1 if L1 < R1 else 0)
        m2 = 1 if L2 > R2 else (-1 if L2 < R2 else 0)
        if m1 == 0 and m2 == 0:
            agree = 0.5
        elif m1 == 0 or m2 == 0:
            agree = 0.5
        elif m1 == m2:
            agree = 1.0
        else:
            agree = 0.0
        p_agree += p_L1 * agree
    return p_agree


def repeatability_probability(L: int, R: int) -> float | None:
    """P(a fresh panel of V = L + R votes reproduces the observed majority).

    The new panel is drawn with replacement from the observed proportions
    (each vote goes to the observed majority side with probability
    max(L, R) / V). The majority is reproduced when the majority side gets
    more than V/2 of the new votes; an exactly tied new panel counts 0.5.
    Closed form: a binomial tail. Tied triplets (L == R) have no observed
    majority and return None.
    """
    V = L + R
    if V < 1 or L == R:
        return None
    p = max(L, R) / V
    prob = float(binom.sf(V // 2, V, p))  # P(k >= floor(V/2) + 1)
    if V % 2 == 0:
        prob += 0.5 * float(binom.pmf(V // 2, V, p))
    return prob


def repeatability_over_counts(counts: dict[tuple[int, int], int]) -> dict:
    """Aggregate fresh-panel repeatability over a (L, R) -> count mapping."""
    per_pattern: dict[str, dict] = {}
    non_tie_n = 0
    non_tie_sum = 0.0
    for (L, R), n in counts.items():
        p = repeatability_probability(L, R)
        if p is None:
            continue
        non_tie_n += n
        non_tie_sum += p * n
        pattern = f"{max(L, R)}-{min(L, R)}"
        rec = per_pattern.setdefault(pattern, {"N": 0, "sum": 0.0})
        rec["N"] += n
        rec["sum"] += p * n
    for rec in per_pattern.values():
        rec["repeatability"] = rec.pop("sum") / rec["N"] * 100

    def _cat(pred):
        s, n = 0.0, 0
        for (L, R), cnt in counts.items():
            if pred(L, R):
                p = repeatability_probability(L, R)
                if p is not None:
                    s += p * cnt
                    n += cnt
        return (s / n * 100, n) if n else (0.0, 0)

    unan, unan_n = _cat(lambda L, R: min(L, R) == 0 and max(L, R) > 0)
    strong, strong_n = _cat(lambda L, R: min(L, R) == 1 and max(L, R) >= 5)
    weak, weak_n = _cat(lambda L, R: (max(L, R), min(L, R)) in {(2, 1), (3, 1), (4, 1), (3, 2)})

    return {
        "overall_non_tie_pct": non_tie_sum / non_tie_n * 100 if non_tie_n else 0.0,
        "unanimous_pct": unan,
        "unanimous_N": unan_n,
        "strong_pct": strong,
        "strong_N": strong_n,
        "weak_pct": weak,
        "weak_N": weak_n,
        "per_pattern": per_pattern,
        "N_non_tie": non_tie_n,
    }


def ceiling_over_counts(counts: dict[tuple[int, int], int]) -> dict:
    """Given a (L, R) -> triplet-count mapping, return ceiling summaries.

    Returns a dict with overall ceiling, non-tie ceiling, per-pattern ceilings,
    and category breakdowns used in the paper.
    """
    per_pattern: dict[str, dict] = {}
    total_n = 0
    total_sum = 0.0
    non_tie_n = 0
    non_tie_sum = 0.0
    for (L, R), n in counts.items():
        p = agreement_probability(L, R)
        if p is None:
            continue
        total_n += n
        total_sum += p * n
        if L != R:
            non_tie_n += n
            non_tie_sum += p * n
        pattern = f"{max(L, R)}-{min(L, R)}"
        if pattern not in per_pattern:
            per_pattern[pattern] = {"N": 0, "ceiling_sum": 0.0}
        per_pattern[pattern]["N"] += n
        per_pattern[pattern]["ceiling_sum"] += p * n

    for pat, rec in per_pattern.items():
        rec["ceiling"] = rec.pop("ceiling_sum") / rec["N"] * 100

    def _cat_sum(pred):
        s, n = 0.0, 0
        for (L, R), cnt in counts.items():
            if pred(L, R):
                p = agreement_probability(L, R)
                if p is not None:
                    s += p * cnt
                    n += cnt
        return (s / n * 100, n) if n else (0.0, 0)

    unan_ceil, unan_n = _cat_sum(lambda L, R: min(L, R) == 0 and max(L, R) > 0)
    strong_ceil, strong_n = _cat_sum(lambda L, R: min(L, R) == 1 and max(L, R) >= 5)
    weak_ceil, weak_n = _cat_sum(lambda L, R: (max(L, R), min(L, R)) in {(2, 1), (3, 1), (4, 1), (3, 2)})

    return {
        "overall_all_triplets_pct": total_sum / total_n * 100 if total_n else 0.0,
        "overall_non_tie_pct": non_tie_sum / non_tie_n * 100 if non_tie_n else 0.0,
        "unanimous_pct": unan_ceil,
        "unanimous_N": unan_n,
        "strong_pct": strong_ceil,
        "strong_N": strong_n,
        "weak_pct": weak_ceil,
        "weak_N": weak_n,
        "per_pattern": per_pattern,
        "N_total": total_n,
        "N_non_tie": non_tie_n,
    }


def ceiling_for_nights(csv_path: Path) -> dict:
    """Both ceiling estimators for NIGHTS-100k from its raw vote CSV."""
    df = pd.read_csv(csv_path)
    left_counts = (df["left_vote"] * df["votes"]).round().astype(int)
    right_counts = df["votes"].astype(int) - left_counts
    counts: Counter[tuple[int, int]] = Counter()
    for L, R in zip(left_counts, right_counts):
        counts[(int(L), int(R))] += 1
    return {
        "fresh_panel": repeatability_over_counts(dict(counts)),
        "split_half": ceiling_over_counts(dict(counts)),
    }


def ceiling_for_bapps(root: Path) -> dict:
    """Both ceiling estimators for BAPPS 2AFC validation (5 voters/triplet)."""
    from cosine_is_human.data.bapps import iter_bapps_judges

    counts: Counter[tuple[int, int]] = Counter()
    for triplet in iter_bapps_judges(Path(root), split="val"):
        counts[(triplet.vote_p0, triplet.vote_p1)] += 1
    return {
        "fresh_panel": repeatability_over_counts(dict(counts)),
        "split_half": ceiling_over_counts(dict(counts)),
    }


def run(args: argparse.Namespace) -> int:
    if args.dataset == "nights":
        # expects environment variable or conventional path
        csv = Path("data/nights_unfiltered/nights_raw_2afc.csv")
        summary = ceiling_for_nights(csv)
    else:
        summary = ceiling_for_bapps(Path("data/bapps"))

    out = args.out
    payload = json.dumps(summary, indent=2)
    if out == "-":
        print(payload)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(payload)
        print(f"Saved {out}")
    return 0
