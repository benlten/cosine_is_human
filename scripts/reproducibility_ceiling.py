#!/usr/bin/env python3
"""Standalone reproducibility ceiling for any 2AFC benchmark, from vote counts.

For a triplet with L votes for one candidate and R for the other (V = L + R),
the fresh-panel repeatability is the probability that a new panel of V votes,
drawn with replacement from the observed proportions, reproduces the observed
majority; an exactly tied new panel counts half. It is a closed-form binomial
tail, so this script needs only the Python standard library.

Usage:
  python reproducibility_ceiling.py 2 1            # one pattern
  python reproducibility_ceiling.py --csv votes.csv --left left_count --right right_count

The CSV mode prints the per-pattern table and the vote-weighted overall
ceiling over all non-tie rows. On the NIGHTS-100k non-tie corpus this gives
about 89% overall, with 74.1% on 2-1, 84.4% on 3-1, and 94.2% on 4-1.
"""
import argparse
import csv
import sys
from collections import Counter
from math import comb


def repeatability(L: int, R: int):
    """P(fresh panel of L+R votes reproduces the observed majority), or None for ties."""
    V = L + R
    if V < 1 or L == R:
        return None
    p = max(L, R) / V
    q = 1.0 - p
    prob = sum(comb(V, k) * p**k * q ** (V - k) for k in range(V // 2 + 1, V + 1))
    if V % 2 == 0:
        prob += 0.5 * comb(V, V // 2) * p ** (V // 2) * q ** (V // 2)
    return prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("votes", nargs="*", type=int, help="L R vote counts for a single pattern")
    ap.add_argument("--csv", help="CSV with one row per triplet")
    ap.add_argument("--left", default="left_count", help="column with votes for the left candidate")
    ap.add_argument("--right", default="right_count", help="column with votes for the right candidate")
    args = ap.parse_args()

    if args.votes:
        if len(args.votes) != 2:
            ap.error("give exactly two integers: L R")
        L, R = args.votes
        p = repeatability(L, R)
        if p is None:
            print(f"{L}-{R}: tie, no observed majority")
        else:
            print(f"{max(L,R)}-{min(L,R)}: fresh-panel repeatability = {100*p:.1f}%")
        return 0

    if not args.csv:
        ap.error("give either L R or --csv")

    counts = Counter()
    with open(args.csv, newline="") as fh:
        for row in csv.DictReader(fh):
            counts[(int(float(row[args.left])), int(float(row[args.right])))] += 1

    per_pattern = Counter()
    per_pattern_sum = {}
    total_n, total_sum = 0, 0.0
    for (L, R), n in counts.items():
        p = repeatability(L, R)
        if p is None:
            continue
        key = (max(L, R), min(L, R))
        per_pattern[key] += n
        per_pattern_sum[key] = per_pattern_sum.get(key, 0.0) + p * n
        total_n += n
        total_sum += p * n

    for (hi, lo), n in sorted(per_pattern.items()):
        print(f"{hi}-{lo}: N={n:>7}  repeatability={100*per_pattern_sum[(hi,lo)]/n:6.1f}%")
    if total_n:
        print(f"overall (non-tie, N={total_n}): {100*total_sum/total_n:.1f}%")
    else:
        print("no non-tie rows found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
