# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Paired GSM8K comparison across runs: accuracy +- stderr and exact (binomial) McNemar
#   on every pair. python paired_stats.py NAME=path/to/gsm8k_records.jsonl [NAME=...]
#   [--pairs A1:C1,A2:C2]
import itertools
import json
import math
import sys


def load(p):
    return {json.loads(ln)["i"]: bool(json.loads(ln)["ok_strict"]) for ln in open(p)}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main():
    runs = {}
    pairs = None
    for arg in sys.argv[1:]:
        if arg.startswith("--pairs="):
            pairs = [tuple(x.split(":")) for x in arg.split("=", 1)[1].split(",")]
            continue
        k, v = arg.split("=", 1)
        runs[k] = load(v)
    print(f"{'run':<12} {'correct':>8} {'n':>5} {'acc':>7} {'stderr':>7}")
    for k, r in runs.items():
        n, c = len(r), sum(r.values())
        p = c / n
        print(
            f"{k:<12} {c:>8} {n:>5} {p:>7.4f} {math.sqrt(p * (1 - p) / (n - 1)):>7.4f}"
        )
    print(
        f"\n{'pair (X vs Y)':<20} {'X-only':>7} {'Y-only':>7} {'both':>6} "
        f"{'p_exact':>9}"
    )
    for x, y in pairs or itertools.combinations(runs, 2):
        common = sorted(set(runs[x]) & set(runs[y]))
        b = sum(runs[x][i] and not runs[y][i] for i in common)
        c = sum(runs[y][i] and not runs[x][i] for i in common)
        both = sum(runs[x][i] and runs[y][i] for i in common)
        print(
            f"{x + ' vs ' + y:<20} {b:>7} {c:>7} {both:>6} {mcnemar_exact(b, c):>9.4f}"
        )


if __name__ == "__main__":
    main()
