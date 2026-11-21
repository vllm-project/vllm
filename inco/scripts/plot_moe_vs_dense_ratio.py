# SPDX-License-Identifier: Apache-2.0
"""Per-step throughput scaling for a dense vs an MoE model on one H100.

Two panels over the same concurrency steps:

* top: the raw throughput ratio between adjacent concurrencies, which is
  inflated early because the sweep doubles concurrency there;
* bottom: that ratio divided by the concurrency ratio, which removes the step
  size so the panels' shapes are comparable across the whole sweep.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "moe-vs-dense-ratio-plots" / "moe-vs-dense-ratio.png"

RUNS = (
    ("Qwen3-4B (dense)", "dense-4b-unpinned-1024-512"),
    ("Qwen3-30B-A3B (MoE)", "chat-1024-512"),
)


def load(run: str) -> list[tuple[int, float]]:
    with (RESULTS / run / "pareto.csv").open() as handle:
        rows = [
            (int(float(r["concurrency"])), float(r["output_token_throughput"]))
            for r in csv.DictReader(handle)
        ]
    return sorted(rows)


def steps(series: list[tuple[int, float]]):
    """(label, raw ratio, step-normalized ratio) for each adjacent pair."""
    out = []
    for (c0, t0), (c1, t1) in zip(series, series[1:]):
        ratio = t1 / t0
        out.append((f"{c0}→{c1}", ratio, ratio / (c1 / c0)))
    return out


def main() -> Path:
    curves = {label: steps(load(run)) for label, run in RUNS}
    labels = max(curves.values(), key=len)
    ticks = [label for label, _, _ in labels]
    index = {label: i for i, label in enumerate(ticks)}

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(14, 10), dpi=160, sharex=True, height_ratios=(1, 1)
    )
    for ax, component in ((top, 1), (bottom, 2)):
        for label, curve in curves.items():
            xs = [index[name] for name, _, _ in curve]
            ys = [point[component] for point in curve]
            ax.plot(xs, ys, marker="o", label=label)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Throughput growth per concurrency step: dense vs MoE\n"
        "ISL 1024 / OSL 512, 1x H100 80GB HBM3, bf16, TP=1",
    )
    top.set_title(
        "Raw ratio: how much total throughput each step adds\n"
        "Steps are not uniform (x2 to 16, then +8 to 80, +16 to 128, +32 to "
        "256), so this partly tracks step size",
        fontsize=11,
    )
    top.set_ylabel("throughput ratio\nT(c) / T(previous c)")
    bottom.set_title(
        "Same ratio divided by the concurrency ratio: throughput added per "
        "unit of concurrency added",
        fontsize=11,
    )
    bottom.set_ylabel(
        "per-step scaling efficiency\n(throughput ratio / concurrency ratio)"
    )
    bottom.set_xlabel("concurrency step")
    bottom.set_xticks(range(len(ticks)))
    bottom.set_xticklabels(ticks, rotation=45, ha="right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
