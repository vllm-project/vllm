# SPDX-License-Identifier: Apache-2.0
"""Expert-activation curve drawn on the throughput sweep's own x-axis.

The original plot used log-spaced batch sizes (1..256 by doubling), which
cannot be laid next to `plot_moe_vs_dense_ratio.py` -- that one steps x2 to 16
and then +8, +16, +32. Here the grid is read straight off the sweep's
`pareto.csv` files and the ticks are evenly spaced, so a point at concurrency
c sits at the same horizontal position in both figures.

Caveat worth remembering when reading the two together: x here is the decode
batch size of a single forward pass, whereas the throughput sweep's x is
client concurrency. With chunked prefill they are not the same number.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INCO = Path(__file__).resolve().parents[1]
RESULTS = INCO / "results"
EXPERTS = RESULTS / "experts"
OUT = EXPERTS / "expert-activation-matched-x.png"

# The sweeps whose concurrency points define the shared x-axis.
SWEEPS = ("dense-4b-unpinned-1024-512", "chat-1024-512")

RUNS = (
    ("unpruned Qwen3-30B-A3B (128 experts)", "Qwen3-30B-A3B-Instruct-2507"),
    ("REAP 50% pruned (64 experts)", "layerwise_reap-renorm_true-seed_42-0.50"),
)


def sweep_grid(sweeps=SWEEPS, results: Path = RESULTS) -> list[int]:
    """Concurrency points of the throughput sweeps, in ascending order."""
    points: set[int] = set()
    for sweep in sweeps:
        with (results / sweep / "pareto.csv").open() as handle:
            points.update(
                int(float(row["concurrency"])) for row in csv.DictReader(handle)
            )
    return sorted(points)


def align(curve: dict[int, float], grid: list[int]):
    """Place a batch-size -> experts curve onto `grid`'s tick positions.

    Returns (positions, values) for the grid entries the curve measured, so a
    partially measured curve draws as a shorter line rather than silently
    resampling onto the wrong ticks.
    """
    return tuple(
        zip(*[(i, curve[c]) for i, c in enumerate(grid) if c in curve])
    ) or ([], [])


def load(name: str, experts: Path = EXPERTS) -> tuple[dict[int, float], int]:
    payload = json.loads((experts / f"{name}.json").read_text())
    curve = {int(k): v for k, v in payload["experts_per_layer"].items()}
    return curve, payload["num_experts"]


def main() -> Path:
    grid = sweep_grid()
    runs = {label: load(name) for label, name in RUNS}

    ceiling = max(n for _, n in runs.values())

    fig, ax = plt.subplots(figsize=(14, 6), dpi=160)
    for i, (label, (curve, n_experts)) in enumerate(runs.items()):
        color = f"C{i}"
        xs, ys = align(curve, grid)
        ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.axhline(n_experts, color=color, linestyle="--", linewidth=1, alpha=0.4)
        ax.annotate(
            f"{n_experts} experts",
            (len(grid) - 1, n_experts),
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=8,
            color="gray",
            ha="right",
        )

    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([str(c) for c in grid], rotation=45, ha="right")
    ax.set_xlabel("decode batch size  (tokens per forward step)")
    ax.set_ylabel("distinct experts activated per layer")
    ax.set_title(
        "Qwen3-30B-A3B-Instruct-2507, top-8 routing, 48 layers, "
        "256 seqs x 16 tok, 64 draws\n"
        "x-axis matched to the throughput sweep (x2 to 16, then +8, +16, +32)",
    )
    ax.set_ylim(0, ceiling * 1.12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
