# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import regex as re
from matplotlib.colors import TwoSlopeNorm

Q_LENS = [256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 32768]
SEQ_LENS = [2048, 4096, 8192, 16384, 32768]
CONTEXT_Q_LENS = [4096, 8192, 16384, 32768]
CONTEXT_LENS = [1024, 2048, 4096, 8192, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Plot diagonal pure-prefill tp{1,2,4,8}.csv results.",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Plot cached-context MHA vs MQA results.",
    )
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[int, int]:
    match = re.fullmatch(r"q(\d+)(?:s(\d+))?", spec)
    if match is None:
        raise ValueError(f"Unsupported batch spec: {spec}")
    q_len = int(match.group(1))
    return q_len, int(match.group(2) or q_len)


def read_results(*paths: Path) -> dict[tuple[int, int], dict[str, float]]:
    data: dict[tuple[int, int], dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                route = "mha" if row["backend"].endswith("masked_mha") else "mqa"
                data.setdefault(parse_spec(row["batch_spec"]), {})[route] = float(
                    row["mean_time"]
                )
    return data


def make_grid(data: dict[tuple[int, int], dict[str, float]]) -> np.ndarray:
    grid = np.full((len(SEQ_LENS), len(Q_LENS)), np.nan)
    for row, seq_len in enumerate(SEQ_LENS):
        for column, q_len in enumerate(Q_LENS):
            if q_len > seq_len:
                continue
            routes = data.get((q_len, seq_len), {})
            if set(routes) != {"mha", "mqa"} or not all(
                math.isfinite(value) for value in routes.values()
            ):
                raise ValueError(f"Missing result for q={q_len}, seq={seq_len}")
            grid[row, column] = routes["mqa"] / routes["mha"]
    return grid


def estimate_last_crossover(points: list[tuple[int, float]]) -> float | None:
    crossing = None
    for (left_q, left_ratio), (right_q, right_ratio) in zip(points, points[1:]):
        if left_ratio >= 1 and right_ratio < 1:
            fraction = (1 - left_ratio) / (right_ratio - left_ratio)
            crossing = left_q + fraction * (right_q - left_q)
    return crossing


def plot_curve(args: argparse.Namespace) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for tp in (1, 2, 4, 8):
        data = read_results(
            args.input_dir / f"tp{tp}.csv",
            args.input_dir / f"tp{tp}_repeat.csv",
            args.input_dir / f"tp{tp}_crossover_repeat.csv",
            args.input_dir / f"tp{tp}_extended.csv",
        )
        points = sorted(
            (q_len, routes["mqa"] / routes["mha"])
            for (q_len, seq_len), routes in data.items()
            if q_len == seq_len
            and set(routes) == {"mha", "mqa"}
            and all(math.isfinite(value) for value in routes.values())
        )
        if not points:
            raise ValueError(f"No paired pure-prefill results for TP{tp}")
        q_lens, ratios = zip(*points)
        label = f"FlashMLA TP{tp} ({64 // tp} local heads)"
        (line,) = axis.plot(q_lens, ratios, marker="o", markersize=4, label=label)
        crossover = estimate_last_crossover(points)
        if crossover is None:
            print(f"TP{tp}: no crossover through {q_lens[-1]} tokens")
            continue
        print(f"TP{tp}: MHA -> MQA crossover at {crossover:.0f} tokens")
        axis.scatter([crossover], [1], color=line.get_color(), zorder=5)
        axis.annotate(
            f"{crossover / 1024:.1f}K",
            (crossover, 1),
            xytext=(4, 8),
            textcoords="offset points",
            color=line.get_color(),
        )

    flashinfer_data = read_results(
        args.input_dir / "tp8_flashinfer.csv",
        args.input_dir / "tp8_flashinfer_extended.csv",
    )
    flashinfer_points = sorted(
        (q_len, routes["mqa"] / routes["mha"])
        for (q_len, seq_len), routes in flashinfer_data.items()
        if q_len == seq_len
        and set(routes) == {"mha", "mqa"}
        and all(math.isfinite(value) for value in routes.values())
    )
    if flashinfer_points:
        q_lens, ratios = zip(*flashinfer_points)
        axis.plot(
            q_lens,
            ratios,
            marker="s",
            markersize=4,
            linestyle="--",
            label="FlashInfer TP8 (8 local heads)",
        )
        crossover = estimate_last_crossover(flashinfer_points)
        if crossover is None:
            print(f"FlashInfer TP8: no crossover through {q_lens[-1]} tokens")
        else:
            print(f"FlashInfer TP8: MHA -> MQA crossover at {crossover:.0f} tokens")

    axis.axhline(1, color="black", linewidth=1.2, linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Pure-prefill sequence length")
    axis.set_ylabel("Speedup = sparse MQA latency / masked MHA latency")
    axis.set_title("GLM-5 sparse MLA routing, QK/V head dim 256/256 (B200)")
    axis.grid(alpha=0.25)
    axis.legend()
    output = args.output or args.input_dir / "speedup_curve_glm5_hd256.png"
    figure.savefig(output, dpi=180)


def plot_context(args: argparse.Namespace) -> None:
    panels = []
    for tp in (1, 2, 4, 8):
        data = read_results(
            args.input_dir / f"tp{tp}_context_probe.csv",
            args.input_dir / f"tp{tp}_context.csv",
            args.input_dir / f"tp{tp}_context_extended.csv",
        )
        grid = np.full((len(CONTEXT_LENS), len(CONTEXT_Q_LENS)), np.nan)
        for column, q_len in enumerate(CONTEXT_Q_LENS):
            for row, context_len in enumerate(CONTEXT_LENS):
                routes = data.get((q_len, q_len + context_len), {})
                if set(routes) == {"mha", "mqa"}:
                    grid[row, column] = routes["mqa"] / routes["mha"]
        panels.append((f"TP{tp} ({64 // tp} local heads)", grid))

    finite = np.concatenate([grid[np.isfinite(grid)] for _, grid in panels])
    norm = TwoSlopeNorm(
        vmin=max(0.2, float(np.percentile(finite, 2))),
        vcenter=1,
        vmax=min(4, float(np.percentile(finite, 98))),
    )
    figure, axes = plt.subplots(
        1, 4, figsize=(18, 5.8), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    x = np.arange(len(CONTEXT_Q_LENS))
    y = np.arange(len(CONTEXT_LENS))
    for axis, (title, grid) in zip(axes, panels):
        masked_grid = np.ma.masked_invalid(grid)
        image = axis.imshow(
            masked_grid, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm
        )
        finite_values = grid[np.isfinite(grid)]
        if finite_values.min() <= 1 <= finite_values.max():
            axis.contour(x, y, masked_grid, levels=[1], colors="black", linewidths=2)
        for row, column in np.ndindex(grid.shape):
            value = grid[row, column]
            if np.isfinite(value):
                color = "white" if value < 0.45 or value > 3.2 else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        axis.set_title(title)
        axis.set_xticks(x, [f"{q // 1024}K" for q in CONTEXT_Q_LENS])
        axis.set_yticks(y, [f"{context // 1024}K" for context in CONTEXT_LENS])
        axis.set_xlabel("Query length")
        axis.set_facecolor("#eee")
    axes[0].set_ylabel("Cached context length")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.93, pad=0.012)
    colorbar.set_label(
        "Speedup = sparse MQA latency / masked MHA latency\n"
        ">1 favors masked MHA; black contour = 1.0"
    )
    figure.suptitle(
        "GLM-5 sparse MLA chunked prefill, QK/V head dim 256/256 (B200)",
        fontsize=16,
    )
    output = args.output or args.input_dir / "speedup_context_heatmap_glm5_hd256.png"
    figure.savefig(output, dpi=180)


def main() -> None:
    args = parse_args()
    if args.context:
        plot_context(args)
        return
    if args.curve:
        plot_curve(args)
        return
    panels: list[tuple[str, np.ndarray]] = []
    for tp in (1, 2, 4, 8):
        results = read_results(
            args.input_dir / f"tp{tp}_dense_mask.csv",
            args.input_dir / f"tp{tp}_dense_mask_extended.csv",
        )
        panels.append(
            (
                f"TP{tp}: FlashMLA Sparse MQA\n({128 // tp} local heads)",
                make_grid(results),
            )
        )
    flashinfer = read_results(args.input_dir / "tp8_flashinfer_dense_mask.csv")
    panels.append(
        ("TP8: FlashInfer Sparse MQA\n(default, 16 local heads)", make_grid(flashinfer))
    )

    finite = np.concatenate([grid[np.isfinite(grid)] for _, grid in panels])
    norm = TwoSlopeNorm(
        vmin=max(0.15, float(np.percentile(finite, 2))),
        vcenter=1,
        vmax=min(7, float(np.percentile(finite, 98))),
    )
    figure, axes = plt.subplots(
        1, 5, figsize=(24, 5.8), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    x = np.arange(len(Q_LENS))
    y = np.arange(len(SEQ_LENS))
    for axis, (title, grid) in zip(axes, panels):
        masked_grid = np.ma.masked_invalid(grid)
        image = axis.imshow(
            masked_grid, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm
        )
        axis.contour(x, y, masked_grid, levels=[1], colors="black", linewidths=2.2)
        for row, column in np.ndindex(grid.shape):
            value = grid[row, column]
            if np.isfinite(value):
                color = "white" if value < 0.4 or value > 4.2 else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=color,
                )
        axis.set_title(title, fontsize=12)
        axis.set_xticks(
            x,
            [str(q) if q < 1024 else f"{q / 1024:g}K" for q in Q_LENS],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(y, [f"{seq // 1024}K" for seq in SEQ_LENS])
        axis.set_xlabel("Query length")
        axis.set_facecolor("#eee")
    axes[0].set_ylabel("Total sequence length")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.93, pad=0.012)
    colorbar.set_label(
        "Speedup = sparse MQA latency / masked MHA latency\n"
        ">1 favors masked MHA; black contour = 1.0"
    )
    figure.suptitle("Sparse MLA routing sweep on B200", fontsize=17)
    output = args.output or args.input_dir / "speedup_heatmap_tp1_tp2_tp4_tp8.png"
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
