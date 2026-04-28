#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate publication-quality figures for the scaled_mm config dispatch paper."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("/home/dev/scaled_mm_bench_results")
CSV_PATH = OUTPUT_DIR / "cross_config_results.csv"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def load_data(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            hs, fs = int(row["hidden_size"]), int(row["feature_size"])
            nts, ntc = int(row["num_tokens_shape"]), int(row["num_tokens_config"])
            lat = float(row["latency_us"])
            if not math.isnan(lat):
                data[(hs, fs, nts, ntc)] = lat
    return data


def group_by_shape(data):
    groups = defaultdict(lambda: defaultdict(dict))
    for (hs, fs, nts, ntc), lat in data.items():
        groups[(hs, fs)][nts][ntc] = lat
    return groups


def greedy_k_configs(lat_matrix, nt_list, config_list, k):
    chosen = []
    remaining = set(config_list)
    for _ in range(min(k, len(config_list))):
        best_c, best_cost = None, float("inf")
        for c in remaining:
            trial = chosen + [c]
            cost = sum(
                min(lat_matrix.get((nts, cc), float("inf")) for cc in trial)
                for nts in nt_list
            )
            if cost < best_cost:
                best_cost = cost
                best_c = c
        if best_c:
            chosen.append(best_c)
            remaining.discard(best_c)
    return chosen


# ── Figure 1: Representative heatmap (4 groups) ──


def fig1_heatmaps_selected(groups):
    selected = [(4096, 4096), (8192, 8192), (12288, 4096), (28672, 8192)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    for idx, (hs, fs) in enumerate(selected):
        ax = axes[idx]
        shape_data = groups[(hs, fs)]
        nt_list = sorted(shape_data.keys())
        n = len(nt_list)
        matrix = np.full((n, n), np.nan)

        for i, nts in enumerate(nt_list):
            auto = shape_data[nts].get(nts, None)
            if auto is None or auto == 0:
                continue
            for j, ntc in enumerate(nt_list):
                lat = shape_data[nts].get(ntc, np.nan)
                matrix[i, j] = lat / auto

        im = ax.imshow(
            matrix,
            cmap="RdYlGn_r",
            vmin=0.9,
            vmax=4.0,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(nt) for nt in nt_list], rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(nt) for nt in nt_list], fontsize=7)
        ax.set_xlabel("Config tuned for num_tokens", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Actual num_tokens", fontsize=9)
        ax.set_title(f"K={hs}, N={fs}", fontsize=10)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Latency ratio vs autotuned (>1 = slower)", fontsize=9)

    fig.suptitle(
        "Figure 1: Config Sensitivity Heatmaps (selected shape groups)",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(OUTPUT_DIR / "fig1_heatmaps.png")
    plt.close(fig)
    print("Saved fig1_heatmaps.png")


# ── Figure 2: Aggregate config reduction curve ──


def fig2_config_reduction(groups):
    k_values = list(range(1, 15))
    avg_slowdowns, max_slowdowns, p90_slowdowns = [], [], []

    for k in k_values:
        slowdowns, weights = [], []
        for (hs, fs), shape_data in groups.items():
            nt_list = sorted(shape_data.keys())
            config_list = sorted(
                set().union(*[set(shape_data[nt].keys()) for nt in nt_list])
            )
            lat_matrix = {}
            for nts in nt_list:
                for ntc in config_list:
                    lat_matrix[(nts, ntc)] = shape_data[nts].get(ntc, float("inf"))

            auto_total = sum(lat_matrix.get((nt, nt), float("inf")) for nt in nt_list)
            chosen = greedy_k_configs(lat_matrix, nt_list, config_list, k)
            total = sum(
                min(lat_matrix.get((nts, c), float("inf")) for c in chosen)
                for nts in nt_list
            )
            sd = total / auto_total if auto_total > 0 else float("nan")
            if not math.isnan(sd):
                slowdowns.append(sd)
                weights.append(auto_total)

        if slowdowns:
            avg_slowdowns.append(
                sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            )
            max_slowdowns.append(max(slowdowns))
            sorted_s = sorted(slowdowns)
            p90_slowdowns.append(
                sorted_s[min(int(0.9 * len(sorted_s)), len(sorted_s) - 1)]
            )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Left: main curve
    ax1.plot(
        k_values,
        avg_slowdowns,
        "o-",
        linewidth=2.5,
        markersize=7,
        label="Weighted average",
        color="#2171b5",
        zorder=3,
    )
    ax1.plot(
        k_values,
        p90_slowdowns,
        "s--",
        linewidth=2,
        markersize=6,
        label="P90 (across groups)",
        color="#fd8d3c",
        zorder=3,
    )
    ax1.plot(
        k_values,
        max_slowdowns,
        "^:",
        linewidth=1.5,
        markersize=6,
        label="Max (worst group)",
        color="#cb181d",
        zorder=3,
    )
    ax1.axhline(
        y=1.0,
        color="#238b45",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label="Autotuned baseline (1.0x)",
    )
    ax1.fill_between(k_values, avg_slowdowns, 1.0, alpha=0.08, color="#2171b5")
    ax1.set_xlabel("Number of configs per (K, N) group")
    ax1.set_ylabel("Slowdown vs per-shape autotuned config")
    ax1.set_title("(a) Slowdown vs config budget")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0.93, 1.5)
    ax1.set_xticks(k_values)
    for k, avg in zip(k_values[:6], avg_slowdowns[:6]):
        ax1.annotate(
            f"{avg:.2f}x",
            (k, avg),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#2171b5",
        )

    # Right: config count table
    ax2.axis("off")
    total_original = 224
    rows = []
    for i, k in enumerate(k_values[:8]):
        total_k = 16 * min(k, 14)
        reduction = (1 - total_k / total_original) * 100
        rows.append(
            [
                str(k),
                str(total_k),
                f"{avg_slowdowns[i]:.3f}x",
                f"{max_slowdowns[i]:.2f}x",
                f"{reduction:.0f}%",
            ]
        )

    table = ax2.table(
        cellText=rows,
        colLabels=[
            "k",
            "Total\nconfigs",
            "Avg\nslowdown",
            "Max\nslowdown",
            "Config\nreduction",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(5):
        table[0, j].set_facecolor("#d0d0d0")
        table[0, j].set_text_props(weight="bold")
    # Highlight recommended row (k=3)
    for j in range(5):
        table[3, j].set_facecolor("#d4edda")

    ax2.set_title("(b) Config reduction summary", fontsize=11, pad=15)

    fig.suptitle(
        "Figure 2: Config budget analysis — diminishing returns after k=3",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(OUTPUT_DIR / "fig2_config_reduction.png")
    plt.close(fig)
    print("Saved fig2_config_reduction.png")


# ── Figure 3: Per-group single-config slowdown summary ──


def fig3_single_config_summary(groups):
    group_results = []
    for (hs, fs), shape_data in sorted(groups.items()):
        nt_list = sorted(shape_data.keys())

        # Oracle baseline: best config for each shape
        oracle = {nt: min(shape_data[nt].values()) for nt in nt_list}
        oracle_total = sum(oracle.values())

        best_nt, best_total = None, float("inf")
        for ntc in nt_list:
            total = sum(shape_data[nts].get(ntc, float("inf")) for nts in nt_list)
            if total < best_total:
                best_total = total
                best_nt = ntc

        overall_sd = best_total / oracle_total if oracle_total > 0 else float("nan")

        per_shape_sd = []
        for nt in nt_list:
            o = oracle.get(nt, float("nan"))
            b = shape_data[nt].get(best_nt, float("nan"))
            if o > 0 and not math.isnan(b):
                per_shape_sd.append(b / o)
        max_sd = max(per_shape_sd) if per_shape_sd else float("nan")

        group_results.append(
            {
                "label": f"({hs},{fs})",
                "overall_sd": overall_sd,
                "max_sd": max_sd,
                "best_nt": best_nt,
            }
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])

    labels = [r["label"] for r in group_results]
    overall = [r["overall_sd"] for r in group_results]
    maxsd = [r["max_sd"] for r in group_results]
    x = np.arange(len(labels))
    w = 0.35

    ax1.bar(
        x - w / 2,
        overall,
        w,
        label="Aggregate slowdown",
        color="#4292c6",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + w / 2,
        maxsd,
        w,
        label="Worst-shape slowdown",
        color="#ef6548",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.axhline(y=1.0, color="#238b45", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Slowdown vs oracle")
    ax1.set_title("(a) Best single config per group: aggregate and worst-shape penalty")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.15, axis="y")
    ax1.set_ylim(0, 7)

    for bar, val in zip(bars2, maxsd):
        if val > 2:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.1,
                f"{val:.1f}x",
                ha="center",
                fontsize=7,
                color="#cb181d",
            )

    # Panel b: selected example — show per-num_tokens penalty for one group
    example = (12288, 4096)
    shape_data = groups[example]
    nt_list = sorted(shape_data.keys())
    oracle = {nt: min(shape_data[nt].values()) for nt in nt_list}
    best_nt = 8192
    per_shape = [
        shape_data[nt].get(best_nt, float("nan")) / oracle[nt]
        if oracle[nt] > 0
        else float("nan")
        for nt in nt_list
    ]

    colors = [
        "#238b45" if v <= 1.1 else "#fd8d3c" if v <= 2.0 else "#cb181d"
        for v in per_shape
    ]
    bars = ax2.bar(
        range(len(nt_list)), per_shape, color=colors, edgecolor="black", linewidth=0.5
    )
    ax2.axhline(y=1.0, color="#238b45", linestyle="--", alpha=0.7)
    ax2.set_xticks(range(len(nt_list)))
    ax2.set_xticklabels([str(nt) for nt in nt_list], fontsize=8)
    ax2.set_xlabel("num_tokens (M dimension)")
    ax2.set_ylabel("Slowdown vs oracle")
    ax2.set_title(
        f"(b) Example: K={example[0]}, N={example[1]} — "
        f"using nt={best_nt} config for all shapes"
    )
    ax2.grid(True, alpha=0.15, axis="y")
    for bar, val in zip(bars, per_shape):
        if val > 1.1:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.08,
                f"{val:.1f}x",
                ha="center",
                fontsize=8,
            )

    fig.suptitle(
        "Figure 3: Single-config penalty across shape groups", fontsize=13, y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_single_config.png")
    plt.close(fig)
    print("Saved fig3_single_config.png")


# ── Figure 4: Latency scaling with num_tokens ──


def fig4_latency_scaling(groups):
    selected = [(2048, 4096), (4096, 6144), (8192, 10240), (28672, 8192)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    colors_map = plt.cm.tab10

    for idx, (hs, fs) in enumerate(selected):
        ax = axes[idx]
        shape_data = groups[(hs, fs)]
        nt_list = sorted(shape_data.keys())

        autotuned_lats = [shape_data[nt].get(nt, float("nan")) for nt in nt_list]
        ax.plot(
            nt_list,
            autotuned_lats,
            "o-",
            linewidth=2,
            markersize=5,
            color="#2171b5",
            label="Autotuned",
            zorder=3,
        )

        # Show a few fixed-config lines
        for ci, fixed_nt in enumerate([1, 64, 1024, 8192]):
            if fixed_nt not in shape_data[nt_list[0]]:
                continue
            lats = [shape_data[nt].get(fixed_nt, float("nan")) for nt in nt_list]
            ax.plot(
                nt_list,
                lats,
                "--",
                linewidth=1.2,
                alpha=0.7,
                color=colors_map(ci + 2),
                label=f"Fixed nt={fixed_nt}",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("num_tokens (M)")
        if idx == 0:
            ax.set_ylabel("Latency (us)")
        ax.set_title(f"K={hs}, N={fs}", fontsize=10)
        ax.grid(True, alpha=0.2, which="both")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Figure 4: Latency scaling — autotuned dispatch vs fixed configs",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_latency_scaling.png")
    plt.close(fig)
    print("Saved fig4_latency_scaling.png")


def main():
    data = load_data(CSV_PATH)
    groups = group_by_shape(data)
    fig1_heatmaps_selected(groups)
    fig2_config_reduction(groups)
    fig3_single_config_summary(groups)
    fig4_latency_scaling(groups)
    print("All figures generated.")


if __name__ == "__main__":
    main()
