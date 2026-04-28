#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark scaled_mm Helion kernel: fine-grained dispatch vs single-config.

For each (hidden_size, feature_size) shape group, benchmarks every num_tokens
shape against all available configs for that group. Then analyzes config
reduction: how many configs per group do we actually need?
"""

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import regex as re

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from vllm.triton_utils import triton

VLLM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(VLLM_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CONFIG_PATH = VLLM_ROOT / "vllm/kernels/helion/configs/scaled_mm/nvidia_b200.json"
OUTPUT_DIR = Path("/home/dev/scaled_mm_bench_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_key(key: str):
    m = re.fullmatch(r"hidden_size_(\d+)_feature_size_(\d+)_num_tokens_(\d+)", key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def make_inputs(num_tokens, hidden_size, feature_size, device="cuda"):
    in_dtype = torch.float8_e4m3fn
    scale = 1.0 / math.sqrt(hidden_size)
    a = (
        scale
        * (
            0.5
            + torch.rand(num_tokens, hidden_size, dtype=torch.float32, device=device)
        )
    ).to(in_dtype)
    b = (
        (
            scale
            * (
                0.5
                + torch.rand(
                    feature_size, hidden_size, dtype=torch.float32, device=device
                )
            )
        )
        .to(in_dtype)
        .t()
    )
    scale_a = 0.5 + torch.rand((num_tokens, 1), dtype=torch.float32, device=device)
    scale_b = 0.5 + torch.rand((feature_size, 1), dtype=torch.float32, device=device)
    bias = 0.5 * (torch.rand(feature_size, dtype=torch.bfloat16, device=device) - 0.5)
    return a, b, scale_a, scale_b, torch.bfloat16, bias


def benchmark_us(fn, warmup=5, rep=20):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return ms * 1000.0


# ── Benchmark ──────────────────────────────────────────────────────────


def run_benchmark():
    import helion
    from helion.runtime.config import Config

    print(f"Loading configs from {CONFIG_PATH}", flush=True)
    with open(CONFIG_PATH) as f:
        all_configs = json.load(f)
    print(f"Loaded {len(all_configs)} configs", flush=True)

    from vllm.kernels.helion.ops.scaled_mm import scaled_mm as wrapper

    raw_fn = wrapper.raw_kernel_func
    decorated = helion.kernel(
        static_shapes=False,
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=5e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )(raw_fn)

    groups: dict[tuple[int, int], dict[int, dict]] = {}
    for key, cfg in all_configs.items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        hs, fs, nt = parsed
        groups.setdefault((hs, fs), {})[nt] = cfg

    groups = {k: v for k, v in groups.items() if len(v) > 1}

    results = []
    total_groups = len(groups)
    total_compilations = 0
    start_time = time.time()

    for gi, ((hs, fs), nt_configs) in enumerate(sorted(groups.items())):
        all_nt_configs = sorted(nt_configs.keys())
        test_shapes = all_nt_configs

        n_benchmarks = len(test_shapes) * len(all_nt_configs)
        ns, nc = len(test_shapes), len(all_nt_configs)
        print(
            f"\n[{gi + 1}/{total_groups}] ({hs}, {fs}): "
            f"{ns} shapes x {nc} cfgs = {n_benchmarks}",
            flush=True,
        )

        count = 0
        for nt_shape in test_shapes:
            a, b, scale_a, scale_b, out_dtype, bias = make_inputs(nt_shape, hs, fs)
            args = (a, b, scale_a, scale_b, out_dtype, bias)
            bound = decorated.bind(args)

            for nt_config in all_nt_configs:
                cfg_dict = nt_configs[nt_config]
                cfg = Config.from_dict(cfg_dict)

                try:
                    compiled = bound.compile_config(cfg, allow_print=False)
                    total_compilations += 1
                    compiled(a, b, scale_a, scale_b, out_dtype, bias)
                    torch.accelerator.synchronize()

                    def _run(
                        c=compiled,
                        a_=a,
                        b_=b,
                        sa=scale_a,
                        sb=scale_b,
                        od=out_dtype,
                        bi=bias,
                    ):
                        return c(a_, b_, sa, sb, od, bi)

                    lat = benchmark_us(_run)
                except Exception as e:
                    lat = float("nan")
                    print(
                        f"  ERROR ({hs},{fs}, shape={nt_shape}, cfg={nt_config}): {e}",
                        flush=True,
                    )

                results.append(
                    {
                        "hidden_size": hs,
                        "feature_size": fs,
                        "num_tokens_shape": nt_shape,
                        "num_tokens_config": nt_config,
                        "latency_us": lat,
                    }
                )

                count += 1
                is_diag = nt_shape == nt_config
                if count % 20 == 0 or is_diag:
                    elapsed = time.time() - start_time
                    tag = " <-- auto" if is_diag else ""
                    if not math.isnan(lat):
                        print(
                            f"  [{count}/{n_benchmarks}]"
                            f" shape={nt_shape:>5},"
                            f" cfg={nt_config:>5}:"
                            f" {lat:8.1f} us{tag}"
                            f"  ({elapsed:.0f}s,"
                            f" {total_compilations} compiles)",
                            flush=True,
                        )

    csv_path = OUTPUT_DIR / "cross_config_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "hidden_size",
                "feature_size",
                "num_tokens_shape",
                "num_tokens_config",
                "latency_us",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    elapsed = time.time() - start_time
    n = len(results)
    print(
        f"\nDone! {n} measurements, {total_compilations} compilations in {elapsed:.0f}s"
    )
    print(f"Results saved to {csv_path}")
    return csv_path


# ── Analysis ───────────────────────────────────────────────────────────


def load_data(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hs = int(row["hidden_size"])
            fs = int(row["feature_size"])
            nt_shape = int(row["num_tokens_shape"])
            nt_config = int(row["num_tokens_config"])
            lat = float(row["latency_us"])
            if not math.isnan(lat):
                data[(hs, fs, nt_shape, nt_config)] = lat
    return data


def group_by_shape(data):
    groups = defaultdict(lambda: defaultdict(dict))
    for (hs, fs, nt_shape, nt_config), lat in data.items():
        groups[(hs, fs)][nt_shape][nt_config] = lat
    return groups


def analyze_fine_grained_vs_single(groups):
    results = {}
    for (hs, fs), shape_data in sorted(groups.items()):
        nt_list = sorted(shape_data.keys())

        autotuned = {}
        for nt in nt_list:
            if nt in shape_data[nt]:
                autotuned[nt] = shape_data[nt][nt]

        config_total_lat = {}
        for nt_config in nt_list:
            total = 0
            valid = True
            for nt_shape in nt_list:
                if nt_config in shape_data[nt_shape]:
                    total += shape_data[nt_shape][nt_config]
                else:
                    valid = False
                    break
            if valid:
                config_total_lat[nt_config] = total

        if not config_total_lat:
            continue

        best_single_nt = min(config_total_lat, key=config_total_lat.get)
        best_single_lats = {
            nt_shape: shape_data[nt_shape].get(best_single_nt, float("nan"))
            for nt_shape in nt_list
        }

        slowdowns = {}
        for nt in nt_list:
            if nt in autotuned and nt in best_single_lats:
                slowdowns[nt] = best_single_lats[nt] / autotuned[nt]

        autotuned_total = sum(autotuned.values())
        best_single_total = sum(
            best_single_lats[nt] for nt in nt_list if nt in best_single_lats
        )

        results[(hs, fs)] = {
            "autotuned": autotuned,
            "best_single_nt": best_single_nt,
            "best_single_lats": best_single_lats,
            "slowdowns": slowdowns,
            "autotuned_total": autotuned_total,
            "best_single_total": best_single_total,
            "overall_slowdown": best_single_total / autotuned_total
            if autotuned_total > 0
            else float("nan"),
        }
    return results


def find_best_k_configs(groups, k_values=None):
    if k_values is None:
        k_values = list(range(1, 15))

    all_results = {}
    for (hs, fs), shape_data in sorted(groups.items()):
        nt_list = sorted(shape_data.keys())
        config_list = sorted(
            set().union(*[set(shape_data[nt].keys()) for nt in nt_list])
        )

        lat_matrix = {}
        for nt_shape in nt_list:
            for nt_config in config_list:
                lat_matrix[(nt_shape, nt_config)] = shape_data[nt_shape].get(
                    nt_config, float("inf")
                )

        autotuned_total = sum(lat_matrix.get((nt, nt), float("inf")) for nt in nt_list)

        group_results = {}
        for k in k_values:
            if k >= len(config_list):
                best_total = sum(
                    min(
                        lat_matrix.get((nt_shape, nt_config), float("inf"))
                        for nt_config in config_list
                    )
                    for nt_shape in nt_list
                )
                group_results[k] = {
                    "total_latency": best_total,
                    "slowdown_vs_autotuned": best_total / autotuned_total
                    if autotuned_total > 0
                    else float("nan"),
                    "configs": config_list,
                }
                continue

            chosen = []
            remaining_configs = set(config_list)
            for _ in range(k):
                best_config = None
                best_cost = float("inf")
                for candidate in remaining_configs:
                    trial_set = chosen + [candidate]
                    cost = sum(
                        min(
                            lat_matrix.get((nt_shape, c), float("inf"))
                            for c in trial_set
                        )
                        for nt_shape in nt_list
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_config = candidate
                if best_config is not None:
                    chosen.append(best_config)
                    remaining_configs.discard(best_config)

            total = sum(
                min(lat_matrix.get((nt_shape, c), float("inf")) for c in chosen)
                for nt_shape in nt_list
            )
            group_results[k] = {
                "total_latency": total,
                "slowdown_vs_autotuned": total / autotuned_total
                if autotuned_total > 0
                else float("nan"),
                "configs": chosen,
            }

        all_results[(hs, fs)] = {
            "autotuned_total": autotuned_total,
            "k_results": group_results,
            "nt_list": nt_list,
            "num_configs": len(config_list),
        }
    return all_results


# ── Plotting ───────────────────────────────────────────────────────────


def plot_heatmaps(groups, output_path):
    sorted_groups = sorted(groups.items())
    n = len(sorted_groups)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, ((hs, fs), shape_data) in enumerate(sorted_groups):
        ax = axes[idx]
        nt_list = sorted(shape_data.keys())
        n_nt = len(nt_list)

        matrix = np.full((n_nt, n_nt), np.nan)
        for i, nt_shape in enumerate(nt_list):
            autotuned_lat = shape_data[nt_shape].get(nt_shape, None)
            if autotuned_lat is None or autotuned_lat == 0:
                continue
            for j, nt_config in enumerate(nt_list):
                lat = shape_data[nt_shape].get(nt_config, np.nan)
                matrix[i, j] = lat / autotuned_lat

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.9, vmax=3.0, aspect="auto")
        ax.set_xticks(range(n_nt))
        ax.set_xticklabels([str(nt) for nt in nt_list], rotation=90, fontsize=6)
        ax.set_yticks(range(n_nt))
        ax.set_yticklabels([str(nt) for nt in nt_list], fontsize=6)
        ax.set_xlabel("Config (num_tokens)", fontsize=8)
        ax.set_ylabel("Shape (num_tokens)", fontsize=8)
        ax.set_title(f"({hs}, {fs})", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Latency Ratio vs Autotuned Config (>1.0 = slower)", fontsize=14, y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps to {output_path}")


def plot_k_configs_per_group(k_results, output_path):
    sorted_groups = sorted(k_results.items())
    n = len(sorted_groups)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, ((hs, fs), info) in enumerate(sorted_groups):
        ax = axes[idx]
        k_res = info["k_results"]
        ks = sorted(k_res.keys())
        slowdowns = [k_res[k]["slowdown_vs_autotuned"] for k in ks]

        ax.plot(ks, slowdowns, "o-", linewidth=2, markersize=5, color="steelblue")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Autotuned")
        ax.set_xlabel("Number of configs (k)", fontsize=9)
        ax.set_ylabel("Slowdown vs autotuned", fontsize=9)
        ax.set_title(f"({hs}, {fs})\n{info['num_configs']} total configs", fontsize=9)
        ax.set_ylim(bottom=0.95)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Slowdown vs Number of Configs per Shape Group", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved k-config per group to {output_path}")


def plot_aggregate_k(k_results, output_path):
    k_values = sorted(next(iter(k_results.values()))["k_results"].keys())

    avg_slowdowns = []
    max_slowdowns = []
    p90_slowdowns = []
    for k in k_values:
        slowdowns = []
        weights = []
        for info in k_results.values():
            s = info["k_results"][k]["slowdown_vs_autotuned"]
            w = info["autotuned_total"]
            if not math.isnan(s) and not math.isinf(s):
                slowdowns.append(s)
                weights.append(w)
        if slowdowns:
            avg_slowdowns.append(
                sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            )
            max_slowdowns.append(max(slowdowns))
            sorted_s = sorted(slowdowns)
            p90_slowdowns.append(
                sorted_s[min(int(0.9 * len(sorted_s)), len(sorted_s) - 1)]
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        k_values[: len(avg_slowdowns)],
        avg_slowdowns,
        "o-",
        linewidth=2.5,
        markersize=7,
        label="Weighted average",
        color="steelblue",
    )
    ax.plot(
        k_values[: len(p90_slowdowns)],
        p90_slowdowns,
        "s--",
        linewidth=2,
        markersize=6,
        label="P90",
        color="orange",
    )
    ax.plot(
        k_values[: len(max_slowdowns)],
        max_slowdowns,
        "^:",
        linewidth=1.5,
        markersize=6,
        label="Max (worst group)",
        color="red",
    )
    ax.axhline(
        y=1.0,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="Autotuned baseline",
    )
    ax.set_xlabel(
        "Number of configs per (hidden_size, feature_size) group", fontsize=12
    )
    ax.set_ylabel("Slowdown vs per-shape autotuned config", fontsize=12)
    ax.set_title("Config Reduction: How Many Configs Do We Actually Need?", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.95)
    ax.set_xticks(k_values[: len(avg_slowdowns)])
    for k, avg in zip(k_values, avg_slowdowns):
        if k <= 8:
            ax.annotate(
                f"{avg:.2f}x",
                (k, avg),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                color="steelblue",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregate k-analysis to {output_path}")


def plot_per_shape_slowdown(comparison, output_path):
    sorted_groups = sorted(comparison.items())
    fig, axes = plt.subplots(
        len(sorted_groups), 1, figsize=(14, 3.5 * len(sorted_groups)), squeeze=False
    )

    for idx, ((hs, fs), info) in enumerate(sorted_groups):
        ax = axes[idx, 0]
        slowdowns = info["slowdowns"]
        nt_list = sorted(slowdowns.keys())
        vals = [slowdowns[nt] for nt in nt_list]
        colors = [
            "green" if v <= 1.05 else "orange" if v <= 1.2 else "red" for v in vals
        ]
        bars = ax.bar(
            range(len(nt_list)), vals, color=colors, edgecolor="black", linewidth=0.5
        )
        ax.set_xticks(range(len(nt_list)))
        ax.set_xticklabels([str(nt) for nt in nt_list], fontsize=8)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.7)
        ax.set_ylabel("Slowdown", fontsize=9)
        ax.set_title(
            f"({hs}, {fs}) — Best single: nt={info['best_single_nt']} — "
            f"Overall: {info['overall_slowdown']:.2f}x",
            fontsize=10,
        )
        ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, vals):
            if val > 1.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.1f}x",
                    ha="center",
                    fontsize=7,
                )
    fig.suptitle(
        "Per-Shape Slowdown: Best Single Config vs Autotuned", fontsize=14, y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-shape slowdown to {output_path}")


def plot_summary_dashboard(comparison, k_results, output_path):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Overall slowdown per shape group (1 config vs autotuned)
    ax1 = fig.add_subplot(gs[0, 0])
    groups_sorted = sorted(comparison.items())
    labels = [f"({hs},{fs})" for (hs, fs), _ in groups_sorted]
    overall_slowdowns = [info["overall_slowdown"] for _, info in groups_sorted]
    colors = [
        "green" if s <= 1.1 else "orange" if s <= 1.5 else "red"
        for s in overall_slowdowns
    ]
    ax1.barh(
        range(len(labels)),
        overall_slowdowns,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.axvline(x=1.0, color="green", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Slowdown vs autotuned", fontsize=10)
    ax1.set_title("Best Single Config Slowdown\nper Shape Group", fontsize=11)
    ax1.grid(True, alpha=0.2, axis="x")
    for i, s in enumerate(overall_slowdowns):
        ax1.text(s + 0.01, i, f"{s:.2f}x", va="center", fontsize=8)

    # Panel 2: Aggregate k-config curve
    ax2 = fig.add_subplot(gs[0, 1])
    k_values = sorted(next(iter(k_results.values()))["k_results"].keys())
    avg_slowdowns = []
    max_slowdowns = []
    for k in k_values:
        slowdowns = []
        weights = []
        for info in k_results.values():
            s = info["k_results"][k]["slowdown_vs_autotuned"]
            w = info["autotuned_total"]
            if not math.isnan(s) and not math.isinf(s):
                slowdowns.append(s)
                weights.append(w)
        if slowdowns:
            avg_slowdowns.append(
                sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            )
            max_slowdowns.append(max(slowdowns))
    ax2.plot(
        k_values[: len(avg_slowdowns)],
        avg_slowdowns,
        "o-",
        linewidth=2.5,
        markersize=7,
        label="Weighted avg",
        color="steelblue",
    )
    ax2.plot(
        k_values[: len(max_slowdowns)],
        max_slowdowns,
        "^:",
        linewidth=1.5,
        markersize=6,
        label="Worst group",
        color="red",
    )
    ax2.axhline(
        y=1.0, color="green", linestyle="--", alpha=0.7, label="Autotuned (1.0x)"
    )
    ax2.set_xlabel("Configs per shape group", fontsize=10)
    ax2.set_ylabel("Slowdown", fontsize=10)
    ax2.set_title("Config Reduction Curve", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values[: len(avg_slowdowns)])

    # Panel 3: Config savings table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    total_original = sum(info["num_configs"] for info in k_results.values())
    table_data = [
        [
            "k (configs/group)",
            "Total configs",
            "Wtd avg slowdown",
            "Max slowdown",
            "Config reduction",
        ]
    ]
    for i, k in enumerate(k_values[: len(avg_slowdowns)]):
        total_k = sum(min(k, info["num_configs"]) for info in k_results.values())
        table_data.append(
            [
                str(k),
                str(total_k),
                f"{avg_slowdowns[i]:.3f}x",
                f"{max_slowdowns[i]:.2f}x",
                f"{(1 - total_k / total_original) * 100:.0f}% fewer",
            ]
        )
    table = ax3.table(
        cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for i, row in enumerate(table_data[1:]):
        if float(row[2].replace("x", "")) < 1.05:
            for j in range(len(row)):
                table[i + 1, j].set_facecolor("#d4edda")
    ax3.set_title("Config Reduction Summary", fontsize=11, pad=20)

    fig.suptitle(
        "Helion scaled_mm: Fine-Grained Config Dispatch Analysis (NVIDIA B200)",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary dashboard to {output_path}")


def print_text_report(comparison, k_results):
    print("\n" + "=" * 80)
    print("SCALED_MM CONFIG DISPATCH ANALYSIS REPORT (NVIDIA B200)")
    print("=" * 80)

    print("\n## 1. Fine-Grained vs Best Single Config (per shape group)")
    hdr = f"{'Shape (hs, fs)':<25} {'Best Single nt':<16} {'Overall':<12} {'Max'}"
    print(hdr)
    print("-" * 80)

    total_auto = 0
    total_single = 0
    for (hs, fs), info in sorted(comparison.items()):
        max_sd = max(info["slowdowns"].values()) if info["slowdowns"] else float("nan")
        print(
            f"({hs:>5}, {fs:>5})          nt={info['best_single_nt']:<10} "
            f"{info['overall_slowdown']:<20.3f}x {max_sd:.2f}x"
        )
        total_auto += info["autotuned_total"]
        total_single += info["best_single_total"]

    overall = total_single / total_auto if total_auto > 0 else float("nan")
    print(f"\nGlobal weighted slowdown (best single config per group): {overall:.3f}x")

    print("\n## 2. Config Reduction Analysis")
    k_values = sorted(next(iter(k_results.values()))["k_results"].keys())
    total_original = sum(info["num_configs"] for info in k_results.values())
    print(f"Total configs currently: {total_original}")
    hdr = f"\n{'k':<5} {'Total':<10} {'Avg':<12} {'Max':<12} {'Reduction'}"
    print(hdr)
    print("-" * 60)

    for k in k_values:
        slowdowns, weights = [], []
        for info in k_results.values():
            s = info["k_results"][k]["slowdown_vs_autotuned"]
            w = info["autotuned_total"]
            if not math.isnan(s) and not math.isinf(s):
                slowdowns.append(s)
                weights.append(w)
        if slowdowns:
            avg = sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            mx = max(slowdowns)
            total_k = sum(min(k, info["num_configs"]) for info in k_results.values())
            reduction = (1 - total_k / total_original) * 100
            print(f"{k:<5} {total_k:<10} {avg:<15.3f}x {mx:<15.2f}x {reduction:.0f}%")

    print("\n## 3. Recommendation")
    for k in k_values:
        slowdowns, weights = [], []
        for info in k_results.values():
            s = info["k_results"][k]["slowdown_vs_autotuned"]
            w = info["autotuned_total"]
            if not math.isnan(s) and not math.isinf(s):
                slowdowns.append(s)
                weights.append(w)
        if slowdowns:
            avg = sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            if avg < 1.05:
                total_k = sum(
                    min(k, info["num_configs"]) for info in k_results.values()
                )
                reduction = (1 - total_k / total_original) * 100
                print(
                    f"With k={k} configs/group"
                    f" ({total_k} total,"
                    f" {reduction:.0f}% reduction),"
                )
                print(
                    f"weighted average slowdown is only {avg:.3f}x vs full autotuning."
                )
                break

    print("\n## 4. Selected Configs for Recommended k")
    recommended_k = None
    for k in k_values:
        slowdowns, weights = [], []
        for info in k_results.values():
            s = info["k_results"][k]["slowdown_vs_autotuned"]
            w = info["autotuned_total"]
            if not math.isnan(s) and not math.isinf(s):
                slowdowns.append(s)
                weights.append(w)
        if slowdowns:
            avg = sum(s * w for s, w in zip(slowdowns, weights)) / sum(weights)
            if avg < 1.05:
                recommended_k = k
                break
    if recommended_k is None:
        recommended_k = k_values[-1]

    for (hs, fs), info in sorted(k_results.items()):
        configs = info["k_results"][recommended_k]["configs"]
        sd = info["k_results"][recommended_k]["slowdown_vs_autotuned"]
        print(f"  ({hs:>5}, {fs:>5}): configs={configs}, slowdown={sd:.3f}x")


def run_analysis(csv_path):
    print(f"\nLoading data from {csv_path}")
    data = load_data(csv_path)
    print(f"Loaded {len(data)} measurements")

    groups = group_by_shape(data)
    print(f"Found {len(groups)} shape groups")

    comparison = analyze_fine_grained_vs_single(groups)
    k_results = find_best_k_configs(groups, k_values=list(range(1, 15)))

    print_text_report(comparison, k_results)

    plot_heatmaps(groups, OUTPUT_DIR / "heatmaps.png")
    plot_k_configs_per_group(k_results, OUTPUT_DIR / "k_config_per_group.png")
    plot_aggregate_k(k_results, OUTPUT_DIR / "aggregate_k_analysis.png")
    plot_per_shape_slowdown(comparison, OUTPUT_DIR / "per_shape_slowdown.png")
    plot_summary_dashboard(comparison, k_results, OUTPUT_DIR / "summary_dashboard.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    csv_path = OUTPUT_DIR / "cross_config_results.csv"
    if csv_path.exists() and "--rerun" not in sys.argv:
        print(f"Found existing results at {csv_path}, running analysis only.")
        print("Pass --rerun to re-run the benchmark.")
    else:
        csv_path = run_benchmark()
    run_analysis(csv_path)
