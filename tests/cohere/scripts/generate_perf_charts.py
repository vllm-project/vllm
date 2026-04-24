#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate performance benchmark charts from gh-pages data.

Reads serving benchmark data from the gh-pages branch and produces
matplotlib PNG charts comparing throughput and latency across devices.

Usage:
    python generate_perf_charts.py --model c4-25a218t_fp8 --model c5-3a30t_fp8
    python generate_perf_charts.py --model c4-25a218t_fp8 --repo-root /path/to/repo
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import regex as re

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

GPU_FILES = {
    "H100": "data/summary.json",
    "B200": "data/summary_b200.json",
    "MI300X": "data/summary_mi300x.json",
}

DEVICE_COLORS = {"H100": "#1f77b4", "B200": "#ff7f0e", "MI300X": "#2ca02c"}
DEVICE_LABEL_COLORS = {"H100": "#0b3d91", "B200": "#b25600", "MI300X": "#1a6e1a"}
DEVICE_MARKERS = {"H100": "o", "B200": "s", "MI300X": "^"}

SHAPES = [
    (1000, 1000),
    (10000, 1000),
    (100000, 1000),
]
SHAPE_LABELS = {
    (1000, 1000): "1K / 1K",
    (10000, 1000): "10K / 1K",
    (100000, 1000): "100K / 1K",
}

MAX_RECENT_RUNS = 5


def load_gpu_data(repo_root: Path, gpu_file: str) -> list[dict]:
    """Load JSON data from a file on the gh-pages branch."""
    try:
        result = subprocess.run(
            ["git", "show", f"gh-pages:{gpu_file}"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: could not load {gpu_file}: {e}", file=sys.stderr)
        return []


def model_matches(test_name: str, model: str) -> bool:
    """Check if a test name belongs to the given model (excluding eagle variants)."""
    if "eagle" in test_name:
        return False
    pattern = rf"serving_{re.escape(model)}_tp\d+"
    return bool(re.search(pattern, test_name))


def extract_serving_rows(entry: dict) -> list[dict]:
    """Convert a pandas to_dict()-style serving dict into a list of row dicts."""
    data = entry.get("data", entry)
    serving = data.get("serving", {})
    if not serving or "Test name" not in serving:
        return []

    indices = list(serving["Test name"].keys())
    rows = []
    for idx in indices:
        row = {}
        for col, values in serving.items():
            row[col] = values.get(idx)
        rows.append(row)
    return rows


def collect_data(
    repo_root: Path, model: str
) -> tuple[
    dict[str, dict[tuple[int, int], dict[int, dict[str, list[float]]]]],
    int | None,
]:
    """Collect benchmark data for a model across all GPUs.

    Returns: (gpu_data_dict, tp_size)
    """
    result = {}
    tp_size = None

    for gpu_name, gpu_file in GPU_FILES.items():
        entries = load_gpu_data(repo_root, gpu_file)
        if not entries:
            continue

        gpu_data: dict[tuple[int, int], dict[int, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        for entry in reversed(entries):
            rows = extract_serving_rows(entry)

            for row in rows:
                test_name = row.get("Test name", "")
                if not model_matches(test_name, model):
                    continue

                out_len = row.get("Output Len")
                in_len = row.get("Input Len")
                if out_len is None or in_len is None:
                    continue

                shape = (int(in_len), int(out_len))
                if shape not in SHAPES:
                    continue

                mc = row.get("# of max concurrency.")
                if mc is None:
                    continue
                mc = int(mc)

                if tp_size is None and row.get("TP Size") is not None:
                    tp_size = int(row["TP Size"])

                output_tput = row.get("Output Tput (tok/s)")
                ttft = row.get("Mean TTFT (ms)")

                if output_tput is None or ttft is None:
                    continue
                if output_tput <= 0 or ttft <= 0:
                    continue

                bucket = gpu_data[shape][mc]
                if len(bucket["output_tput"]) < MAX_RECENT_RUNS:
                    bucket["output_tput"].append(float(output_tput))
                    bucket["ttft"].append(float(ttft))

        if gpu_data:
            result[gpu_name] = dict(gpu_data)

    return result, tp_size


def compute_medians(
    data: dict[str, dict[tuple[int, int], dict[int, dict[str, list[float]]]]],
) -> dict[str, dict[tuple[int, int], dict[int, dict[str, float]]]]:
    """Collapse lists of values into medians."""
    result: dict[str, dict[tuple[int, int], dict[int, dict[str, float]]]] = {}
    for gpu, shapes in data.items():
        result[gpu] = {}
        for shape, mcs in shapes.items():
            result[gpu][shape] = {}
            for mc, metrics in mcs.items():
                result[gpu][shape][mc] = {
                    k: statistics.median(v) for k, v in metrics.items()
                }
    return result


def plot_throughput_vs_batchsize(
    medians: dict[str, dict[tuple[int, int], dict[int, dict[str, float]]]],
    model: str,
    output_dir: Path,
) -> Path:
    """Generate the throughput vs batch size chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(f"{model} — Output Throughput vs Batch Size", fontsize=16, y=1.02)

    for ax_idx, shape in enumerate(SHAPES):
        ax = axes[ax_idx]
        ax.set_title(f"Input/Output: {SHAPE_LABELS[shape]}", fontsize=12)
        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_ylabel("Output Tput (tok/s)", fontsize=11)

        has_data = False
        texts = []
        for gpu in ["H100", "B200", "MI300X"]:
            if gpu not in medians:
                continue
            shape_data = medians[gpu].get(shape, {})
            if not shape_data:
                continue

            batch_sizes = sorted(shape_data.keys())
            tputs = [shape_data[bs]["output_tput"] for bs in batch_sizes]

            ax.plot(
                batch_sizes,
                tputs,
                color=DEVICE_COLORS[gpu],
                marker=DEVICE_MARKERS[gpu],
                label=gpu,
                linewidth=2,
                markersize=8,
            )
            for bs, tput in zip(batch_sizes, tputs):
                texts.append(
                    ax.text(
                        bs,
                        tput,
                        f"{tput:.0f}",
                        fontsize=9,
                        fontweight="bold",
                        color=DEVICE_LABEL_COLORS[gpu],
                    )
                )
            has_data = True

        if texts:
            adjust_text(texts, ax=ax)
        if has_data:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        all_bs = sorted({bs for gpu in medians for bs in medians[gpu].get(shape, {})})
        if all_bs:
            ax.set_xticks(all_bs)
            ax.set_xticklabels([str(b) for b in all_bs])

    fig.tight_layout()
    out_path = output_dir / f"{model}_throughput_vs_batchsize.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_throughput_vs_latency(
    medians: dict[str, dict[tuple[int, int], dict[int, dict[str, float]]]],
    model: str,
    output_dir: Path,
) -> Path:
    """Generate the throughput vs latency chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(f"{model} — Mean TTFT vs Output Throughput", fontsize=16, y=1.02)

    for ax_idx, shape in enumerate(SHAPES):
        ax = axes[ax_idx]
        ax.set_title(f"Input/Output: {SHAPE_LABELS[shape]}", fontsize=12)
        ax.set_xlabel("Mean TTFT (ms)", fontsize=11)
        ax.set_ylabel("Output Tput (tok/s)", fontsize=11)

        has_data = False
        texts = []
        for gpu in ["H100", "B200", "MI300X"]:
            if gpu not in medians:
                continue
            shape_data = medians[gpu].get(shape, {})
            if not shape_data:
                continue

            batch_sizes = sorted(shape_data.keys())
            tputs = [shape_data[bs]["output_tput"] for bs in batch_sizes]
            ttfts = [shape_data[bs]["ttft"] for bs in batch_sizes]

            ax.plot(
                ttfts,
                tputs,
                color=DEVICE_COLORS[gpu],
                marker=DEVICE_MARKERS[gpu],
                label=gpu,
                linewidth=2,
                markersize=8,
            )
            for bs, x, y in zip(batch_sizes, ttfts, tputs):
                texts.append(
                    ax.text(
                        x,
                        y,
                        f"bs={bs} ({y:.0f})",
                        fontsize=9,
                        fontweight="bold",
                        color=DEVICE_LABEL_COLORS[gpu],
                    )
                )
            has_data = True

        if texts:
            adjust_text(texts, ax=ax)
        if has_data:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"{model}_throughput_vs_latency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def update_markdown(
    models: list[str],
    tp_sizes: dict[str, int | None],
    docs_dir: Path,
    images_dir: Path,
):
    """Write or update the performance.md file."""
    md_path = docs_dir / "performance.md"
    lines = ["# Serving Performance Benchmarks\n"]
    lines.append(
        "Auto-generated charts comparing serving throughput and latency "
        "across H100, B200, and MI300X devices.\n"
    )
    lines.append(
        "Input/output shapes: 1K/1K, 10K/1K, 100K/1K (all with output length 1000). "
        "Values are medians of up to the last 5 CI benchmark runs.\n"
    )

    rel_images = os.path.relpath(images_dir, docs_dir)

    for model in models:
        tput_img = f"{model}_throughput_vs_batchsize.png"
        lat_img = f"{model}_throughput_vs_latency.png"

        tp = tp_sizes.get(model)
        tp_str = f" (TP={tp})" if tp is not None else ""
        lines.append(f"## {model}{tp_str}\n")
        lines.append("### Throughput vs Batch Size\n")
        lines.append(f"![{model} throughput vs batch size]({rel_images}/{tput_img})\n")
        lines.append("### TTFT vs Throughput\n")
        lines.append(f"![{model} ttft vs throughput]({rel_images}/{lat_img})\n")

    md_path.write_text("\n".join(lines))
    print(f"  Updated {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance benchmark charts from gh-pages data."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model name(s) to generate charts for (can be repeated).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the git repository root. Auto-detected if not specified.",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = args.repo_root.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent.parent

    docs_dir = repo_root / "docs" / "cohere" / "tests"
    images_dir = docs_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model_tp_sizes: dict[str, int | None] = {}
    models_with_data: list[str] = []

    for model in args.model:
        print(f"Processing {model}...")
        raw_data, tp_size = collect_data(repo_root, model)
        if not raw_data:
            print(f"  No data found for {model}, skipping.")
            continue
        model_tp_sizes[model] = tp_size
        models_with_data.append(model)
        if tp_size is not None:
            print(f"  TP size: {tp_size}")
        medians = compute_medians(raw_data)

        for gpu, shapes in medians.items():
            for shape, mcs in shapes.items():
                for mc, vals in mcs.items():
                    print(
                        f"  {gpu} {SHAPE_LABELS[shape]} bs={mc}: "
                        f"output_tput={vals['output_tput']:.1f} tok/s, "
                        f"ttft={vals['ttft']:.1f} ms"
                    )

        plot_throughput_vs_batchsize(medians, model, images_dir)
        plot_throughput_vs_latency(medians, model, images_dir)

    update_markdown(models_with_data, model_tp_sizes, docs_dir, images_dir)
    print("Done.")


if __name__ == "__main__":
    main()
