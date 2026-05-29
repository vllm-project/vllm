# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mtp_ep_load_balance_utils import (
    SCHEMA_VERSION,
    build_condition_metrics,
    build_speedup_rows,
    normalize_time_components,
    reorder_histograms_by_expert_order,
    sort_experts_desc,
    summarize_step_time_components,
)

PLOT_MODULE = None


@dataclass
class LoadedConditionData:
    batch_size: int
    draft_length: int
    selected_dataset_indices: np.ndarray
    prompt_lengths: np.ndarray
    condition_latency_ms: float
    step_histograms: np.ndarray
    step_total_tokens: np.ndarray
    step_total_ms: np.ndarray
    step_attention_ms: np.ndarray
    step_routing_ms: np.ndarray
    step_prepare_ms: np.ndarray
    step_finalize_ms: np.ndarray
    step_ffn_ms: np.ndarray
    captured_step_kinds: np.ndarray
    layers: np.ndarray
    avg_histograms: np.ndarray
    num_forward_steps_total: int
    num_captured_steps: int
    num_dropped_steps: int


def import_plot_module():
    global PLOT_MODULE
    if PLOT_MODULE is None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        PLOT_MODULE = plt
    return PLOT_MODULE


def ensure_analysis_dirs(output_dir: Path) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    speedup_dir = plots_dir / "speedup"
    time_dir = plots_dir / "step_time_breakdown"
    load_grids_dir = plots_dir / "expert_load" / "grids"
    load_overlays_dir = plots_dir / "expert_load" / "overlays"
    for path in (
        tables_dir,
        speedup_dir,
        time_dir,
        load_grids_dir,
        load_overlays_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "speedup": speedup_dir,
        "time": time_dir,
        "load_grids": load_grids_dir,
        "load_overlays": load_overlays_dir,
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scalar(npz: Any, key: str, cast: type) -> Any:
    value = npz[key]
    return cast(value.reshape(-1)[0])


def load_condition_data(path: Path) -> LoadedConditionData:
    with np.load(path, allow_pickle=False) as npz:
        schema_version = _scalar(npz, "schema_version", int)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version} for {path}."
            )
        return LoadedConditionData(
            batch_size=_scalar(npz, "batch_size", int),
            draft_length=_scalar(npz, "draft_length", int),
            selected_dataset_indices=np.asarray(npz["selected_dataset_indices"]),
            prompt_lengths=np.asarray(npz["prompt_lengths"]),
            condition_latency_ms=_scalar(npz, "condition_latency_ms", float),
            step_histograms=np.asarray(npz["step_histograms"]),
            step_total_tokens=np.asarray(npz["step_total_tokens"]),
            step_total_ms=np.asarray(npz["step_total_ms"]),
            step_attention_ms=np.asarray(npz["step_attention_ms"]),
            step_routing_ms=np.asarray(npz["step_routing_ms"]),
            step_prepare_ms=np.asarray(npz["step_prepare_ms"]),
            step_finalize_ms=np.asarray(npz["step_finalize_ms"]),
            step_ffn_ms=np.asarray(npz["step_ffn_ms"]),
            captured_step_kinds=np.asarray(npz["captured_step_kinds"]),
            layers=np.asarray(npz["layers"]),
            avg_histograms=np.asarray(npz["avg_histograms"]),
            num_forward_steps_total=_scalar(npz, "num_forward_steps_total", int),
            num_captured_steps=_scalar(npz, "num_captured_steps", int),
            num_dropped_steps=_scalar(npz, "num_dropped_steps", int),
        )


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "collect_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={manifest['schema_version']} in manifest."
        )
    return manifest


def load_all_conditions(output_dir: Path) -> tuple[dict[str, Any], dict[tuple[int, int], LoadedConditionData]]:
    manifest = load_manifest(output_dir)
    results: dict[tuple[int, int], LoadedConditionData] = {}
    for condition in manifest["conditions"]:
        path = output_dir / condition["raw_path"]
        data = load_condition_data(path)
        results[(data.batch_size, data.draft_length)] = data
    return manifest, results


def build_step_time_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> list[dict[str, float | int]]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    rows: list[dict[str, float | int]] = []
    baseline_totals: dict[int, float] = {}
    partial_rows: dict[tuple[int, int], dict[str, float | int]] = {}

    for batch_size in batch_sizes:
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            summary = summarize_step_time_components(
                data.step_total_ms,
                data.step_attention_ms,
                data.step_routing_ms,
                data.step_prepare_ms,
                data.step_finalize_ms,
                data.step_ffn_ms,
            )
            row: dict[str, float | int] = {
                "batch_size": batch_size,
                "draft_length": draft_length,
                "num_steps": int(data.step_histograms.shape[0]),
                **summary,
            }
            partial_rows[(batch_size, draft_length)] = row
            if draft_length == 0:
                baseline_totals[batch_size] = float(summary["avg_step_total_ms"])

    for batch_size in batch_sizes:
        baseline_total = baseline_totals[batch_size]
        for draft_length in draft_lengths:
            row = dict(partial_rows[(batch_size, draft_length)])
            row.update(normalize_time_components(row, baseline_total))
            rows.append(row)
    return rows


def build_load_distribution_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    layers = tuple(results[(batch_sizes[0], draft_lengths[0])].layers.tolist())

    load_metric_rows: list[dict[str, Any]] = []
    condition_sorted_rows: list[dict[str, Any]] = []
    baseline_sorted_rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        baseline_data = results[(batch_size, 0)]
        baseline_avg = baseline_data.avg_histograms
        _, baseline_order = sort_experts_desc(baseline_avg)
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            metrics_rows = build_condition_metrics(
                batch_size=batch_size,
                draft_length=draft_length,
                num_steps=data.num_captured_steps,
                layers=layers,
                avg_histograms=data.avg_histograms,
                baseline_histograms=baseline_avg,
            )
            load_metric_rows.extend(metrics_rows)

            condition_sorted_counts, condition_order = sort_experts_desc(
                data.avg_histograms
            )
            baseline_sorted_counts = reorder_histograms_by_expert_order(
                data.avg_histograms,
                baseline_order,
            )

            for layer_row, layer_idx in enumerate(layers):
                for expert_rank, expert_id in enumerate(condition_order[layer_row]):
                    condition_sorted_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "expert_rank": expert_rank,
                            "expert_id": int(expert_id),
                            "avg_routed_assignments_per_step": float(
                                condition_sorted_counts[layer_row, expert_rank]
                            ),
                        }
                    )
                for expert_rank, expert_id in enumerate(baseline_order[layer_row]):
                    baseline_sorted_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "expert_rank": expert_rank,
                            "expert_id": int(expert_id),
                            "avg_routed_assignments_per_step": float(
                                baseline_sorted_counts[layer_row, expert_rank]
                            ),
                        }
                    )
    return load_metric_rows, condition_sorted_rows, baseline_sorted_rows


def plot_speedup_vs_draft_length(
    plot_dir: Path,
    speedup_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size in batch_sizes:
        y = [
            next(
                row["speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for draft_length in draft_lengths
        ]
        ax.plot(draft_lengths, y, marker="o", linewidth=2, label=f"bs={batch_size}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("draft_length")
    ax.set_ylabel("speedup vs draft_length=0")
    ax.set_title("Speedup vs Draft Length")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "speedup_vs_draft_length.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_speedup_vs_batch_size(
    plot_dir: Path,
    speedup_rows: list[dict[str, float | int]],
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for draft_length in draft_lengths:
        y = [
            next(
                row["speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, y, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("speedup vs draft_length=0")
    ax.set_title("Speedup vs Batch Size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "speedup_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_step_time_breakdown(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    step_rows: list[dict[str, float | int]],
) -> Path:
    plt = import_plot_module()
    rows = [
        next(
            row
            for row in step_rows
            if row["batch_size"] == batch_size and row["draft_length"] == draft_length
        )
        for draft_length in draft_lengths
    ]
    x = np.arange(len(draft_lengths))
    attention = np.asarray(
        [row["normalized_attention_ms"] for row in rows], dtype=np.float64
    )
    routing = np.asarray(
        [row["normalized_routing_ms"] for row in rows], dtype=np.float64
    )
    all2all = np.asarray(
        [row["normalized_all2all_ms"] for row in rows], dtype=np.float64
    )
    ffn = np.asarray(
        [row["normalized_ffn_ms"] for row in rows], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, attention, label="Attention", color="#4e79a7")
    ax.bar(x, routing, bottom=attention, label="Routing", color="#f28e2b")
    ax.bar(
        x,
        all2all,
        bottom=attention + routing,
        label="ALL2ALL",
        color="#59a14f",
    )
    ax.bar(
        x,
        ffn,
        bottom=attention + routing + all2all,
        label="FFN",
        color="#e15759",
    )
    for idx, row in enumerate(rows):
        total_height = attention[idx] + routing[idx] + all2all[idx] + ffn[idx]
        ax.text(
            idx,
            total_height + 0.02,
            f"{float(row['ffn_share']) * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in draft_lengths])
    ax.set_xlabel("draft_length")
    ax.set_ylabel("normalized avg step time")
    ax.set_title(f"Step Time Breakdown (batch_size={batch_size})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = plot_dir / f"batch_size_{batch_size:03d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_condition_grid(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    load_metric_rows: list[dict[str, Any]],
    results: dict[tuple[int, int], LoadedConditionData],
) -> Path:
    plt = import_plot_module()
    sample = results[(batch_size, draft_lengths[0])]
    layers = tuple(sample.layers.tolist())
    fig, axes = plt.subplots(
        len(layers),
        len(draft_lengths),
        figsize=(4.2 * len(draft_lengths), 2.8 * len(layers)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, layer_idx in enumerate(layers):
        for col_idx, draft_length in enumerate(draft_lengths):
            ax = axes[row_idx][col_idx]
            data = results[(batch_size, draft_length)]
            sorted_counts, sorted_ids = sort_experts_desc(data.avg_histograms)
            counts = sorted_counts[row_idx]
            expert_ids = sorted_ids[row_idx]
            metrics_row = next(
                row
                for row in load_metric_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
                and row["layer"] == layer_idx
            )
            tick_positions = np.linspace(
                0,
                counts.size - 1,
                num=min(8, counts.size),
                dtype=int,
            )
            ax.bar(np.arange(counts.size), counts, color="#1f77b4", width=1.0)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(expert_ids[pos]) for pos in tick_positions])
            ax.set_title(
                f"layer={layer_idx}, draft={draft_length}\n"
                f"bal={metrics_row['balancedness']:.4f}, "
                f"gini={metrics_row['gini']:.4f}",
                fontsize=10,
            )
            if col_idx == 0:
                ax.set_ylabel("avg routed count")
            if row_idx == len(layers) - 1:
                ax.set_xlabel("expert id (condition-sorted)")
    path = plot_dir / f"batch_size_{batch_size:03d}_grid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_overlay(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    results: dict[tuple[int, int], LoadedConditionData],
) -> Path:
    plt = import_plot_module()
    baseline = results[(batch_size, 0)]
    layers = tuple(baseline.layers.tolist())
    _, baseline_order = sort_experts_desc(baseline.avg_histograms)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3.0 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    for row_idx, layer_idx in enumerate(layers):
        ax = axes[row_idx]
        tick_positions = np.linspace(
            0,
            baseline_order.shape[1] - 1,
            num=min(8, baseline_order.shape[1]),
            dtype=int,
        )
        tick_labels = [str(baseline_order[row_idx, pos]) for pos in tick_positions]
        for color, draft_length in zip(colors[: len(draft_lengths)], draft_lengths):
            data = results[(batch_size, draft_length)]
            counts = reorder_histograms_by_expert_order(
                data.avg_histograms,
                baseline_order,
            )[row_idx]
            ax.plot(
                np.arange(counts.size),
                counts,
                linewidth=1.8,
                color=color,
                label=f"d={draft_length}",
            )
        ax.set_title(f"layer={layer_idx}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("avg routed count")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("expert id (baseline-sorted)")
    axes[0].legend(ncol=len(draft_lengths), loc="upper right")
    path = plot_dir / f"batch_size_{batch_size:03d}_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_report(
    output_dir: Path,
    manifest: dict[str, Any],
    speedup_rows: list[dict[str, float | int]],
    step_rows: list[dict[str, float | int]],
    load_metric_rows: list[dict[str, Any]],
) -> str:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    lines = [
        "# Qwen3.6 MTP-EP 三子实验报告",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{manifest['model']}`",
        (
            f"- 数据集：`{manifest['dataset']}` / "
            f"`{manifest['dataset_config']}` / `{manifest['dataset_split']}`"
        ),
        f"- batch_size：`{', '.join(map(str, batch_sizes))}`",
        f"- draft_length：`{', '.join(map(str, draft_lengths))}`",
        f"- max_tokens：`{manifest['max_tokens']}`",
        f"- warmup_rounds：`{manifest.get('warmup_rounds', 0)}`",
        "",
        "## Speedup",
        "",
    ]

    for batch_size in batch_sizes:
        candidates = [
            row
            for row in speedup_rows
            if row["batch_size"] == batch_size and row["draft_length"] != 0
        ]
        best = max(candidates, key=lambda row: float(row["speedup"]))
        speedup_summaries = ", ".join(
            f"d={row['draft_length']}:{float(row['speedup']):.3f}x"
            for row in candidates
        )
        lines.append(
            f"- batch_size={batch_size}: {speedup_summaries}; "
            f"best=d={best['draft_length']} ({float(best['speedup']):.3f}x)"
        )

    lines.extend(["", "## 每次验证时间开销分解", ""])
    for batch_size in batch_sizes:
        rows = [
            row for row in step_rows if row["batch_size"] == batch_size
        ]
        lines.append(f"### batch_size={batch_size}")
        for row in rows:
            lines.append(
                f"- draft_length={row['draft_length']}: "
                f"avg_step={float(row['avg_step_total_ms']):.2f} ms, "
                f"attention={float(row['avg_attention_ms']):.2f} ms, "
                f"routing={float(row['avg_routing_ms']):.2f} ms, "
                f"all2all={float(row['avg_all2all_ms']):.2f} ms, "
                f"ffn={float(row['avg_ffn_ms']):.2f} ms "
                f"({float(row['ffn_share']) * 100:.1f}%)"
            )

    lines.extend(["", "## Expert Load 分布", ""])
    for batch_size in batch_sizes:
        lines.append(f"### batch_size={batch_size}")
        batch_rows = [row for row in load_metric_rows if row["batch_size"] == batch_size]
        layers = sorted({int(row["layer"]) for row in batch_rows})
        for layer in layers:
            candidates = [
                row
                for row in batch_rows
                if row["layer"] == layer and row["draft_length"] != 0
            ]
            worst = min(candidates, key=lambda row: float(row["balancedness_delta"]))
            lines.append(
                f"- layer {layer}: worst balancedness delta at "
                f"d={worst['draft_length']} "
                f"(Δbal={float(worst['balancedness_delta']):+.4f}, "
                f"Δg={float(worst['gini_delta']):+.4f})"
            )
    return "\n".join(lines) + "\n"


def analyze_experiment(
    input_dir: Path,
    *,
    skip_plots: bool = False,
    skip_report: bool = False,
) -> None:
    manifest, results = load_all_conditions(input_dir)
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    dirs = ensure_analysis_dirs(input_dir)

    latency_by_condition = {
        condition: data.condition_latency_ms for condition, data in results.items()
    }
    speedup_rows = build_speedup_rows(
        latency_by_condition,
        batch_sizes,
        draft_lengths,
    )
    step_rows = build_step_time_rows(manifest, results)
    (
        load_metric_rows,
        condition_sorted_rows,
        baseline_sorted_rows,
    ) = build_load_distribution_rows(manifest, results)

    save_csv(dirs["tables"] / "speedup_metrics.csv", speedup_rows)
    save_csv(dirs["tables"] / "step_time_breakdown.csv", step_rows)
    save_csv(dirs["tables"] / "load_balance_metrics.csv", load_metric_rows)
    save_csv(
        dirs["tables"] / "averaged_distributions_condition_sorted.csv",
        condition_sorted_rows,
    )
    save_csv(
        dirs["tables"] / "averaged_distributions_baseline_sorted.csv",
        baseline_sorted_rows,
    )

    if not skip_plots:
        plot_speedup_vs_draft_length(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_speedup_vs_batch_size(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        for batch_size in batch_sizes:
            plot_step_time_breakdown(
                dirs["time"],
                batch_size,
                draft_lengths,
                step_rows,
            )
            plot_condition_grid(
                dirs["load_grids"],
                batch_size,
                draft_lengths,
                load_metric_rows,
                results,
            )
            plot_overlay(
                dirs["load_overlays"],
                batch_size,
                draft_lengths,
                results,
            )

    if not skip_report:
        report = build_report(
            input_dir,
            manifest,
            speedup_rows,
            step_rows,
            load_metric_rows,
        )
        with (input_dir / "实验报告.md").open("w", encoding="utf-8") as fp:
            fp.write(report)
