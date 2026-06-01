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
    TPOT_DEFINITION,
    build_rank_load_from_histograms,
    build_condition_metrics,
    build_speedup_rows,
    normalize_global_time_components,
    reorder_histograms_by_expert_order,
    sort_experts_desc,
    summarize_global_step_time_components,
)

PLOT_MODULE = None


@dataclass
class LoadedConditionData:
    batch_size: int
    draft_length: int
    data_parallel_size: int
    num_samples: int
    batch_size_scope: str
    mixed_step_policy: str
    tpot_definition: str
    selected_dataset_indices: np.ndarray
    prompt_lengths: np.ndarray
    output_lengths: np.ndarray
    condition_latency_ms: float
    decode_time_total_ms: float
    num_output_tokens_total: int
    num_generation_tokens_total: int
    num_output_tokens_excl_first_total: int
    tpot_ms: float
    decode_throughput_tok_s: float
    step_histograms: np.ndarray
    step_total_tokens: np.ndarray
    step_total_ms: np.ndarray
    step_attention_ms: np.ndarray
    step_routing_ms: np.ndarray
    step_prepare_ms: np.ndarray
    step_finalize_ms: np.ndarray
    step_ffn_ms: np.ndarray
    captured_step_kinds: np.ndarray
    global_step_indices: np.ndarray
    global_step_total_ms: np.ndarray
    global_step_ffn_ms: np.ndarray
    global_step_other_ms: np.ndarray
    global_step_kinds: np.ndarray
    expert_to_ep_rank: np.ndarray
    layers: np.ndarray
    avg_histograms: np.ndarray
    num_forward_steps_total: int
    num_captured_steps: int
    num_global_candidate_steps: int
    num_global_captured_steps: int
    num_dropped_steps: int
    num_prefill_dropped_steps: int
    num_mixed_dropped_steps: int
    num_global_prefill_dropped_steps: int
    num_global_mixed_dropped_steps: int
    num_global_non_target_dropped_steps: int


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
    expert_load_dir = plots_dir / "expert_load"
    rank_load_dir = plots_dir / "rank_load"
    rank_traces_dir = plots_dir / "rank_traces"
    for path in (
        tables_dir,
        speedup_dir,
        time_dir,
        expert_load_dir,
        rank_load_dir,
        rank_traces_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "speedup": speedup_dir,
        "time": time_dir,
        "expert_load": expert_load_dir,
        "rank_load": rank_load_dir,
        "rank_traces": rank_traces_dir,
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rank_trace_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rank_trace_rows(
    condition_name: str,
    trace_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not trace_payloads:
        return rows
    if not any(payload.get("trace_samples") for payload in trace_payloads):
        return rows

    origin_ms = min(
        sample["step_start_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )
    for payload in trace_payloads:
        dp_rank = int(payload["dp_rank"])
        for order_idx, sample in enumerate(payload.get("trace_samples", [])):
            phase_totals = sample["phase_totals_ms"]
            rows.append(
                {
                    "condition": condition_name,
                    "batch_size": int(payload["batch_size"]),
                    "draft_length": int(payload["draft_length"]),
                    "dp_rank": dp_rank,
                    "trace_order": order_idx,
                    "step_index": int(sample["step_index"]),
                    "step_kind": sample["step_kind"],
                    "total_scheduled_tokens": int(sample["total_scheduled_tokens"]),
                    "step_start_offset_ms": (
                        float(sample["step_start_time_ms"]) - origin_ms
                    ),
                    "step_end_offset_ms": (
                        float(sample["step_end_time_ms"]) - origin_ms
                    ),
                    "step_total_ms": float(sample["step_total_ms"]),
                    "attention_ms": float(phase_totals["attention"]),
                    "routing_ms": float(phase_totals["routing"]),
                    "prepare_ms": float(phase_totals["prepare"]),
                    "finalize_ms": float(phase_totals["finalize"]),
                    "ffn_ms": float(phase_totals["ffn"]),
                }
            )
    return rows


def plot_rank_trace_timeline(
    plot_dir: Path,
    condition_name: str,
    trace_payloads: list[dict[str, Any]],
) -> Path | None:
    samples_exist = any(payload.get("trace_samples") for payload in trace_payloads)
    if not samples_exist:
        return None

    plt = import_plot_module()
    label_colors = {
        "attention": "#4E79A7",
        "routing": "#F28E2B",
        "prepare": "#E15759",
        "ffn": "#59A14F",
        "finalize": "#76B7B2",
    }
    origin_ms = min(
        sample["step_start_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )
    max_end_ms = max(
        sample["step_end_time_ms"]
        for payload in trace_payloads
        for sample in payload.get("trace_samples", [])
    )

    fig, axes = plt.subplots(
        len(trace_payloads),
        1,
        figsize=(12, 2.8 * len(trace_payloads)),
        sharex=True,
    )
    if len(trace_payloads) == 1:
        axes = [axes]

    for ax, payload in zip(axes, trace_payloads):
        dp_rank = int(payload["dp_rank"])
        samples = sorted(
            payload.get("trace_samples", []),
            key=lambda item: float(item["step_start_time_ms"]),
        )
        for sample_idx, sample in enumerate(samples):
            y = len(samples) - sample_idx - 1
            start_ms = float(sample["step_start_time_ms"]) - origin_ms
            total_ms = float(sample["step_total_ms"])
            ax.broken_barh(
                [(start_ms, total_ms)],
                (y - 0.35, 0.7),
                facecolors="none",
                edgecolors="#444444",
                linewidth=0.8,
            )
            for event in sample["events"]:
                label = str(event["label"])
                event_start = start_ms + float(event["start_ms"])
                duration = float(event["duration_ms"])
                ax.broken_barh(
                    [(event_start, duration)],
                    (y - 0.35, 0.7),
                    facecolors=label_colors.get(label, "#999999"),
                    edgecolors="none",
                )
            ax.text(
                start_ms - 1.0,
                y,
                f"step={int(sample['step_index'])}",
                ha="right",
                va="center",
                fontsize=8,
            )
        ax.set_title(f"dp_rank={dp_rank}")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in label_colors.values()
    ]
    fig.legend(
        handles,
        list(label_colors.keys()),
        ncol=len(label_colors),
        loc="upper center",
    )
    axes[-1].set_xlabel("wall-clock offset within traced samples (ms)")
    axes[-1].set_xlim(0.0, max_end_ms - origin_ms)
    fig.suptitle(f"{condition_name} DP rank event timeline", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = plot_dir / f"{condition_name}_timeline.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _scalar(npz: Any, key: str, cast: type) -> Any:
    value = npz[key]
    return cast(value.reshape(-1)[0])


def load_condition_data(path: Path) -> LoadedConditionData:
    with np.load(path, allow_pickle=False) as npz:
        schema_version = _scalar(npz, "schema_version", int)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported raw schema_version="
                f"{schema_version} for {path}. Re-run `collect` with the current "
                "runtime before `analyze`."
            )
        return LoadedConditionData(
            batch_size=_scalar(npz, "batch_size", int),
            draft_length=_scalar(npz, "draft_length", int),
            data_parallel_size=_scalar(npz, "data_parallel_size", int),
            num_samples=_scalar(npz, "num_samples", int),
            batch_size_scope=_scalar(npz, "batch_size_scope", str),
            mixed_step_policy=_scalar(npz, "mixed_step_policy", str),
            tpot_definition=_scalar(npz, "tpot_definition", str),
            selected_dataset_indices=np.asarray(npz["selected_dataset_indices"]),
            prompt_lengths=np.asarray(npz["prompt_lengths"]),
            output_lengths=np.asarray(npz["output_lengths"]),
            condition_latency_ms=_scalar(npz, "condition_latency_ms", float),
            decode_time_total_ms=_scalar(npz, "decode_time_total_ms", float),
            num_output_tokens_total=_scalar(npz, "num_output_tokens_total", int),
            num_generation_tokens_total=_scalar(
                npz, "num_generation_tokens_total", int
            ),
            num_output_tokens_excl_first_total=_scalar(
                npz, "num_output_tokens_excl_first_total", int
            ),
            tpot_ms=_scalar(npz, "tpot_ms", float),
            decode_throughput_tok_s=_scalar(
                npz, "decode_throughput_tok_s", float
            ),
            step_histograms=np.asarray(npz["step_histograms"]),
            step_total_tokens=np.asarray(npz["step_total_tokens"]),
            step_total_ms=np.asarray(npz["step_total_ms"]),
            step_attention_ms=np.asarray(npz["step_attention_ms"]),
            step_routing_ms=np.asarray(npz["step_routing_ms"]),
            step_prepare_ms=np.asarray(npz["step_prepare_ms"]),
            step_finalize_ms=np.asarray(npz["step_finalize_ms"]),
            step_ffn_ms=np.asarray(npz["step_ffn_ms"]),
            captured_step_kinds=np.asarray(npz["captured_step_kinds"]),
            global_step_indices=np.asarray(npz["global_step_indices"]),
            global_step_total_ms=np.asarray(npz["global_step_total_ms"]),
            global_step_ffn_ms=np.asarray(npz["global_step_ffn_ms"]),
            global_step_other_ms=np.asarray(npz["global_step_other_ms"]),
            global_step_kinds=np.asarray(npz["global_step_kinds"]),
            expert_to_ep_rank=np.asarray(npz["expert_to_ep_rank"]),
            layers=np.asarray(npz["layers"]),
            avg_histograms=np.asarray(npz["avg_histograms"]),
            num_forward_steps_total=_scalar(npz, "num_forward_steps_total", int),
            num_captured_steps=_scalar(npz, "num_captured_steps", int),
            num_global_candidate_steps=_scalar(
                npz, "num_global_candidate_steps", int
            ),
            num_global_captured_steps=_scalar(
                npz, "num_global_captured_steps", int
            ),
            num_dropped_steps=_scalar(npz, "num_dropped_steps", int),
            num_prefill_dropped_steps=_scalar(npz, "num_prefill_dropped_steps", int),
            num_mixed_dropped_steps=_scalar(npz, "num_mixed_dropped_steps", int),
            num_global_prefill_dropped_steps=_scalar(
                npz, "num_global_prefill_dropped_steps", int
            ),
            num_global_mixed_dropped_steps=_scalar(
                npz, "num_global_mixed_dropped_steps", int
            ),
            num_global_non_target_dropped_steps=_scalar(
                npz, "num_global_non_target_dropped_steps", int
            ),
        )


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "collect_manifest.json"
    if not manifest_path.exists():
        return synthesize_manifest(output_dir)
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported manifest schema_version="
            f"{manifest['schema_version']}. Re-run `collect` with the current "
            "runtime before `analyze`."
        )
    return manifest


def synthesize_manifest(output_dir: Path) -> dict[str, Any]:
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata: dict[str, Any] = {}
    if run_metadata_path.exists():
        with run_metadata_path.open("r", encoding="utf-8") as fp:
            run_metadata = json.load(fp)

    raw_paths = sorted((output_dir / "raw").glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No collect_manifest.json or raw/*.npz found under {output_dir}."
        )

    conditions: list[dict[str, Any]] = []
    batch_sizes: set[int] = set()
    draft_lengths: set[int] = set()
    first_data: LoadedConditionData | None = None
    for raw_path in raw_paths:
        data = load_condition_data(raw_path)
        if first_data is None:
            first_data = data
        batch_sizes.add(data.batch_size)
        draft_lengths.add(data.draft_length)
        conditions.append(
            {
                "batch_size": data.batch_size,
                "draft_length": data.draft_length,
                "raw_path": str(Path("raw") / raw_path.name),
                "condition_latency_ms": data.condition_latency_ms,
                "decode_time_total_ms": data.decode_time_total_ms,
                "num_output_tokens_total": data.num_output_tokens_total,
                "num_generation_tokens_total": data.num_generation_tokens_total,
                "num_output_tokens_excl_first_total": (
                    data.num_output_tokens_excl_first_total
                ),
                "tpot_ms": data.tpot_ms,
                "decode_throughput_tok_s": data.decode_throughput_tok_s,
                "num_forward_steps_total": data.num_forward_steps_total,
                "num_captured_steps": data.num_captured_steps,
                "num_global_candidate_steps": data.num_global_candidate_steps,
                "num_global_captured_steps": data.num_global_captured_steps,
                "num_dropped_steps": data.num_dropped_steps,
                "num_prefill_dropped_steps": data.num_prefill_dropped_steps,
                "num_mixed_dropped_steps": data.num_mixed_dropped_steps,
                "num_global_prefill_dropped_steps": (
                    data.num_global_prefill_dropped_steps
                ),
                "num_global_mixed_dropped_steps": (
                    data.num_global_mixed_dropped_steps
                ),
                "num_global_non_target_dropped_steps": (
                    data.num_global_non_target_dropped_steps
                ),
            }
        )

    assert first_data is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "model": run_metadata.get("model", "unknown"),
        "dataset": run_metadata.get("dataset", "unknown"),
        "dataset_config": run_metadata.get("dataset_config"),
        "dataset_split": run_metadata.get("dataset_split", "unknown"),
        "batch_sizes": sorted(batch_sizes),
        "draft_lengths": sorted(draft_lengths),
        "data_parallel_size": int(
            run_metadata.get("data_parallel_size", first_data.data_parallel_size)
        ),
        "batch_size_scope": run_metadata.get(
            "batch_size_scope", first_data.batch_size_scope
        ),
        "num_samples": int(run_metadata.get("num_samples", first_data.num_samples)),
        "max_tokens": int(run_metadata.get("max_tokens", 0)),
        "layers": run_metadata.get("layers", first_data.layers.tolist()),
        "warmup_rounds": int(run_metadata.get("warmup_rounds", 0)),
        "trace_steps_per_rank": int(run_metadata.get("trace_steps_per_rank", 0)),
        "mixed_step_policy": run_metadata.get(
            "mixed_step_policy", first_data.mixed_step_policy
        ),
        "tpot_definition": run_metadata.get(
            "tpot_definition", first_data.tpot_definition
        ),
        "conditions": conditions,
    }


def load_all_conditions(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], LoadedConditionData]]:
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
            summary = summarize_global_step_time_components(
                data.global_step_total_ms,
                data.global_step_ffn_ms,
                data.global_step_other_ms,
            )
            row: dict[str, float | int] = {
                "batch_size": batch_size,
                "draft_length": draft_length,
                "decode_step_scope": (
                    "verification_only" if draft_length > 0 else "decode_only"
                ),
                "num_steps": int(data.global_step_indices.shape[0]),
                "num_forward_steps_total": data.num_forward_steps_total,
                "num_captured_steps": data.num_captured_steps,
                "num_global_candidate_steps": data.num_global_candidate_steps,
                "num_global_captured_steps": data.num_global_captured_steps,
                "num_dropped_steps": data.num_dropped_steps,
                "num_prefill_dropped_steps": data.num_prefill_dropped_steps,
                "num_mixed_dropped_steps": data.num_mixed_dropped_steps,
                "num_global_prefill_dropped_steps": (
                    data.num_global_prefill_dropped_steps
                ),
                "num_global_mixed_dropped_steps": (
                    data.num_global_mixed_dropped_steps
                ),
                "num_global_non_target_dropped_steps": (
                    data.num_global_non_target_dropped_steps
                ),
                "global_captured_step_ratio": (
                    data.num_global_captured_steps / data.num_global_candidate_steps
                    if data.num_global_candidate_steps > 0
                    else 0.0
                ),
                **summary,
            }
            partial_rows[(batch_size, draft_length)] = row
            if draft_length == 0:
                baseline_totals[batch_size] = float(summary["avg_step_total_ms"])

    for batch_size in batch_sizes:
        baseline_total = baseline_totals[batch_size]
        for draft_length in draft_lengths:
            row = dict(partial_rows[(batch_size, draft_length)])
            row.update(normalize_global_time_components(row, baseline_total))
            rows.append(row)
    return rows


def build_load_distribution_rows(
    manifest: dict[str, Any],
    results: dict[tuple[int, int], LoadedConditionData],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    batch_sizes = tuple(manifest["batch_sizes"])
    draft_lengths = tuple(manifest["draft_lengths"])
    layers = tuple(results[(batch_sizes[0], draft_lengths[0])].layers.tolist())

    load_metric_rows: list[dict[str, Any]] = []
    condition_sorted_rows: list[dict[str, Any]] = []
    baseline_sorted_rows: list[dict[str, Any]] = []
    rank_load_rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        baseline_data = results[(batch_size, 0)]
        baseline_avg = baseline_data.avg_histograms
        _, baseline_order = sort_experts_desc(baseline_avg)
        for draft_length in draft_lengths:
            data = results[(batch_size, draft_length)]
            metrics_rows = build_condition_metrics(
                batch_size=batch_size,
                draft_length=draft_length,
                num_steps=data.num_global_captured_steps,
                layers=layers,
                avg_histograms=data.avg_histograms,
                baseline_histograms=baseline_avg,
            )
            load_metric_rows.extend(metrics_rows)
            rank_load = build_rank_load_from_histograms(
                data.avg_histograms,
                data.expert_to_ep_rank,
                data.data_parallel_size,
            )

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
                for ep_rank in range(data.data_parallel_size):
                    rank_load_rows.append(
                        {
                            "batch_size": batch_size,
                            "draft_length": draft_length,
                            "layer": layer_idx,
                            "ep_rank": ep_rank,
                            "avg_routed_assignments_per_step": float(
                                rank_load[layer_row, ep_rank]
                            ),
                        }
                    )
    return (
        load_metric_rows,
        condition_sorted_rows,
        baseline_sorted_rows,
        rank_load_rows,
    )


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
                row["tpot_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for draft_length in draft_lengths
        ]
        ax.plot(draft_lengths, y, marker="o", linewidth=2, label=f"bs={batch_size}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("draft_length")
    ax.set_ylabel("TPOT speedup vs draft_length=0")
    ax.set_title("TPOT Speedup vs Draft Length")
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
                row["tpot_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, y, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("TPOT speedup vs draft_length=0")
    ax.set_title("TPOT Speedup vs Global Batch Size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "tpot_speedup_vs_batch_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_decode_throughput_speedup_vs_batch_size(
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
                row["decode_throughput_speedup"]
                for row in speedup_rows
                if row["batch_size"] == batch_size
                and row["draft_length"] == draft_length
            )
            for batch_size in batch_sizes
        ]
        ax.plot(batch_sizes, y, marker="o", linewidth=2, label=f"d={draft_length}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("decode throughput speedup vs draft_length=0")
    ax.set_title("Decode Throughput Speedup vs Global Batch Size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "decode_throughput_speedup_vs_batch_size.png"
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
    other = np.asarray(
        [row["normalized_other_ms"] for row in rows], dtype=np.float64
    )
    ffn = np.asarray(
        [row["normalized_ffn_ms"] for row in rows], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, other, label="Other", color="#4e79a7")
    ax.bar(
        x,
        ffn,
        bottom=other,
        label="FFN",
        color="#e15759",
    )
    for idx, row in enumerate(rows):
        total_height = other[idx] + ffn[idx]
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
    ax.set_ylabel("normalized avg decode/verification-only step time")
    ax.set_title(f"FFN/Other Step Time Breakdown (batch_size={batch_size})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = plot_dir / f"batch_size_{batch_size:03d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ffn_vs_draft_length(
    plot_dir: Path,
    batch_sizes: tuple[int, ...],
    draft_lengths: tuple[int, ...],
    step_rows: list[dict[str, float | int]],
) -> Path:
    plt = import_plot_module()
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size in batch_sizes:
        baseline_ffn_ms = float(
            next(
                row["avg_ffn_ms"]
                for row in step_rows
                if row["batch_size"] == batch_size and row["draft_length"] == 0
            )
        )
        y = [
            float(
                next(
                    row["avg_ffn_ms"]
                    for row in step_rows
                    if row["batch_size"] == batch_size
                    and row["draft_length"] == draft_length
                )
            )
            / baseline_ffn_ms
            for draft_length in draft_lengths
        ]
        ax.plot(draft_lengths, y, marker="o", linewidth=2, label=f"bs={batch_size}")
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("draft_length")
    ax.set_ylabel("avg_ffn_ms / avg_ffn_ms(d=0)")
    ax.set_title("FFN vs Draft Length")
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "ffn_vs_draft_length.png"
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


def plot_expert_load(
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
        ax.set_ylabel("avg routed assignments per global step")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("expert id (baseline-sorted)")
    axes[0].legend(ncol=len(draft_lengths), loc="upper right")
    path = plot_dir / f"batch_size_{batch_size:03d}_expert_load.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_rank_load(
    plot_dir: Path,
    batch_size: int,
    draft_lengths: tuple[int, ...],
    rank_load_rows: list[dict[str, Any]],
) -> Path:
    plt = import_plot_module()
    layers = sorted(
        {
            int(row["layer"])
            for row in rank_load_rows
            if row["batch_size"] == batch_size
        }
    )
    ep_ranks = sorted(
        {
            int(row["ep_rank"])
            for row in rank_load_rows
            if row["batch_size"] == batch_size
        }
    )
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, axes = plt.subplots(len(layers), 1, figsize=(9, 3.0 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    x = np.arange(len(ep_ranks))
    for row_idx, layer_idx in enumerate(layers):
        ax = axes[row_idx]
        width = 0.8 / max(len(draft_lengths), 1)
        for draft_idx, draft_length in enumerate(draft_lengths):
            values = [
                next(
                    float(row["avg_routed_assignments_per_step"])
                    for row in rank_load_rows
                    if row["batch_size"] == batch_size
                    and row["draft_length"] == draft_length
                    and row["layer"] == layer_idx
                    and row["ep_rank"] == ep_rank
                )
                for ep_rank in ep_ranks
            ]
            offset = (draft_idx - (len(draft_lengths) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[draft_idx % len(colors)],
                label=f"d={draft_length}",
            )
        ax.set_title(f"layer={layer_idx}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(rank) for rank in ep_ranks])
        ax.set_ylabel("avg routed assignments per global step")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("EP rank")
    axes[0].legend(ncol=len(draft_lengths), loc="upper right")
    path = plot_dir / f"batch_size_{batch_size:03d}_rank_load.png"
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
        "# Qwen3.6 MTP DP+EP 三子实验报告",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{manifest['model']}`",
        (
            f"- 数据集：`{manifest['dataset']}` / "
            f"`{manifest['dataset_config']}` / `{manifest['dataset_split']}`"
        ),
        f"- global batch_size：`{', '.join(map(str, batch_sizes))}`",
        f"- draft_length：`{', '.join(map(str, draft_lengths))}`",
        f"- data_parallel_size：`{manifest['data_parallel_size']}`",
        f"- num_samples：`{manifest['num_samples']}`",
        f"- batch_size_scope：`{manifest['batch_size_scope']}`",
        f"- mixed_step_policy：`{manifest['mixed_step_policy']}`",
        f"- TPOT 定义：`{manifest.get('tpot_definition', TPOT_DEFINITION)}`",
        f"- max_tokens：`{manifest['max_tokens']}`",
        f"- warmup_rounds：`{manifest.get('warmup_rounds', 0)}`",
        "",
        "## TPOT Speedup",
        "",
    ]

    for batch_size in batch_sizes:
        candidates = [
            row
            for row in speedup_rows
            if row["batch_size"] == batch_size and row["draft_length"] != 0
        ]
        best = max(candidates, key=lambda row: float(row["tpot_speedup"]))
        speedup_summaries = ", ".join(
            f"d={row['draft_length']}:{float(row['tpot_speedup']):.3f}x"
            for row in candidates
        )
        lines.append(
            f"- batch_size={batch_size}: {speedup_summaries}; "
            f"best=d={best['draft_length']} ({float(best['tpot_speedup']):.3f}x)"
        )

    lines.extend(["", "## Decode/Verification-only 时间开销分解", ""])
    for batch_size in batch_sizes:
        rows = [
            row for row in step_rows if row["batch_size"] == batch_size
        ]
        lines.append(f"### batch_size={batch_size}")
        for row in rows:
            lines.append(
                f"- draft_length={row['draft_length']}: "
                f"avg_step={float(row['avg_step_total_ms']):.2f} ms, "
                f"ffn={float(row['avg_ffn_ms']):.2f} ms, "
                f"other={float(row['avg_other_ms']):.2f} ms, "
                f"({float(row['ffn_share']) * 100:.1f}%), "
                f"global_captured={int(row['num_global_captured_steps'])}, "
                f"global_candidates={int(row['num_global_candidate_steps'])}, "
                f"prefill_drop={int(row['num_global_prefill_dropped_steps'])}, "
                f"mixed_drop={int(row['num_global_mixed_dropped_steps'])}"
            )

    lines.extend(["", "## Decode/Verification-only Expert Load 分布", ""])
    for batch_size in batch_sizes:
        lines.append(f"### batch_size={batch_size}")
        batch_rows = [
            row for row in load_metric_rows if row["batch_size"] == batch_size
        ]
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

    decode_time_ms_by_condition = {
        condition: data.decode_time_total_ms for condition, data in results.items()
    }
    generation_tokens_by_condition = {
        condition: data.num_generation_tokens_total
        for condition, data in results.items()
    }
    output_tokens_excl_first_by_condition = {
        condition: data.num_output_tokens_excl_first_total
        for condition, data in results.items()
    }
    speedup_rows = build_speedup_rows(
        decode_time_ms_by_condition,
        generation_tokens_by_condition,
        output_tokens_excl_first_by_condition,
        batch_sizes,
        draft_lengths,
    )
    step_rows = build_step_time_rows(manifest, results)
    (
        load_metric_rows,
        condition_sorted_rows,
        baseline_sorted_rows,
        rank_load_rows,
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
    save_csv(dirs["tables"] / "rank_load_metrics.csv", rank_load_rows)

    rank_trace_rows: list[dict[str, Any]] = []
    for condition in manifest["conditions"]:
        condition_name = Path(condition["raw_path"]).stem
        partial_dir = input_dir / "_dp_partials" / condition_name
        trace_paths = sorted(partial_dir.glob("rank_*.trace.json"))
        if not trace_paths:
            continue
        trace_payloads = [load_rank_trace_payload(path) for path in trace_paths]
        rank_trace_rows.extend(build_rank_trace_rows(condition_name, trace_payloads))
        if not skip_plots:
            plot_rank_trace_timeline(
                dirs["rank_traces"],
                condition_name,
                trace_payloads,
            )

    if rank_trace_rows:
        save_csv(dirs["tables"] / "rank_trace_summary.csv", rank_trace_rows)

    if not skip_plots:
        plot_speedup_vs_batch_size(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_decode_throughput_speedup_vs_batch_size(
            dirs["speedup"], speedup_rows, batch_sizes, draft_lengths
        )
        plot_ffn_vs_draft_length(
            dirs["time"],
            batch_sizes,
            draft_lengths,
            step_rows,
        )
        for batch_size in batch_sizes:
            plot_step_time_breakdown(
                dirs["time"],
                batch_size,
                draft_lengths,
                step_rows,
            )
            plot_expert_load(
                dirs["expert_load"],
                batch_size,
                draft_lengths,
                results,
            )
            plot_rank_load(
                dirs["rank_load"],
                batch_size,
                draft_lengths,
                rank_load_rows,
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
