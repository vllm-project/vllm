#!/usr/bin/env python3
"""Analyze a vLLM job results folder (Chrome trace JSON + optional Slurm log)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Allow `python plotting_tools/analyze_job.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting_tools.classify import FABRIC_COMM_OPS, MOVEMENT_OPS  # noqa: E402
from plotting_tools.classify_report import (  # noqa: E402
    ClassificationReport,
    format_plotting_log,
    merge_reports,
    write_plotting_log,
)
from plotting_tools.plots import (  # noqa: E402
    PLOT_EXT,
    analyze_idle_gaps,
    build_gpu_traffic_matrix,
    build_traffic_matrix,
    collect_nccl_ops,
    comm_delta,
    comm_operation_breakdown,
    merge_comm_operation_breakdowns,
    merge_rank_traffic_matrices,
    nocomm_windows,
    plot_average_duty_pct,
    plot_classification_histogram,
    plot_classic_timeline,
    plot_collective_ops_breakdown_stats,
    plot_data_movement_breakdown,
    plot_fabric_comm_breakdown,
    plot_decomposed_timeline,
    plot_multi_node_decomposed,
    plot_duty_by_node,
    plot_expert_traffic_by_node,
    plot_expert_traffic_volume,
    plot_idle_context,
    plot_idle_transition_heatmap,
    plot_message_stats,
    plot_multi_window_cdf,
    plot_traffic_heatmap,
    plot_traffic_volume_pct,
    plot_window_cdf,
    gpu_idle_windows_ms,
    summarize_idle_transitions,
    write_collective_ops_table,
    write_comm_nvtx_table,
    write_summary_json,
)
from plotting_tools.trace_io import (  # noqa: E402
    clock_offset_ms,
    duty_by_sub,
    global_align_t0,
    sync_capture_t0,
    adjust_duty_with_pp_balance,
    infer_device_id,
    infer_global_rank,
    infer_local_rank,
    infer_node_name,
    pp_comm_reference_per_rank_us,
    pp_rank_order,
    pp_sendrecv_duration_us,
    load_chrome_trace,
    load_trace,
    parse_duration_events,
    parse_job_metadata,
    parse_nvtx_ranges,
    tag_phase,
    trace_t0_us,
)


def _find_traces(job_dir: Path) -> list[Path]:
    workers = _find_worker_traces(job_dir)
    if workers:
        return workers
    found: list[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        for p in sorted(job_dir.rglob(pattern)):
            if p.stat().st_size == 0:
                continue
            if "summary" in p.name:
                continue
            if "api_server" in p.name:
                continue
            found.append(p)
    return found


def _find_worker_traces(job_dir: Path) -> list[Path]:
    root = job_dir / "ray_worker_nsight"
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*.jsonl")):
        if p.stat().st_size == 0:
            continue
        out.append(p)
    return out


def _trace_output_prefix(trace_path: Path) -> str:
    """Node + worker pid so TP multi-GPU traces do not overwrite outputs."""
    import re

    node = infer_node_name(trace_path) or trace_path.stem.replace(".", "_")
    m = re.search(r"worker_process_(\d+)", trace_path.name)
    if m:
        return f"{node}_w{m.group(1)}"
    return node


def _build_rank_node_names(
    job_meta: dict[str, Any],
    n_ranks: int,
    tp: int,
    loaded: list[tuple[Path, str, list[dict[str, Any]], ClassificationReport]],
) -> dict[int, str]:
    names: dict[int, str] = {}
    for path, _, events, _, _ in loaded:
        gr = infer_global_rank(path, job_meta, events)
        names[gr] = infer_node_name(path) or path.parent.name
    order = pp_rank_order(job_meta)
    for r in range(n_ranks):
        if r not in names and order:
            stage = r // max(tp, 1)
            if stage < len(order):
                names[r] = order[stage]
    return names


def _load_trace_events(
    trace_path: Path,
    *,
    strict: bool,
    report: ClassificationReport,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trace_path.suffix.lower() == ".jsonl":
        return load_trace(trace_path, strict=strict, report=report)
    raw = load_chrome_trace(trace_path)
    report.jsonl_lines = len(raw)
    nvtx = parse_nvtx_ranges(raw)
    events = parse_duration_events(raw, strict=strict)
    for e in events:
        report.record_parsed_event(
            e["name"],
            e["cat"],
            e["kind"],
            e["sub"],
            args=e.get("args"),
        )
    if nvtx:
        tag_phase(events, nvtx, [])
    return events, []


def _time_axis_label(*, align_global: bool, trim_capture_skew: bool) -> str:
    if align_global and trim_capture_skew:
        return "synced capture time"
    if align_global:
        return "global wall time"
    return "local trace time"


def analyze_trace(
    trace_path: Path,
    out_dir: Path,
    *,
    prefix: str,
    job_meta: dict,
    n_ranks: int,
    num_experts: int,
    strict: bool,
    max_plot_ms: float | None,
    events: list[dict[str, Any]] | None = None,
    time_origin_us: int | None = None,
    align_global: bool = True,
    trim_capture_skew: bool = True,
    pp_comm_split: bool = False,
    global_t0_us: int | None = None,
    pp_reference_us_per_rank: float = 0.0,
    rank_node_names: dict[int, str] | None = None,
    classification_report: ClassificationReport | None = None,
    comm_nvtx_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if events is None:
        events, comm_nvtx_records = _load_trace_events(
            trace_path,
            strict=strict,
            report=classification_report or ClassificationReport(
                trace_path=str(trace_path)
            ),
        )
    comm_nvtx_records = comm_nvtx_records or []

    trace_out = out_dir / prefix
    trace_out.mkdir(parents=True, exist_ok=True)

    pp = job_meta.get("pipeline_parallel", 1)
    tp = job_meta.get("tensor_parallel", 1)
    ep = job_meta.get("expert_parallel", 1)
    parallel_label = f"TP={tp} PP={pp} EP={ep}"
    pp_rank = infer_local_rank(trace_path, job_meta)
    global_rank = infer_global_rank(trace_path, job_meta, events)
    device_id = infer_device_id(events)
    node = infer_node_name(trace_path) or prefix
    local_t0_us = trace_t0_us(events)
    axis = _time_axis_label(
        align_global=align_global, trim_capture_skew=trim_capture_skew
    )
    decomp_title = (
        f"{node} (global rank {global_rank}, PP {pp_rank}, GPU {device_id}): "
        f"decomposed compute / comm ({axis})"
    )
    if pp_comm_split:
        decomp_title += " [PP vs local comm]"

    duty = duty_by_sub(events)
    if pp > 1 and pp_reference_us_per_rank > 0:
        duty = adjust_duty_with_pp_balance(
            duty,
            events,
            pp_reference_us_per_rank=pp_reference_us_per_rank,
        )

    plot_decomposed_timeline(
        events,
        trace_out / f"decomposed_timeline{PLOT_EXT}",
        title=decomp_title,
        max_ms=max_plot_ms,
        time_origin_us=time_origin_us,
        pp_comm_split=pp_comm_split,
    )
    plot_traffic_volume_pct(
        events,
        trace_out / f"traffic_volume_pct{PLOT_EXT}",
        title=f"{node}: category time share",
        parallel_label=parallel_label,
        duty=duty,
    )
    plot_classic_timeline(
        events,
        trace_out / f"compute_comm_control_timeline{PLOT_EXT}",
        title=f"{node}: compute / comm / control ({axis})",
        time_origin_us=time_origin_us,
    )

    expert_gb = plot_expert_traffic_volume(
        events,
        trace_out / f"expert_traffic_gb{PLOT_EXT}",
        num_experts=num_experts,
    )

    n_ranks_use = max(n_ranks, pp * tp, 2)
    mat = build_traffic_matrix(
        events,
        n_ranks_use,
        local_rank=global_rank,
        tp=tp,
        pp=pp,
    )
    plot_traffic_heatmap(
        mat,
        trace_out / f"rank_traffic_heatmap{PLOT_EXT}",
        title=f"{node}: rank-to-rank comm (rank {global_rank}, {parallel_label})",
        xlabel="Destination rank",
        ylabel="Source rank",
        tp=tp,
        pp=pp,
        rank_node_names=rank_node_names,
    )
    plot_traffic_heatmap(
        mat,
        trace_out / f"all2all_traffic_heatmap{PLOT_EXT}",
        title=f"{node}: comm traffic matrix (rank x rank)",
        tp=tp,
        pp=pp,
        rank_node_names=rank_node_names,
    )
    gpu_mat = build_gpu_traffic_matrix(events, n_gpus=tp if tp > 1 else None)
    plot_traffic_heatmap(
        gpu_mat,
        trace_out / f"gpu_traffic_heatmap{PLOT_EXT}",
        title=f"{node}: GPU-to-GPU comm on node (TP={tp})",
        xlabel="Destination GPU",
        ylabel="Source GPU",
        tp=tp,
        pp=1,
        rank_node_names={i: node for i in range(gpu_mat.shape[0])},
    )

    msg_stats = plot_message_stats(
        events,
        trace_out / f"message_stats_prefill_decode{PLOT_EXT}",
        prefix=node,
        comm_nvtx_records=comm_nvtx_records,
    )
    if comm_nvtx_records:
        write_comm_nvtx_table(
            comm_nvtx_records, trace_out / "comm_nvtx_messages.json"
        )

    fabric_breakdown, fabric_unclassified = plot_fabric_comm_breakdown(
        events,
        trace_out / f"fabric_comm_breakdown{PLOT_EXT}",
        title=f"{node}: fabric communication (count & GPU time)",
    )
    movement_breakdown, movement_unclassified = plot_data_movement_breakdown(
        events,
        trace_out / f"data_movement_breakdown{PLOT_EXT}",
        title=f"{node}: data movement — device & host (not fabric)",
    )
    # Backward-compatible filename (fabric-only).
    plot_collective_ops_breakdown_stats(
        fabric_breakdown,
        trace_out / f"collective_ops_breakdown{PLOT_EXT}",
        title=f"{node}: fabric communication (count & GPU time)",
    )
    if classification_report is not None:
        classification_report.record_comm_breakdown(
            fabric_breakdown, fabric_unclassified
        )
    write_summary_json(
        trace_out / "fabric_comm_breakdown.json",
        {
            "scope": "fabric",
            "ops": fabric_breakdown,
            "unclassified_comm": dict(fabric_unclassified.most_common(50)),
        },
    )
    write_summary_json(
        trace_out / "data_movement_breakdown.json",
        {
            "scope": "movement",
            "ops": movement_breakdown,
            "unclassified_comm": dict(movement_unclassified.most_common(50)),
        },
    )
    write_summary_json(
        trace_out / "collective_ops_breakdown.json",
        {
            "scope": "fabric",
            "ops": fabric_breakdown,
            "unclassified_comm": dict(fabric_unclassified.most_common(50)),
        },
    )

    nccl_ops = collect_nccl_ops(events, comm_nvtx_records=comm_nvtx_records)
    write_collective_ops_table(nccl_ops, trace_out / "collective_ops.json")

    nocomm = nocomm_windows(events)
    if nocomm:
        plot_window_cdf(
            nocomm,
            trace_out / f"nocomm_windows_cdf{PLOT_EXT}",
            xlabel="No-comm window (ms)",
            title=f"{node}: CDF of gaps without communication",
        )
    deltas = comm_delta(events)
    if deltas:
        plot_window_cdf(
            deltas,
            trace_out / f"comm_start_delta_cdf{PLOT_EXT}",
            xlabel="Δt between comm starts (ms)",
            title=f"{node}: CDF of comm start deltas",
        )

    idle = gpu_idle_windows_ms(events)
    if idle:
        plot_window_cdf(
            idle,
            trace_out / f"gpu_idle_windows_cdf{PLOT_EXT}",
            xlabel="GPU idle window (ms)",
            title=f"{node}: CDF of GPU idle gaps",
        )

    idle_ctx = plot_idle_context(events, trace_out, prefix=node, min_gap_us=1000)
    if idle_ctx.get("gap_count", 0):
        print(
            f"  [{node}] idle context: {idle_ctx['gap_count']} gaps, "
            f"top transition: {idle_ctx.get('top_transitions_by_ms', [])[:1]}"
        )

    timing: dict[str, Any] = {"local_t0_us": local_t0_us}
    if global_t0_us is not None:
        timing["global_t0_us"] = global_t0_us
        timing["clock_offset_ms"] = clock_offset_ms(
            events, time_origin_us=global_t0_us
        )
    if time_origin_us is not None:
        timing["plot_t0_us"] = time_origin_us
        timing["plot_offset_ms"] = (
            (local_t0_us - time_origin_us) / 1000.0
            if local_t0_us is not None
            else None
        )
    summary = {
        "trace": str(trace_path),
        "node": node,
        "pp_rank": pp_rank,
        "global_rank": global_rank,
        "device_id": device_id,
        "align_global": align_global,
        "trim_capture_skew": trim_capture_skew,
        "pp_comm_split": pp_comm_split,
        "timing": timing,
        "event_count": len(events),
        "duty_by_subcategory": duty,
        "expert_traffic_gb_heuristic": expert_gb,
        "message_stats": msg_stats,
        "fabric_ops_breakdown": fabric_breakdown,
        "movement_ops_breakdown": movement_breakdown,
        "comm_ops_breakdown": fabric_breakdown,
        "comm_unclassified": dict(fabric_unclassified.most_common(30)),
        "nccl_op_count": len(nccl_ops),
        "idle_context": idle_ctx,
        "job_metadata": job_meta,
        "parallel_label": parallel_label,
        "rank_traffic_matrix": mat.tolist(),
    }
    write_summary_json(trace_out / "summary.json", summary)
    return summary, events


def build_summary_plots(
    node_data: dict[str, dict[str, Any]],
    summary_dir: Path,
    *,
    job_meta: dict,
    n_ranks: int,
    job_label: str,
    time_origin_us: int | None = None,
    align_global: bool = True,
    trim_capture_skew: bool = True,
    pp_comm_split: bool = False,
    max_plot_ms: float | None = None,
    global_t0_us: int | None = None,
) -> dict[str, Any]:
    """Cross-node summary plots under plots/summary_plots/."""
    if len(node_data) < 1:
        return {}

    summary_dir.mkdir(parents=True, exist_ok=True)
    pp = job_meta.get("pipeline_parallel", 1)
    tp = job_meta.get("tensor_parallel", 1)
    ep = job_meta.get("expert_parallel", 1)
    parallel_label = f"TP={tp} PP={pp} EP={ep}"
    n_ranks_use = max(n_ranks, pp * tp, 2)

    node_duty = {n: d["duty_by_subcategory"] for n, d in node_data.items()}
    plot_duty_by_node(
        node_duty,
        summary_dir / f"duty_by_node{PLOT_EXT}",
        title=f"{job_label}: category duty by node",
    )
    plot_average_duty_pct(
        node_duty,
        summary_dir / f"traffic_volume_pct_mean{PLOT_EXT}",
        title=f"{job_label}: mean category duty across nodes",
        parallel_label=parallel_label,
    )

    if align_global and time_origin_us is not None and len(node_data) > 1:
        node_events = {
            n: d["events"] for n, d in node_data.items() if d.get("events")
        }
        if node_events:
            axis = _time_axis_label(
                align_global=True, trim_capture_skew=trim_capture_skew
            )
            title = f"{job_label}: all nodes ({axis})"
            if pp_comm_split:
                title += " [PP vs local comm]"
            plot_multi_node_decomposed(
                node_events,
                summary_dir / f"decomposed_timeline_aligned{PLOT_EXT}",
                title=title,
                max_ms=max_plot_ms,
                time_origin_us=time_origin_us,
                pp_comm_split=pp_comm_split,
            )

    expert_gb = {n: d["expert_traffic_gb_heuristic"] for n, d in node_data.items()}
    plot_expert_traffic_by_node(
        expert_gb,
        summary_dir / f"expert_traffic_gb_by_node{PLOT_EXT}",
        title=f"{job_label}: expert routing heuristic by node",
    )

    fabric_breakdowns = [
        d["fabric_ops_breakdown"]
        for d in node_data.values()
        if d.get("fabric_ops_breakdown")
    ]
    movement_breakdowns = [
        d["movement_ops_breakdown"]
        for d in node_data.values()
        if d.get("movement_ops_breakdown")
    ]
    merged_fabric = merge_comm_operation_breakdowns(
        fabric_breakdowns, include_ops=FABRIC_COMM_OPS
    )
    merged_movement = merge_comm_operation_breakdowns(
        movement_breakdowns, include_ops=MOVEMENT_OPS
    )
    merged_uncl: Counter[str] = Counter()
    for d in node_data.values():
        merged_uncl.update(d.get("comm_unclassified") or {})
    plot_collective_ops_breakdown_stats(
        merged_fabric,
        summary_dir / f"fabric_comm_breakdown{PLOT_EXT}",
        title=f"{job_label}: fabric communication (all nodes combined)",
    )
    plot_collective_ops_breakdown_stats(
        merged_movement,
        summary_dir / f"data_movement_breakdown{PLOT_EXT}",
        title=f"{job_label}: data movement — device & host (all nodes)",
    )
    plot_collective_ops_breakdown_stats(
        merged_fabric,
        summary_dir / f"collective_ops_breakdown{PLOT_EXT}",
        title=f"{job_label}: fabric communication (all nodes combined)",
    )
    write_summary_json(
        summary_dir / "fabric_comm_breakdown.json",
        {
            "scope": "fabric",
            "ops": merged_fabric,
            "unclassified_comm": dict(merged_uncl.most_common(50)),
        },
    )
    write_summary_json(
        summary_dir / "data_movement_breakdown.json",
        {"scope": "movement", "ops": merged_movement},
    )
    write_summary_json(
        summary_dir / "collective_ops_breakdown.json",
        {
            "scope": "fabric",
            "ops": merged_fabric,
            "unclassified_comm": dict(merged_uncl.most_common(50)),
        },
    )

    matrices: list[np.ndarray] = []
    ranks_seen: list[int] = []
    for node, data in sorted(node_data.items()):
        mat = np.array(data.get("rank_traffic_matrix") or [], dtype=np.float64)
        if mat.size:
            matrices.append(mat)
            ranks_seen.append(data.get("global_rank", data.get("pp_rank", 0)))

    rank_summary: dict[str, Any] = {}
    if matrices:
        merged = merge_rank_traffic_matrices(matrices, tp=tp, pp=pp)
        rank_node_names: dict[int, str] = {}
        for _node, data in node_data.items():
            gr = data.get("global_rank")
            if gr is not None:
                rank_node_names[int(gr)] = str(data.get("node") or _node).split("_")[0]

        plot_traffic_heatmap(
            merged,
            summary_dir / f"rank_traffic_heatmap{PLOT_EXT}",
            title=f"{job_label}: rank-to-rank comm ({parallel_label}, all nodes)",
            xlabel="Destination PP/TP rank",
            ylabel="Source PP/TP rank",
            colorbar_label="Bytes or µs proxy (summed across nodes)",
            tp=tp,
            pp=pp,
            rank_node_names=rank_node_names,
        )
        rank_summary = {
            "n_ranks": n_ranks_use,
            "ranks_seen": ranks_seen,
            "nodes": sorted(node_data),
            "matrix_sum": float(merged.sum()),
        }

    cdf_series: dict[str, list[float]] = {}
    pooled: dict[str, list[float]] = {
        "comm_delta": [],
        "nocomm": [],
        "gpu_idle": [],
    }
    all_gaps: list[dict[str, Any]] = []
    for node, data in sorted(node_data.items()):
        events = data.get("events") or []
        if not events:
            continue
        deltas = comm_delta(events)
        if deltas:
            cdf_series[f"{node} comm Δ"] = deltas
            pooled["comm_delta"].extend(deltas)
        nocomm = nocomm_windows(events)
        if nocomm:
            cdf_series[f"{node} no-comm"] = nocomm
            pooled["nocomm"].extend(nocomm)
        idle = gpu_idle_windows_ms(events)
        if idle:
            cdf_series[f"{node} GPU idle"] = idle
            pooled["gpu_idle"].extend(idle)
        all_gaps.extend(analyze_idle_gaps(events, min_gap_us=1000))

    if pooled["comm_delta"]:
        plot_multi_window_cdf(
            {"pooled": pooled["comm_delta"], **{k: v for k, v in cdf_series.items() if "comm Δ" in k}},
            summary_dir / f"comm_start_delta_cdf{PLOT_EXT}",
            xlabel="Δt between comm starts (ms)",
            title=f"{job_label}: comm start delta CDF (per node + pooled)",
        )
    if pooled["nocomm"]:
        plot_multi_window_cdf(
            {"pooled": pooled["nocomm"], **{k: v for k, v in cdf_series.items() if "no-comm" in k}},
            summary_dir / f"nocomm_windows_cdf{PLOT_EXT}",
            xlabel="No-comm window (ms)",
            title=f"{job_label}: no-comm window CDF (per node + pooled)",
        )
    if pooled["gpu_idle"]:
        plot_multi_window_cdf(
            {"pooled": pooled["gpu_idle"], **{k: v for k, v in cdf_series.items() if "GPU idle" in k}},
            summary_dir / f"gpu_idle_windows_cdf{PLOT_EXT}",
            xlabel="GPU idle window (ms)",
            title=f"{job_label}: GPU idle gap CDF (per node + pooled)",
        )

    idle_summary: dict[str, Any] = {}
    if all_gaps:
        idle_summary = summarize_idle_transitions(all_gaps)
        plot_idle_transition_heatmap(
            all_gaps,
            summary_dir / f"idle_transition_heatmap{PLOT_EXT}",
            title=f"{job_label}: idle time (ms) by before/after activity (all nodes)",
        )
        write_summary_json(
            summary_dir / "idle_gaps.json",
            {
                "summary": idle_summary,
                "gaps_sample": idle_summary.get("longest_gaps", []),
                "nodes": sorted(node_data),
            },
        )

    offsets = {
        n: (d.get("timing") or {}).get("clock_offset_ms")
        for n, d in node_data.items()
    }
    out = {
        "nodes": sorted(node_data),
        "align_global": align_global,
        "trim_capture_skew": trim_capture_skew,
        "pp_comm_split": pp_comm_split,
        "global_t0_us": global_t0_us,
        "plot_t0_us": time_origin_us,
        "clock_offset_ms_by_node": offsets,
        "parallel_label": parallel_label,
        "rank_traffic": rank_summary,
        "merged_fabric_comm_ops": merged_fabric,
        "merged_data_movement_ops": merged_movement,
        "merged_comm_ops": merged_fabric,
        "idle_context": idle_summary,
        "benchmark": {
            k: job_meta.get(k)
            for k in (
                "successful_requests",
                "benchmark_duration_s",
                "mean_ttft_ms",
                "mean_tpot_ms",
            )
            if job_meta.get(k) is not None
        },
    }
    write_summary_json(summary_dir / "summary.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot vLLM Nsight Chrome traces")
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Job results directory (e.g. results/7692897)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        action="append",
        help="Trace file(s): .jsonl (Nsight GUI) or Chrome .json",
    )
    parser.add_argument(
        "--slurm-out",
        type=Path,
        default=None,
        help="Slurm .out file for TP/PP/EP and benchmark metadata",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <job-dir>/plots)",
    )
    parser.add_argument("--n-ranks", type=int, default=2, help="Ranks for heatmap")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Model expert count (Qwen3-30B-A3B default)",
    )
    parser.add_argument(
        "--max-plot-ms",
        type=float,
        default=None,
        help="Limit timeline x-axis (ms) for readability",
    )
    parser.add_argument(
        "--strict-classify",
        action="store_true",
        help="Exit on unclassified events (DeepSeek parser mode)",
    )
    parser.add_argument(
        "--local-timeline",
        action="store_true",
        help=(
            "Per-trace t=0 (disable default synced multi-node timelines)"
        ),
    )
    parser.add_argument(
        "--pp-comm-split",
        action="store_true",
        help=(
            "Split collective_comm into PP SendRecv vs local memcpy bands "
            "in decomposed timelines"
        ),
    )
    args = parser.parse_args()

    sync_timelines = not args.local_timeline

    job_dir = args.job_dir.resolve()
    out_dir = (args.out_dir or job_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary_plots"

    slurm = args.slurm_out
    if slurm is None:
        candidates = sorted(job_dir.glob("*.out"))
        slurm = candidates[0] if candidates else None
    job_meta = parse_job_metadata(slurm)

    traces = args.trace or []
    if not traces:
        traces = _find_traces(job_dir)

    if not traces:
        print(
            "No trace files found (.jsonl or .json under job-dir).\n"
            "Export from Nsight GUI (JSONL) or: bash plotting_tools/export_nsys.sh ...",
            file=sys.stderr,
        )
        sys.exit(1)

    loaded: list[
        tuple[
            Path,
            str,
            list[dict[str, Any]],
            list[dict[str, Any]],
            ClassificationReport,
        ]
    ] = []
    classification_reports: list[ClassificationReport] = []
    for tp in traces:
        if tp.stat().st_size == 0:
            print(f"Skipping empty trace: {tp}")
            continue
        path = tp.resolve()
        prefix = _trace_output_prefix(path)
        print(f"Loading {path}")
        rep = ClassificationReport(trace_path=str(path))
        events, comm_records = _load_trace_events(
            path, strict=args.strict_classify, report=rep
        )
        classification_reports.append(rep)
        loaded.append((path, prefix, events, comm_records, rep))

    global_t0_us: int | None = None
    plot_t0_us: int | None = None
    align_global = sync_timelines
    trim_capture_skew = sync_timelines and len(loaded) > 1
    if sync_timelines and loaded:
        all_ev = [ev for _, _, ev, _, _ in loaded]
        global_t0_us = global_align_t0(all_ev)
        plot_t0_us = global_t0_us
        if trim_capture_skew:
            plot_t0_us = sync_capture_t0(all_ev) or plot_t0_us
        if global_t0_us is not None:
            print(f"Synced timelines: global t0_us (earliest capture)={global_t0_us}")
            for _, prefix, ev, _, _ in loaded:
                off = clock_offset_ms(ev, time_origin_us=global_t0_us)
                if off is not None:
                    print(f"  {prefix} capture starts +{off:.1f} ms vs global t0")
        if plot_t0_us is not None and plot_t0_us != global_t0_us:
            print(
                f"  plot t0_us (all captures active)={plot_t0_us} "
                f"(trim {((plot_t0_us - global_t0_us) / 1000):.1f} ms preamble)"
            )

    per_node_summaries: dict[str, dict[str, Any]] = {}
    all_summaries: dict[str, Any] = {}

    tp = int(job_meta.get("tensor_parallel", 1))
    pp = int(job_meta.get("pipeline_parallel", 1))
    n_ranks_default = max(args.n_ranks, pp * tp)

    pp_reference_us_per_rank = 0.0
    if pp > 1 and loaded:
        stage_pp: dict[int, int] = {}
        for path, _, events, _, _ in loaded:
            pr = infer_local_rank(path, job_meta)
            stage_pp[pr] = stage_pp.get(pr, 0) + pp_sendrecv_duration_us(events)
        pp_reference_us_per_rank = pp_comm_reference_per_rank_us(stage_pp, tp=tp)

    rank_node_names = _build_rank_node_names(
        job_meta, n_ranks_default, tp, loaded
    )

    tp_meta = int(job_meta.get("tensor_parallel", 1))
    pp_meta = int(job_meta.get("pipeline_parallel", 1))
    ep_meta = int(job_meta.get("expert_parallel", 1))
    parallel_label = f"TP={tp_meta} PP={pp_meta} EP={ep_meta}"

    for path, prefix, events, comm_records, rep in loaded:
        print(f"Analyzing {path} -> {out_dir / prefix}")
        summary, events = analyze_trace(
            path,
            out_dir,
            prefix=prefix,
            job_meta=job_meta,
            n_ranks=n_ranks_default,
            num_experts=args.num_experts,
            strict=args.strict_classify,
            max_plot_ms=args.max_plot_ms,
            events=events,
            time_origin_us=plot_t0_us,
            align_global=align_global,
            trim_capture_skew=trim_capture_skew,
            pp_comm_split=args.pp_comm_split,
            global_t0_us=global_t0_us,
            pp_reference_us_per_rank=pp_reference_us_per_rank,
            rank_node_names=rank_node_names,
            classification_report=rep,
            comm_nvtx_records=comm_records,
        )
        summary["events"] = events  # kept in memory for summary plots only
        per_node_summaries[prefix] = summary
        all_summaries[prefix] = {k: v for k, v in summary.items() if k != "events"}

    if len(per_node_summaries) >= 1:
        print(f"Building summary plots -> {summary_dir}")
        summary_meta = build_summary_plots(
            per_node_summaries,
            summary_dir,
            job_meta=job_meta,
            n_ranks=n_ranks_default,
            job_label=job_dir.name,
            time_origin_us=plot_t0_us,
            align_global=align_global,
            trim_capture_skew=trim_capture_skew,
            pp_comm_split=args.pp_comm_split,
            max_plot_ms=args.max_plot_ms,
            global_t0_us=global_t0_us,
        )
        all_summaries["summary_plots"] = summary_meta

    write_summary_json(out_dir / "job_summary.json", all_summaries)

    merged_amb: dict[str, int] = {}
    merged_skip: dict[str, int] = {}
    for rep in classification_reports:
        for k, v in rep.ambiguous_control.items():
            merged_amb[k] = merged_amb.get(k, 0) + v
        for k, v in rep.skipped_runtime_names.items():
            merged_skip[k] = merged_skip.get(k, 0) + v
    plot_classification_histogram(
        merged_amb,
        out_dir / f"unclassified_ambiguous_control{PLOT_EXT}",
        title=f"{job_dir.name}: ambiguous control (unclassified runtime labels)",
    )
    plot_classification_histogram(
        merged_skip,
        out_dir / f"skipped_runtime_histogram{PLOT_EXT}",
        title=f"{job_dir.name}: skipped runtime API (not loaded into plots)",
    )

    merged_fabric_for_log: dict[str, dict[str, float | int]] | None = None
    merged_movement_for_log: dict[str, dict[str, float | int]] | None = None
    if len(per_node_summaries) >= 1:
        sp = all_summaries.get("summary_plots") or {}
        merged_fabric_for_log = sp.get("merged_fabric_comm_ops") or sp.get(
            "merged_comm_ops"
        )
        merged_movement_for_log = sp.get("merged_data_movement_ops")
    if merged_fabric_for_log is None and per_node_summaries:
        merged_fabric_for_log = merge_comm_operation_breakdowns(
            [d["fabric_ops_breakdown"] for d in per_node_summaries.values()],
            include_ops=FABRIC_COMM_OPS,
        )
    if merged_movement_for_log is None and per_node_summaries:
        merged_movement_for_log = merge_comm_operation_breakdowns(
            [d["movement_ops_breakdown"] for d in per_node_summaries.values()],
            include_ops=MOVEMENT_OPS,
        )

    job_merged_report = merge_reports(classification_reports)
    job_inventory = job_merged_report.inventory_dict()

    log_text = format_plotting_log(
        str(job_dir),
        job_meta,
        classification_reports,
        parallel_label=parallel_label,
        merged_fabric_comm=merged_fabric_for_log,
        merged_data_movement=merged_movement_for_log,
        job_inventory=job_inventory,
    )
    write_plotting_log(
        out_dir,
        log_text,
        classification_reports,
        job_meta,
        merged_fabric_comm=merged_fabric_for_log,
        merged_data_movement=merged_movement_for_log,
        job_inventory=job_inventory,
    )
    inv_path = out_dir / "classification_inventory.json"
    write_summary_json(
        inv_path,
        {
            "job_inventory": job_inventory,
            "per_trace": {
                Path(r.trace_path).name: r.inventory_dict()
                for r in classification_reports
            },
        },
    )
    print(
        f"Wrote plotting_log.txt, plotting_log.json, and {inv_path.name} under {out_dir}"
    )

    print(f"Wrote per-node plots under {out_dir}")
    if len(per_node_summaries) > 1:
        print(f"Wrote cross-node summary under {summary_dir}")


if __name__ == "__main__":
    main()
