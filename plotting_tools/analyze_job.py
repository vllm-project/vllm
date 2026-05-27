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
from plotting_tools.nccl_logs import (  # noqa: E402
    build_nccl_channel_matrix,
    build_nccl_log_rank_matrices,
    infer_nccl_world_size,
    nccl_rank_summary,
    parse_nccl_logs,
    summarize_nccl_log_ops,
)
from plotting_tools.plots import (  # noqa: E402
    PLOT_EXT,
    analyze_idle_gaps,
    build_gpu_traffic_matrix,
    build_traffic_matrix,
    collect_nccl_ops,
    comm_delta,
    gpu_idle_windows_ms,
    merge_comm_operation_breakdowns,
    merge_rank_traffic_matrices,
    nocomm_windows,
    plot_active_segments_timeline,
    plot_active_window_detection,
    plot_average_duty_pct,
    plot_batch_tokens_per_iteration,
    plot_classification_histogram,
    plot_classic_timeline,
    plot_collective_ops_breakdown_stats,
    plot_comm_timeline,
    plot_data_movement_breakdown,
    plot_decomposed_timeline,
    plot_duty_by_node,
    plot_expert_traffic_by_node,
    plot_expert_traffic_volume,
    plot_fabric_comm_breakdown,
    plot_fabric_comm_timeline,
    plot_fabric_comm_timeline_active,
    plot_fabric_comm_timeline_trimmed,
    plot_idle_context,
    plot_idle_transition_heatmap,
    plot_key_value_table,
    plot_message_size_cdf,
    plot_message_stats,
    plot_multi_node_decomposed,
    plot_multi_window_cdf,
    plot_nccl_log_ops_breakdown,
    plot_request_timeline,
    plot_time_breakdown_bar,
    plot_single_layer_breakdown,
    extract_single_layer_events,
    plot_traffic_heatmap,
    plot_traffic_volume_pct,
    plot_window_cdf,
    _map_events_to_bar_categories,
    summarize_idle_transitions,
    write_collective_ops_table,
    write_comm_nvtx_table,
    write_summary_json,
)
from plotting_tools.trace_io import (  # noqa: E402
    adjust_duty_with_pp_balance,
    clock_offset_ms,
    duty_by_sub,
    duty_by_sub_windows,
    global_align_t0,
    infer_active_segments_us,
    infer_device_id,
    infer_global_rank,
    infer_local_rank,
    infer_node_name,
    load_chrome_trace,
    load_trace,
    merge_intervals,
    parse_duration_events,
    parse_iteration_log,
    parse_job_metadata,
    parse_nvtx_ranges,
    pp_comm_reference_per_rank_us,
    pp_rank_order,
    pp_sendrecv_duration_us,
    sync_capture_t0,
    tag_phase,
    trim_events_to_window,
    trim_events_to_windows,
    trace_t0_us,
)


def _top_kernels_by_duration(
    events: list[dict[str, Any]], *, top_n: int = 50
) -> list[dict[str, Any]]:
    """Aggregate kernel durations by name, return top-N sorted by total time."""
    from collections import defaultdict

    dur_by_name: dict[str, float] = defaultdict(float)
    cnt_by_name: dict[str, int] = defaultdict(int)
    kind_by_name: dict[str, str] = {}
    sub_by_name: dict[str, str] = {}
    for e in events:
        name = e.get("name", "")
        dur_by_name[name] += e["dur"]
        cnt_by_name[name] += 1
        kind_by_name.setdefault(name, e.get("kind", ""))
        sub_by_name.setdefault(name, e.get("sub", ""))

    ranked = sorted(dur_by_name.items(), key=lambda x: -x[1])[:top_n]
    return [
        {
            "name": name,
            "total_dur_us": round(dur, 1),
            "total_dur_s": round(dur / 1e6, 4),
            "count": cnt_by_name[name],
            "avg_dur_us": round(dur / cnt_by_name[name], 1),
            "kind": kind_by_name[name],
            "sub": sub_by_name[name],
        }
        for name, dur in ranked
    ]


def _host_transfer_breakdown(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Break down host_transfer events by direction and size bucket."""
    ht_events = [e for e in events if e.get("sub") == "host_transfer"]
    if not ht_events:
        return {"total_count": 0}

    by_direction: dict[str, dict[str, float | int]] = {}
    size_buckets = {"<1KB": 0, "1KB-1MB": 0, "1MB-100MB": 0, ">100MB": 0}
    size_bucket_dur: dict[str, float] = {
        "<1KB": 0,
        "1KB-1MB": 0,
        "1MB-100MB": 0,
        ">100MB": 0,
    }

    for e in ht_events:
        name = e.get("name", "").lower()
        if "htod" in name or "h2d" in name:
            direction = "HtoD"
        elif "dtoh" in name or "d2h" in name:
            direction = "DtoH"
        elif "dtod" in name or "d2d" in name:
            direction = "DtoD"
        elif "cudamemcpyasync" in name or "cudamemcpy" in name:
            direction = "async_api_cpu_time"
        else:
            direction = "other"

        if direction not in by_direction:
            by_direction[direction] = {"count": 0, "dur_us": 0.0, "bytes": 0}
        by_direction[direction]["count"] += 1
        by_direction[direction]["dur_us"] += e["dur"]
        nbytes = e.get("bytes", 0) or 0
        by_direction[direction]["bytes"] += nbytes

        if nbytes < 1024:
            size_buckets["<1KB"] += 1
            size_bucket_dur["<1KB"] += e["dur"]
        elif nbytes < 1024 * 1024:
            size_buckets["1KB-1MB"] += 1
            size_bucket_dur["1KB-1MB"] += e["dur"]
        elif nbytes < 100 * 1024 * 1024:
            size_buckets["1MB-100MB"] += 1
            size_bucket_dur["1MB-100MB"] += e["dur"]
        else:
            size_buckets[">100MB"] += 1
            size_bucket_dur[">100MB"] += e["dur"]

    for d in by_direction.values():
        d["dur_s"] = round(d["dur_us"] / 1e6, 4)
        d["bytes_gb"] = round(d["bytes"] / (1024**3), 4)
    size_bucket_dur_s = {k: round(v / 1e6, 4) for k, v in size_bucket_dur.items()}

    return {
        "total_count": len(ht_events),
        "total_dur_s": round(sum(e["dur"] for e in ht_events) / 1e6, 4),
        "by_direction": by_direction,
        "by_size_bucket_count": size_buckets,
        "by_size_bucket_dur_s": size_bucket_dur_s,
    }


ACTIVE_DENSITY_THRESHOLD = 50
ACTIVE_BIN_SIZE_US = 1_000_000


def _active_window_envelope(
    segments: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if not segments:
        return None
    return (min(s for s, _ in segments), max(e for _, e in segments))


def _active_segments_summary(
    events: list[dict[str, Any]],
    segments: list[tuple[int, int]],
) -> dict[str, Any]:
    if not events:
        return {"segment_count": 0}
    capture_start = min(e["ts"] for e in events)
    capture_end = max(e["end"] for e in events)
    active_us = sum(end - start for start, end in segments)
    capture_us = max(capture_end - capture_start, 1)
    return {
        "segment_count": len(segments),
        "segments_us": [list(s) for s in segments],
        "segments_from_trace_start_ms": [
            [
                round((start - capture_start) / 1000.0, 3),
                round((end - capture_start) / 1000.0, 3),
            ]
            for start, end in segments
        ],
        "active_duration_s": round(active_us / 1e6, 3),
        "capture_duration_s": round(capture_us / 1e6, 3),
        "active_capture_fraction": round(active_us / capture_us, 6),
        "density_threshold_events_per_bin": ACTIVE_DENSITY_THRESHOLD,
        "bin_size_us": ACTIVE_BIN_SIZE_US,
    }


def _matrix_locality_summary(
    matrix: np.ndarray,
    rank_node_names: dict[int, str] | None,
    *,
    metric: str = "rank traffic matrix bytes-or-duration proxy",
) -> dict[str, Any]:
    """Summarize inferred rank traffic as intra-node vs inter-node proxy."""
    if matrix.size == 0:
        return {}
    intra = 0.0
    inter = 0.0
    unknown = 0.0
    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            val = float(matrix[src, dst])
            if val <= 0:
                continue
            src_node = (rank_node_names or {}).get(src)
            dst_node = (rank_node_names or {}).get(dst)
            if src_node is None or dst_node is None:
                unknown += val
            elif src_node == dst_node:
                intra += val
            else:
                inter += val
    total = intra + inter + unknown
    return {
        "metric": metric,
        "intra_node": intra,
        "inter_node": inter,
        "unknown_node": unknown,
        "total": total,
        "inter_node_fraction": (inter / total) if total else 0.0,
    }


def _flat_sanity_rows(sanity: dict[str, Any]) -> list[tuple[str, Any]]:
    parallel = sanity.get("parallelism", {})
    benchmark = sanity.get("benchmark_params", {})
    results = sanity.get("results", {})
    traces = sanity.get("traces", {})
    perf = sanity.get("performance", {})
    quality = sanity.get("trace_quality", {})
    return [
        ("job_id", sanity.get("job_id")),
        ("model", sanity.get("model")),
        ("TP / PP / EP / DP", (
            f"{parallel.get('TP')} / {parallel.get('PP')} / "
            f"{parallel.get('EP')} / {parallel.get('DP')}"
        )),
        ("nodes", ", ".join(sanity.get("nodes", []))),
        ("workers expected", traces.get("expected_workers")),
        ("workers found", traces.get("found_workers")),
        ("SP", benchmark.get("SP")),
        ("SD", benchmark.get("SD")),
        ("NUM_PROMPTS", benchmark.get("NUM_PROMPTS")),
        ("REQUEST_RATE", benchmark.get("REQUEST_RATE")),
        ("burstiness", benchmark.get("burstiness")),
        ("max_num_seqs", benchmark.get("max_num_seqs")),
        ("max_num_batched_tokens", benchmark.get("max_num_batched_tokens")),
        ("custom_output_len", benchmark.get("custom_output_len")),
        ("ignore_eos", benchmark.get("ignore_eos")),
        ("successful requests", results.get("successful_requests")),
        ("failed requests", results.get("failed_requests")),
        ("mean TTFT ms", perf.get("mean_ttft_ms")),
        ("mean TPOT ms", perf.get("mean_tpot_ms")),
        ("output tok/s", perf.get("output_token_throughput_tps")),
        ("trace files readable", traces.get("trace_files_readable")),
        ("active duration s", quality.get("active_duration_s")),
        ("active segments", quality.get("segment_count")),
        ("trace quality grade", quality.get("paper_figure_quality")),
    ]


def _traffic_class_rows(
    job_meta: dict[str, Any],
    summary_meta: dict[str, Any],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    fabric = summary_meta.get("merged_fabric_comm_ops", {})
    movement = summary_meta.get("merged_data_movement_ops", {})
    nccl_log_ops = (summary_meta.get("nccl_log") or {}).get("ops", {})
    nccl_logical_bytes = sum(
        int((nccl_log_ops.get(op) or {}).get("logical_bytes", 0))
        for op in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
    )
    avg_duty = quality.get("avg_duty_pct", {})
    tp_count = sum(
        int((fabric.get(op) or {}).get("count", 0))
        for op in ("all_reduce", "all_gather", "reduce_scatter")
    )
    tp_dur_us = sum(
        int((fabric.get(op) or {}).get("dur_us", 0))
        for op in ("all_reduce", "all_gather", "reduce_scatter")
    )
    p2p_count = int((fabric.get("point_to_point") or {}).get("count", 0))
    p2p_dur_us = int((fabric.get("point_to_point") or {}).get("dur_us", 0))
    return [
        {
            "traffic_class": "TP collective",
            "frequency": f"{tp_count} events",
            "message_size": (
                "CUPTI bytes available"
                if quality.get("nccl_bytes_total", 0) > 0
                else (
                    "NCCL log op-size bytes available"
                    if nccl_logical_bytes > 0
                    else "logical bytes unavailable"
                )
            ),
            "synchronization_sensitivity": "high",
            "locality": "TP rank group",
            "burstiness": "see comm_start_delta_cdf and active timeline",
            "candidate_plane": "low-latency collective plane",
            "evidence": {
                "duration_s": round(tp_dur_us / 1e6, 4),
                "network_collective_duty_pct": (
                    avg_duty.get("network_collective")
                ),
                "nccl_logical_bytes": nccl_logical_bytes,
            },
        },
        {
            "traffic_class": "PP activation",
            "frequency": f"{p2p_count} events",
            "message_size": "logical NVTX bytes when PP markers are present",
            "synchronization_sensitivity": "medium",
            "locality": "pipeline stage boundary",
            "burstiness": "stage-aligned bursts",
            "candidate_plane": "bandwidth-oriented P2P plane",
            "evidence": {
                "duration_s": round(p2p_dur_us / 1e6, 4),
                "pipeline_parallel": job_meta.get("pipeline_parallel", 1),
                "network_p2p_duty_pct": avg_duty.get("network_p2p"),
            },
        },
        {
            "traffic_class": "Host/control",
            "frequency": (
                f"{int((movement.get('host_transfer') or {}).get('count', 0))} "
                "host-transfer/runtime events"
            ),
            "message_size": "inspect host_device_transfer_size_histogram",
            "synchronization_sensitivity": "control path",
            "locality": "CPU/GPU runtime",
            "burstiness": "runtime dependent",
            "candidate_plane": "separate control/management plane",
            "evidence": {
                "host_transfer_duty_pct": avg_duty.get("host_transfer"),
                "control_duty_pct": avg_duty.get("control"),
            },
        },
    ]


def _missing_data_report(
    job_meta: dict[str, Any],
    per_node_summaries: dict[str, dict[str, Any]],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    any_message_bytes = any(
        (s.get("message_size_cdf") or {}).get("bytes_available")
        for s in per_node_summaries.values()
    )
    p2p_duty = (quality.get("avg_duty_pct") or {}).get("network_p2p", 0.0) or 0.0
    pp = int(job_meta.get("pipeline_parallel", 1))
    return [
        {
            "item": "CUPTI NCCL bytes unavailable",
            "status": (
                "missing" if quality.get("nccl_bytes_total", 0) == 0
                else "available"
            ),
            "impact": (
                "Use NVTX logical bytes, NCCL log op sizes, or theoretical "
                "estimates."
            ),
        },
        {
            "item": "Comm NVTX logical bytes",
            "status": "available" if any_message_bytes else "missing",
            "impact": "Required for message_size_cdf without estimation.",
        },
        {
            "item": "PP markers",
            "status": (
                "not_applicable" if pp <= 1
                else ("visible" if p2p_duty > 0 else "missing")
            ),
            "impact": "Required for PP activation message-size claims.",
        },
        {
            "item": "Active segments",
            "status": (
                "available"
                if any(s.get("active_segments_us")
                       for s in per_node_summaries.values())
                else "missing"
            ),
            "impact": "Needed to exclude startup, idle, and teardown.",
        },
        {
            "item": "Worker reports",
            "status": (
                "ok"
                if quality.get("trace_count") == quality.get("expected_workers")
                else "mismatch"
            ),
            "impact": "Rank-level plots are incomplete if reports are missing.",
        },
    ]


def _run_active_window_summary(
    per_node_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segments = merge_intervals([
        (int(start), int(end))
        for s in per_node_summaries.values()
        for start, end in s.get("active_segments_us", [])
    ])
    active_union_ms = sum(end - start for start, end in segments) / 1000.0

    capture_start_us: int | None = None
    capture_end_us: int | None = None
    for summary in per_node_summaries.values():
        events = summary.get("events") or []
        if events:
            local_start = min(e["ts"] for e in events)
            local_end = max(e["end"] for e in events)
            capture_start_us = (
                local_start if capture_start_us is None
                else min(capture_start_us, local_start)
            )
            capture_end_us = (
                local_end if capture_end_us is None
                else max(capture_end_us, local_end)
            )

    if capture_start_us is not None and capture_end_us is not None:
        full_capture_span_ms = (capture_end_us - capture_start_us) / 1000.0
    else:
        full_capture_span_ms = max(
            (float(s.get("trace_span_s", 0.0)) * 1000.0)
            for s in per_node_summaries.values()
        )

    compact_gap_ms = 10.0 if len(segments) > 1 else 0.0
    compact_x_max_ms = active_union_ms + compact_gap_ms * max(
        len(segments) - 1, 0
    )
    return {
        "segment_count": len(segments),
        "segments_us": [list(s) for s in segments],
        "active_union_ms": round(active_union_ms, 3),
        "full_capture_span_ms": round(full_capture_span_ms, 3),
        "active_fraction": (
            round(active_union_ms / full_capture_span_ms, 6)
            if full_capture_span_ms else None
        ),
        "axis_report": {
            "fabric_comm_timeline_active": {
                "mode": "active segments compacted; idle gaps removed",
                "plotted_x_min_ms": 0.0,
                "plotted_x_max_ms": round(compact_x_max_ms, 3),
                "expected_span_ms": round(compact_x_max_ms, 3),
            },
            "fabric_comm_timeline_trimmed": {
                "mode": "active event envelope; full capture preamble removed",
                "plotted_x_min_ms": 0.0,
                "expected_span_ms": round(active_union_ms, 3),
                "note": (
                    "Actual x_max is first-to-last active fabric event; "
                    "it should be close to active_union_ms and far below "
                    "full_capture_span_ms."
                ),
            },
            "active_window_detection": {
                "mode": "full capture axis with active segment overlays",
                "expected_span_ms": round(full_capture_span_ms, 3),
            },
        },
    }


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


def _find_nccl_logs(job_dir: Path) -> list[Path]:
    root = job_dir / "nccl_logs"
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.glob("*.log")
        if p.is_file() and p.stat().st_size > 0
    )


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
    loaded: list[
        tuple[
            Path,
            str,
            list[dict[str, Any]],
            list[dict[str, Any]],
            ClassificationReport,
        ]
    ],
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


def build_nccl_log_summary_plots(
    job_dir: Path,
    summary_dir: Path,
    *,
    job_meta: dict[str, Any],
    n_ranks: int,
    rank_node_names: dict[int, str],
    job_label: str,
) -> dict[str, Any]:
    """Build NCCL INFO log derived summaries next to Nsight summaries."""
    log_paths = _find_nccl_logs(job_dir)
    if not log_paths:
        return {}

    parsed = parse_nccl_logs(log_paths)
    world_size = max(infer_nccl_world_size(parsed, fallback=n_ranks), n_ranks, 2)
    tp = int(job_meta.get("tensor_parallel", 1))
    pp = int(job_meta.get("pipeline_parallel", 1))
    op_summary = summarize_nccl_log_ops(parsed)
    matrices = build_nccl_log_rank_matrices(parsed, n_ranks=world_size)
    channel_matrix, channel_payload = build_nccl_channel_matrix(
        parsed, n_ranks=world_size
    )
    rank_meta = nccl_rank_summary(parsed)

    plot_nccl_log_ops_breakdown(
        op_summary.get("ops", {}),
        summary_dir / f"nccl_log_ops_breakdown{PLOT_EXT}",
        title=f"{job_label}: NCCL INFO operation sizes",
    )

    logical_matrix = matrices["logical_bytes"]
    algorithmic_matrix = matrices["algorithmic_bytes_estimated"]
    plot_traffic_heatmap(
        algorithmic_matrix,
        summary_dir / f"nccl_log_rank_traffic_heatmap{PLOT_EXT}",
        title=f"{job_label}: NCCL-log rank traffic estimate",
        xlabel="Destination rank",
        ylabel="Source rank",
        colorbar_label="Estimated algorithmic bytes from NCCL log op sizes",
        tp=tp,
        pp=pp,
        rank_node_names=rank_node_names,
    )
    plot_traffic_heatmap(
        logical_matrix,
        summary_dir / f"nccl_logical_rank_traffic_heatmap{PLOT_EXT}",
        title=f"{job_label}: NCCL-log logical op bytes",
        xlabel="Destination rank",
        ylabel="Source rank",
        colorbar_label="Logical op bytes from NCCL log, evenly split across peers",
        tp=tp,
        pp=pp,
        rank_node_names=rank_node_names,
    )
    plot_traffic_heatmap(
        channel_matrix,
        summary_dir / f"nccl_log_topology_heatmap{PLOT_EXT}",
        title=f"{job_label}: NCCL channel setup links",
        xlabel="Destination rank",
        ylabel="Source rank",
        colorbar_label="NCCL channel setup link count",
        tp=tp,
        pp=pp,
        rank_node_names=rank_node_names,
    )

    locality = _matrix_locality_summary(
        algorithmic_matrix,
        rank_node_names,
        metric="NCCL log algorithmic byte estimate",
    )
    if locality:
        write_summary_json(
            summary_dir / "nccl_log_intra_vs_inter_node_comm.json",
            locality,
        )
        plot_key_value_table(
            [
                ("metric", locality.get("metric")),
                ("intra_node", round(locality.get("intra_node", 0.0), 3)),
                ("inter_node", round(locality.get("inter_node", 0.0), 3)),
                ("unknown_node", round(locality.get("unknown_node", 0.0), 3)),
                (
                    "inter_node_fraction",
                    round(locality.get("inter_node_fraction", 0.0), 6),
                ),
            ],
            summary_dir / f"nccl_log_intra_vs_inter_node_comm{PLOT_EXT}",
            title=f"{job_label}: NCCL-log intra-node vs inter-node estimate",
        )

    rank_payload = {
        "metric": "NCCL log derived rank traffic",
        "notes": [
            (
                "logical_bytes uses NCCL INFO 'N Bytes' operation sizes, "
                "distributed evenly across collective peers."
            ),
            (
                "algorithmic_bytes_estimated applies collective-specific "
                "rank transfer factors; it is not measured per-link wire bytes."
            ),
            (
                "channel_topology_count counts NCCL channel setup links and "
                "does not represent runtime byte volume."
            ),
        ],
        "world_size": world_size,
        "rank_meta": rank_meta,
        "locality": locality,
        "logical_bytes_matrix": logical_matrix.tolist(),
        "algorithmic_bytes_estimated_matrix": algorithmic_matrix.tolist(),
    }
    topology_payload = {
        "metric": "NCCL channel setup link count",
        "matrix": channel_payload["matrix"].tolist(),
        "ring_matrix": channel_payload["ring_matrix"].tolist(),
        "meta": channel_payload["meta"],
    }
    write_summary_json(
        summary_dir / "nccl_log_ops_breakdown.json",
        {
            "scope": "nccl_info_log",
            "ops": op_summary.get("ops", {}),
            "per_rank_ops": op_summary.get("per_rank_ops", {}),
        },
    )
    write_summary_json(summary_dir / "nccl_log_rank_traffic.json", rank_payload)
    write_summary_json(summary_dir / "nccl_log_topology.json", topology_payload)

    out = {
        "log_files": [str(p.relative_to(job_dir)) for p in log_paths],
        "world_size": world_size,
        "rank_meta": rank_meta,
        "ops": op_summary.get("ops", {}),
        "rank_traffic": {
            "matrix_sum_algorithmic_bytes_estimated": float(
                algorithmic_matrix.sum()
            ),
            "matrix_sum_logical_bytes": float(logical_matrix.sum()),
            "locality": locality,
        },
        "topology": topology_payload["meta"],
    }
    write_summary_json(summary_dir / "nccl_log_summary.json", out)
    return out


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

    active_segments = infer_active_segments_us(
        events,
        density_threshold=ACTIVE_DENSITY_THRESHOLD,
        bin_size_us=ACTIVE_BIN_SIZE_US,
    )
    active_window = _active_window_envelope(active_segments)
    if active_window is not None:
        active_s = sum(end - start for start, end in active_segments) / 1e6
        offset_s = (
            active_window[0] - (time_origin_us or min(e["ts"] for e in events))
        ) / 1e6
        print(
            f"  Active inference segments: {len(active_segments)} segment(s), "
            f"{active_s:.1f}s total (+{offset_s:.1f}s)"
        )
    active_events = (
        trim_events_to_windows(events, active_segments)
        if active_segments else events
    )

    duty = (
        duty_by_sub_windows(active_events, active_segments)
        if active_segments else duty_by_sub(events)
    )
    if pp > 1 and pp_reference_us_per_rank > 0:
        duty = adjust_duty_with_pp_balance(
            duty,
            active_events,
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
    if active_window is not None:
        plot_decomposed_timeline(
            active_events,
            trace_out / f"decomposed_timeline_active{PLOT_EXT}",
            title=f"{decomp_title} [active window only]",
            max_ms=max_plot_ms,
            time_origin_us=active_window[0],
            pp_comm_split=pp_comm_split,
        )
        plot_active_window_detection(
            events,
            active_segments,
            trace_out / f"active_window_detection{PLOT_EXT}",
            title=f"{node} (rank {global_rank}): active-window detection",
            density_threshold=ACTIVE_DENSITY_THRESHOLD,
            bin_size_us=ACTIVE_BIN_SIZE_US,
            time_origin_us=time_origin_us,
        )
        plot_active_segments_timeline(
            active_segments,
            trace_out / f"active_segments_timeline{PLOT_EXT}",
            title=f"{node} (rank {global_rank}): active segments",
            time_origin_us=time_origin_us or min(e["ts"] for e in events),
        )
    plot_comm_timeline(
        events,
        trace_out / f"comm_timeline{PLOT_EXT}",
        title=f"{node} (rank {global_rank}): communication-only timeline",
        max_ms=max_plot_ms,
        time_origin_us=time_origin_us,
    )
    plot_fabric_comm_timeline(
        events,
        trace_out / f"fabric_comm_timeline{PLOT_EXT}",
        title=f"{node} (rank {global_rank}): collective communication timeline",
        max_ms=max_plot_ms,
        time_origin_us=time_origin_us,
    )
    plot_fabric_comm_timeline_trimmed(
        active_events,
        trace_out / f"fabric_comm_timeline_trimmed{PLOT_EXT}",
        title=(
            f"{node} (rank {global_rank}): fabric comm "
            "(active envelope, gaps retained)"
        ),
    )
    plot_fabric_comm_timeline_active(
        events,
        active_segments,
        trace_out / f"fabric_comm_timeline_active{PLOT_EXT}",
        title=f"{node} (rank {global_rank}): fabric comm (active segments)",
        max_ms=max_plot_ms,
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
    message_size_summary = plot_message_size_cdf(
        comm_nvtx_records,
        trace_out / f"message_size_cdf{PLOT_EXT}",
        title=f"{node}: logical message size CDF",
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

    nocomm = nocomm_windows(active_events)
    if nocomm:
        plot_window_cdf(
            nocomm,
            trace_out / f"nocomm_windows_cdf{PLOT_EXT}",
            xlabel="No-comm window (ms)",
            title=f"{node}: CDF of gaps without communication",
        )
    deltas = comm_delta(active_events)
    if deltas:
        plot_window_cdf(
            deltas,
            trace_out / f"comm_start_delta_cdf{PLOT_EXT}",
            xlabel="Δt between comm starts (ms)",
            title=f"{node}: CDF of comm start deltas",
        )

    idle = gpu_idle_windows_ms(active_events)
    if idle:
        plot_window_cdf(
            idle,
            trace_out / f"gpu_idle_windows_cdf{PLOT_EXT}",
            xlabel="GPU idle window (ms)",
            title=f"{node}: CDF of GPU idle gaps (active window)",
        )

    idle_ctx = plot_idle_context(active_events, trace_out, prefix=node, min_gap_us=1000)
    if idle_ctx.get("gap_count", 0):
        print(
            f"  [{node}] idle context (active window): {idle_ctx['gap_count']} gaps, "
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
    trace_span_s = (max(e["end"] for e in events) - min(e["ts"] for e in events)) / 1e6
    active_window_s = (
        sum(end - start for start, end in active_segments) / 1e6
        if active_segments else trace_span_s
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
        "trace_span_s": round(trace_span_s, 2),
        "active_window_s": round(active_window_s, 2),
        "active_window_us": list(active_window) if active_window else None,
        "active_segments_us": [list(s) for s in active_segments],
        "active_segments": _active_segments_summary(events, active_segments),
        "event_count": len(events),
        "active_event_count": len(active_events),
        "duty_by_subcategory": duty,
        "expert_traffic_gb_heuristic": expert_gb,
        "message_stats": msg_stats,
        "message_size_cdf": message_size_summary,
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
    plot_average_duty_pct(
        node_duty,
        summary_dir / f"category_duty_by_run{PLOT_EXT}",
        title=f"{job_label}: category duty by run",
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

    node_events = {
        n: d["events"] for n, d in node_data.items() if d.get("events")
    }
    if node_events:
        all_comm_events: list[dict[str, Any]] = []
        all_active_comm_events: list[dict[str, Any]] = []
        all_active_segments: list[tuple[int, int]] = []
        for n, evs in node_events.items():
            all_comm_events.extend(evs)
            segments = [
                (int(start), int(end))
                for start, end in node_data[n].get("active_segments_us", [])
            ]
            if segments:
                all_active_segments.extend(segments)
                all_active_comm_events.extend(trim_events_to_windows(evs, segments))
            else:
                all_active_comm_events.extend(evs)
        merged_active_segments = merge_intervals(all_active_segments)
        plot_comm_timeline(
            all_comm_events,
            summary_dir / f"comm_timeline{PLOT_EXT}",
            title=f"{job_label}: communication-only timeline (all ranks)",
            max_ms=max_plot_ms,
            time_origin_us=time_origin_us,
        )
        plot_fabric_comm_timeline(
            all_comm_events,
            summary_dir / f"fabric_comm_timeline{PLOT_EXT}",
            title=f"{job_label}: collective communication timeline (all ranks)",
            max_ms=max_plot_ms,
            time_origin_us=time_origin_us,
        )
        plot_fabric_comm_timeline_trimmed(
            all_active_comm_events,
            summary_dir / f"fabric_comm_timeline_trimmed{PLOT_EXT}",
            title=f"{job_label}: fabric comm (active envelope, gaps retained)",
        )
        plot_fabric_comm_timeline_active(
            all_comm_events,
            merged_active_segments,
            summary_dir / f"fabric_comm_timeline_active{PLOT_EXT}",
            title=f"{job_label}: fabric comm (active segments, all ranks)",
            max_ms=max_plot_ms,
        )
        plot_active_window_detection(
            all_comm_events,
            merged_active_segments,
            summary_dir / f"active_window_detection{PLOT_EXT}",
            title=f"{job_label}: active-window detection (all ranks)",
            density_threshold=ACTIVE_DENSITY_THRESHOLD,
            bin_size_us=ACTIVE_BIN_SIZE_US,
            time_origin_us=time_origin_us,
        )
        plot_active_segments_timeline(
            merged_active_segments,
            summary_dir / f"active_segments_timeline{PLOT_EXT}",
            title=f"{job_label}: active segments (all ranks)",
            time_origin_us=time_origin_us or (
                min(e["ts"] for e in all_comm_events)
                if all_comm_events else None
            ),
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
        locality = _matrix_locality_summary(merged, rank_node_names)
        if locality:
            write_summary_json(
                summary_dir / "intra_vs_inter_node_comm.json",
                locality,
            )
            plot_key_value_table(
                [
                    ("metric", locality.get("metric")),
                    ("intra_node", round(locality.get("intra_node", 0.0), 3)),
                    ("inter_node", round(locality.get("inter_node", 0.0), 3)),
                    ("unknown_node", round(locality.get("unknown_node", 0.0), 3)),
                    (
                        "inter_node_fraction",
                        round(locality.get("inter_node_fraction", 0.0), 6),
                    ),
                ],
                summary_dir / f"intra_vs_inter_node_comm{PLOT_EXT}",
                title=f"{job_label}: intra-node vs inter-node comm proxy",
            )
        rank_summary = {
            "n_ranks": n_ranks_use,
            "ranks_seen": ranks_seen,
            "nodes": sorted(node_data),
            "matrix_sum": float(merged.sum()),
            "locality": locality,
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
        segments = [
            (int(start), int(end))
            for start, end in data.get("active_segments_us", [])
        ]
        node_active = trim_events_to_windows(events, segments) if segments else events
        deltas = comm_delta(node_active)
        if deltas:
            cdf_series[f"{node} comm Δ"] = deltas
            pooled["comm_delta"].extend(deltas)
        nocomm = nocomm_windows(node_active)
        if nocomm:
            cdf_series[f"{node} no-comm"] = nocomm
            pooled["nocomm"].extend(nocomm)
        idle = gpu_idle_windows_ms(node_active)
        if idle:
            cdf_series[f"{node} GPU idle"] = idle
            pooled["gpu_idle"].extend(idle)
        all_gaps.extend(analyze_idle_gaps(node_active, min_gap_us=1000))

    if pooled["comm_delta"]:
        plot_multi_window_cdf(
            {
                "pooled": pooled["comm_delta"],
                **{
                    k: v for k, v in cdf_series.items()
                    if "comm Δ" in k
                },
            },
            summary_dir / f"comm_start_delta_cdf{PLOT_EXT}",
            xlabel="Δt between comm starts (ms)",
            title=f"{job_label}: comm start delta CDF (per node + pooled)",
        )
    if pooled["nocomm"]:
        plot_multi_window_cdf(
            {
                "pooled": pooled["nocomm"],
                **{
                    k: v for k, v in cdf_series.items()
                    if "no-comm" in k
                },
            },
            summary_dir / f"nocomm_windows_cdf{PLOT_EXT}",
            xlabel="No-comm window (ms)",
            title=f"{job_label}: no-comm window CDF (per node + pooled)",
        )
    if pooled["gpu_idle"]:
        plot_multi_window_cdf(
            {
                "pooled": pooled["gpu_idle"],
                **{
                    k: v for k, v in cdf_series.items()
                    if "GPU idle" in k
                },
            },
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


def evaluate_trace_quality(
    per_node_summaries: dict[str, dict[str, Any]],
    job_meta: dict[str, Any],
    *,
    job_dir: Path,
) -> dict[str, Any]:
    """Run quality gate checks and return a quality report with warnings."""
    warnings: list[str] = []
    grades: dict[str, str] = {}

    # Aggregate duty across ranks
    all_duty: list[dict[str, float]] = [
        s["duty_by_subcategory"] for s in per_node_summaries.values()
        if "duty_by_subcategory" in s
    ]
    if not all_duty:
        warnings.append("No duty data available")
        grades["overall"] = "unusable"
        return {"warnings": warnings, "grades": grades}

    avg_duty: dict[str, float] = {}
    for d in all_duty:
        for k, v in d.items():
            avg_duty[k] = avg_duty.get(k, 0.0) + v / len(all_duty)

    compute_subs = [
        "matmul_gemm",
        "attention_comp",
        "moe_expert",
        "moe_routing",
        "add_norm_comp",
        "gate_comp",
        "rotary_embedding",
        "kv_cache_write",
        "sampling_overhead",
        "other_compute",
    ]
    total_compute_duty = sum(avg_duty.get(s, 0.0) for s in compute_subs)
    nccl_duty = (
        avg_duty.get("network_collective", 0.0)
        + avg_duty.get("network_p2p", 0.0)
    )

    # Check 1: compute duty
    if total_compute_duty < 0.01:
        warnings.append(
            f"Compute duty extremely low ({total_compute_duty*100:.2f}%) "
            "— model kernels may be missing or window not trimmed"
        )
        grades["compute_breakdown"] = "not_usable"
    elif total_compute_duty < 0.05:
        warnings.append(
            f"Compute duty low ({total_compute_duty*100:.1f}%) — "
            "verify decode batch size"
        )
        grades["compute_breakdown"] = "marginal"
    else:
        grades["compute_breakdown"] = "usable"

    # Check 2: NCCL bytes
    nccl_bytes_total = 0
    for s in per_node_summaries.values():
        fb = s.get("fabric_ops_breakdown", {})
        for op_name, op_data in fb.items():
            if isinstance(op_data, dict):
                nccl_bytes_total += op_data.get("bytes", 0)
    if nccl_bytes_total == 0 and nccl_duty > 0:
        warnings.append(
            "NCCL bytes = 0 for all collectives — "
            "byte volume analysis not available"
        )
        grades["byte_volume"] = "not_usable"
    else:
        grades["byte_volume"] = "usable"

    # Check 3: idle dominance
    idle_duty = avg_duty.get("idle", 0.0) + avg_duty.get("none", 0.0)
    if idle_duty > 0.5:
        warnings.append(
            f"Idle/none duty is {idle_duty*100:.1f}% — "
            "trace may include significant startup or capture skew"
        )
        grades["idle_analysis"] = "not_usable"
    else:
        grades["idle_analysis"] = "usable"

    # Check 4: active window vs trace span
    for s in per_node_summaries.values():
        trace_span = s.get("trace_span_s", 0)
        active_span = s.get("active_window_s", trace_span)
        if trace_span > 0 and active_span / trace_span < 0.1:
            warnings.append(
                f"Active window ({active_span:.1f}s) is <10% of trace span "
                f"({trace_span:.1f}s) "
                "— large startup/teardown overhead captured"
            )
            break

    # Check 5: worker count
    expected_workers = (
        job_meta.get("tensor_parallel", 1)
        * job_meta.get("pipeline_parallel", 1)
    )
    actual_workers = len(per_node_summaries)
    if actual_workers != expected_workers:
        warnings.append(
            f"Expected {expected_workers} workers "
            f"(TP={job_meta.get('tensor_parallel', 1)} "
            f"× PP={job_meta.get('pipeline_parallel', 1)}), "
            f"found {actual_workers} traces"
        )
        grades["worker_count"] = "mismatch"
    else:
        grades["worker_count"] = "ok"

    # Check 6: TP collective visibility
    if nccl_duty > 0.001:
        grades["tp_collective_visibility"] = "usable"
    else:
        grades["tp_collective_visibility"] = "not_usable"

    # Overall
    if all(v in ("usable", "ok") for v in grades.values()):
        grades["paper_figure_quality"] = "ready"
    elif (
        grades.get("compute_breakdown") == "usable"
        and grades.get("idle_analysis") == "usable"
    ):
        grades["paper_figure_quality"] = "partial"
    else:
        grades["paper_figure_quality"] = "not_yet"

    quality_report = {
        "warnings": warnings,
        "grades": grades,
        "avg_duty_pct": {k: round(v * 100, 2) for k, v in avg_duty.items()},
        "total_compute_duty_pct": round(total_compute_duty * 100, 2),
        "nccl_duty_pct": round(nccl_duty * 100, 2),
        "nccl_bytes_total": nccl_bytes_total,
        "trace_count": actual_workers,
        "expected_workers": expected_workers,
    }
    return quality_report


def generate_run_sanity(
    job_dir: Path,
    job_meta: dict[str, Any],
    per_node_summaries: dict[str, dict[str, Any]],
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate run_sanity.json with experiment metadata and basic validation."""
    tp = job_meta.get("tensor_parallel", 1)
    pp = job_meta.get("pipeline_parallel", 1)
    ep = job_meta.get("expert_parallel", 1)
    nodes = sorted({
        str(s.get("node") or prefix).split("_")[0]
        for prefix, s in per_node_summaries.items()
    })
    trace_files = [s.get("trace") for s in per_node_summaries.values()]
    merged_active_segments = merge_intervals([
        (int(start), int(end))
        for s in per_node_summaries.values()
        for start, end in s.get("active_segments_us", [])
    ])
    active_duration_s = sum(
        end - start for start, end in merged_active_segments
    ) / 1e6
    segment_count = len(merged_active_segments)

    sanity: dict[str, Any] = {
        "job_id": job_meta.get("job_id") or job_dir.name,
        "job_dir": str(job_dir),
        "model": job_meta.get("model_id", "unknown"),
        "nodes": nodes,
        "parallelism": {
            "TP": tp,
            "PP": pp,
            "EP": ep,
            "DP": job_meta.get("data_parallel", 1),
        },
        "benchmark_params": {
            "NUM_PROMPTS": job_meta.get("num_prompts"),
            "REQUEST_RATE": job_meta.get("request_rate"),
            "SP": job_meta.get("sp") or job_meta.get("max_model_len"),
            "SD": job_meta.get("sd") or job_meta.get("custom_output_len"),
            "burstiness": job_meta.get("burstiness"),
            "max_num_seqs": job_meta.get("max_num_seqs"),
            "max_num_batched_tokens": job_meta.get("max_num_batched_tokens"),
            "custom_output_len": job_meta.get("custom_output_len"),
            "ignore_eos": job_meta.get("ignore_eos"),
        },
        "results": {
            "successful_requests": job_meta.get("successful_requests"),
            "failed_requests": job_meta.get("failed_requests"),
        },
        "performance": {
            "mean_ttft_ms": job_meta.get("mean_ttft_ms"),
            "median_ttft_ms": job_meta.get("median_ttft_ms"),
            "p99_ttft_ms": job_meta.get("p99_ttft_ms"),
            "mean_tpot_ms": job_meta.get("mean_tpot_ms"),
            "median_tpot_ms": job_meta.get("median_tpot_ms"),
            "p99_tpot_ms": job_meta.get("p99_tpot_ms"),
            "request_throughput_rps": job_meta.get("request_throughput_rps"),
            "output_token_throughput_tps": (
                job_meta.get("output_token_throughput_tps")
            ),
            "total_token_throughput_tps": (
                job_meta.get("total_token_throughput_tps")
            ),
        },
        "traces": {
            "expected_workers": tp * pp,
            "found_workers": len(per_node_summaries),
            "trace_files": trace_files,
            "trace_files_readable": all(
                bool(path) and Path(str(path)).is_file()
                for path in trace_files
            ),
        },
        "trace_quality": {
            "active_duration_s": round(active_duration_s, 3),
            "segment_count": segment_count,
            "segments_us": [list(s) for s in merged_active_segments],
            "paper_figure_quality": (
                (quality or {}).get("grades", {}).get("paper_figure_quality")
            ),
            "warnings": (quality or {}).get("warnings", []),
        },
    }

    # Check nsys-rep readability
    nsys_reps = list(job_dir.rglob("*.nsys-rep"))
    sanity["nsys_rep_files"] = {
        "count": len(nsys_reps),
        "paths": [str(p.relative_to(job_dir)) for p in nsys_reps[:20]],
    }

    return sanity


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
        nccl_log_meta = build_nccl_log_summary_plots(
            job_dir,
            summary_dir,
            job_meta=job_meta,
            n_ranks=n_ranks_default,
            rank_node_names=rank_node_names,
            job_label=job_dir.name,
        )
        if nccl_log_meta:
            summary_meta["nccl_log"] = nccl_log_meta
        all_summaries["summary_plots"] = summary_meta

    write_summary_json(out_dir / "job_summary.json", all_summaries)

    # Request arrival and scheduler-token timelines (EngineCore iteration log)
    iteration_token_summary: dict[str, Any] = {}
    iterations = parse_iteration_log(slurm)
    if iterations:
        plot_request_timeline(
            iterations,
            out_dir / f"request_timeline{PLOT_EXT}",
            title=f"{job_dir.name}: request arrival & active batch",
        )
        max_batched = job_meta.get("max_num_batched_tokens")
        iteration_token_summary = plot_batch_tokens_per_iteration(
            iterations,
            out_dir / f"batch_tokens_per_iteration{PLOT_EXT}",
            title=f"{job_dir.name}: tokens per scheduler iteration",
            max_num_batched_tokens=(
                int(max_batched) if max_batched is not None else None
            ),
        )
        write_summary_json(
            out_dir / "batch_tokens_per_iteration.json",
            {
                "summary": iteration_token_summary,
                "iterations": [
                    {
                        **it,
                        "prefill_tokens": int(it.get("context_tokens", 0)),
                        "decode_tokens": int(it.get("generation_tokens", 0)),
                        "total_tokens": (
                            int(it.get("context_tokens", 0))
                            + int(it.get("generation_tokens", 0))
                        ),
                        "phase": (
                            "mixed"
                            if (
                                int(it.get("context_tokens", 0)) > 0
                                and int(it.get("generation_tokens", 0)) > 0
                            )
                            else (
                                "prefill"
                                if int(it.get("context_tokens", 0)) > 0
                                else (
                                    "decode"
                                    if int(it.get("generation_tokens", 0)) > 0
                                    else "idle"
                                )
                            )
                        ),
                    }
                    for it in iterations
                ],
            },
        )

    # Time breakdown stacked bar (like the parallelism decomposition figure)
    if per_node_summaries:
        all_active_for_bar: list[dict[str, Any]] = []
        for prefix_key, s in per_node_summaries.items():
            evts = s.get("events", [])
            w = s.get("active_window_us")
            if w:
                evts = trim_events_to_window(evts, tuple(w))
            all_active_for_bar.extend(evts)
        cat_seconds = _map_events_to_bar_categories(all_active_for_bar)
        # Average across ranks (events from all ranks were summed)
        n_ranks = len(per_node_summaries)
        cat_seconds_per_rank = {k: v / n_ranks for k, v in cat_seconds.items()}

        config_label = f"T{tp}-P{pp}"
        if int(job_meta.get("expert_parallel", 1)) > 1:
            config_label = f"E{job_meta['expert_parallel']}-{config_label}"

        plot_time_breakdown_bar(
            [{"label": config_label, "category_seconds": cat_seconds_per_rank}],
            out_dir / f"time_breakdown_bar{PLOT_EXT}",
            title=f"{job_meta.get('model_id', job_dir.name)}: time breakdown (active window, per rank)",
        )

        # Single-layer breakdown (average across all ranks)
        all_layer_groups: list[list[dict[str, Any]]] = []
        for prefix_key, s in per_node_summaries.items():
            evts = s.get("events", [])
            w = s.get("active_window_us")
            if w:
                evts = trim_events_to_window(evts, tuple(w))
            rank_layers = extract_single_layer_events(evts, n_layers=48)
            all_layer_groups.extend(rank_layers)
        if all_layer_groups:
            layer_stats = plot_single_layer_breakdown(
                all_layer_groups,
                out_dir / f"single_layer_breakdown{PLOT_EXT}",
                title=f"{job_meta.get('model_id', job_dir.name)} ({config_label}): "
                      f"single layer breakdown (decode, batch={job_meta.get('num_prompts', '?')}, "
                      f"avg across {n_ranks} ranks)",
            )
            write_summary_json(out_dir / "single_layer_stats.json", layer_stats)

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
    all_active_events: list[dict[str, Any]] = []
    for _prefix_key, s in per_node_summaries.items():
        evts = s.get("events", [])
        segments = [
            (int(start), int(end))
            for start, end in s.get("active_segments_us", [])
        ]
        if segments:
            evts = trim_events_to_windows(evts, segments)
        all_active_events.extend(evts)

    top_kernels = _top_kernels_by_duration(all_active_events, top_n=50)
    host_transfer_breakdown = _host_transfer_breakdown(all_active_events)

    inv_path = out_dir / "classification_inventory.json"
    write_summary_json(
        inv_path,
        {
            "job_inventory": job_inventory,
            "top_kernels_by_duration": top_kernels,
            "host_transfer_breakdown": host_transfer_breakdown,
            "per_trace": {
                Path(r.trace_path).name: r.inventory_dict()
                for r in classification_reports
            },
        },
    )
    print(
        f"Wrote plotting_log.txt, plotting_log.json, and {inv_path.name} "
        f"under {out_dir}"
    )

    # Quality gate
    quality = evaluate_trace_quality(per_node_summaries, job_meta, job_dir=job_dir)
    write_summary_json(out_dir / "trace_quality.json", quality)
    if quality.get("warnings"):
        print("\n=== TRACE QUALITY WARNINGS ===")
        for w in quality["warnings"]:
            print(f"  ⚠ {w}")
        print(f"  Grades: {quality.get('grades', {})}")
        print()

    # Run sanity
    sanity = generate_run_sanity(
        job_dir, job_meta, per_node_summaries, quality=quality
    )
    write_summary_json(out_dir / "run_sanity.json", sanity)
    write_summary_json(out_dir / "run_sanity_summary.json", sanity)
    plot_key_value_table(
        _flat_sanity_rows(sanity),
        out_dir / f"run_sanity_summary{PLOT_EXT}",
        title=f"{job_dir.name}: run sanity summary",
    )

    summary_meta = all_summaries.get("summary_plots") or {}
    traffic_rows = _traffic_class_rows(job_meta, summary_meta, quality)
    write_summary_json(
        out_dir / "traffic_class_summary.json",
        {"rows": traffic_rows},
    )
    plot_key_value_table(
        [
            (
                row["traffic_class"],
                (
                    f"{row['frequency']}; {row['candidate_plane']}; "
                    f"evidence={row['evidence']}"
                ),
            )
            for row in traffic_rows
        ],
        out_dir / f"traffic_class_summary{PLOT_EXT}",
        title=f"{job_dir.name}: traffic class summary",
    )

    implications = [
        {
            "observed_pattern": "High-frequency TP collectives",
            "evidence_plot": "fabric_comm_breakdown, comm_start_delta_cdf",
            "topology_implication": "low-latency local collective support",
            "glass_design_implication": "place TP groups near each other",
        },
        {
            "observed_pattern": "Stage-to-stage PP traffic",
            "evidence_plot": "pipeline_p2p_timeline, pp_activation_message_size_cdf",
            "topology_implication": "predictable P2P bandwidth",
            "glass_design_implication": "dedicated switched or reconfigurable plane",
        },
        {
            "observed_pattern": "Tiny host/runtime transfers",
            "evidence_plot": "data_movement_breakdown, transfer_size_histogram",
            "topology_implication": "do not mix control with fabric claims",
            "glass_design_implication": "separate control/management plane",
        },
    ]
    write_summary_json(
        out_dir / "topology_implications.json",
        {"rows": implications},
    )

    missing_rows = _missing_data_report(job_meta, per_node_summaries, quality)
    write_summary_json(
        out_dir / "missing_data_report.json",
        {"rows": missing_rows},
    )
    plot_key_value_table(
        [(row["item"], f"{row['status']}: {row['impact']}") for row in missing_rows],
        out_dir / f"missing_data_report{PLOT_EXT}",
        title=f"{job_dir.name}: missing data report",
    )
    write_summary_json(out_dir / "trace_quality_dashboard.json", quality)
    active_window_summary = _run_active_window_summary(per_node_summaries)
    write_summary_json(
        out_dir / "plot_axis_report.json",
        active_window_summary["axis_report"],
    )

    generated = [
        "run_sanity_summary",
        "active_window_detection",
        "active_segments_timeline",
        "fabric_comm_breakdown",
        "category_duty_by_run",
        "decomposed_timeline_active",
        "rank_traffic_heatmap",
        "comm_start_delta_cdf",
        "nocomm_windows_cdf",
        "idle_transition_heatmap",
        "message_size_cdf",
        "traffic_class_summary",
        "missing_data_report",
    ]
    if summary_meta.get("nccl_log"):
        generated.extend([
            "nccl_log_ops_breakdown",
            "nccl_log_rank_traffic_heatmap",
            "nccl_logical_rank_traffic_heatmap",
            "nccl_log_topology_heatmap",
            "nccl_log_intra_vs_inter_node_comm",
        ])
    if iteration_token_summary:
        generated.append("batch_tokens_per_iteration")
    write_summary_json(
        out_dir / "communication_signature_suite.json",
        {
            "schema_version": 1,
            "generated_targets": generated,
            "nccl_log": summary_meta.get("nccl_log"),
            "iteration_tokens": iteration_token_summary,
            "active_window": active_window_summary,
            "traffic_class_summary": traffic_rows,
            "topology_implications": implications,
            "missing_data": missing_rows,
            "missing_data_report": {"rows": missing_rows},
            "trace_quality_dashboard": quality,
            "plot_axis_report": active_window_summary["axis_report"],
            "notes": [
                "fabric_comm_timeline_trimmed keeps active-envelope gaps",
                "fabric_comm_timeline_active compacts detected active segments",
            ],
        },
    )

    print(f"Wrote per-node plots under {out_dir}")
    if len(per_node_summaries) > 1:
        print(f"Wrote cross-node summary under {summary_dir}")


if __name__ == "__main__":
    main()
