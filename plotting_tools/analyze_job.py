#!/usr/bin/env python3
"""Analyze a vLLM job results folder (Chrome trace JSON + optional Slurm log)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python plotting_tools/analyze_job.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting_tools.plots import (  # noqa: E402
    build_traffic_matrix,
    collect_nccl_ops,
    comm_delta,
    nocomm_windows,
    plot_classic_timeline,
    plot_collective_ops_breakdown,
    plot_decomposed_timeline,
    plot_expert_traffic_volume,
    plot_message_stats,
    plot_traffic_heatmap,
    plot_traffic_volume_pct,
    gpu_idle_windows_ms,
    plot_idle_context,
    plot_window_cdf,
    write_collective_ops_table,
    write_summary_json,
)
from plotting_tools.trace_io import (  # noqa: E402
    duty_by_sub,
    load_chrome_trace,
    load_trace,
    parse_duration_events,
    parse_job_metadata,
    parse_nvtx_ranges,
    tag_phase,
)


def _find_traces(job_dir: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        for p in sorted(job_dir.rglob(pattern)):
            if p.stat().st_size == 0:
                continue
            if "summary" in p.name:
                continue
            found.append(p)
    return found


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
) -> dict:
    if trace_path.suffix.lower() == ".jsonl":
        events = load_trace(trace_path, strict=strict)
    else:
        raw = load_chrome_trace(trace_path)
        nvtx = parse_nvtx_ranges(raw)
        events = parse_duration_events(raw, strict=strict)
        if nvtx:
            tag_phase(events, nvtx)

    trace_out = out_dir / prefix
    trace_out.mkdir(parents=True, exist_ok=True)

    pp = job_meta.get("pipeline_parallel", 1)
    tp = job_meta.get("tensor_parallel", 1)
    ep = job_meta.get("expert_parallel", 1)
    parallel_label = f"TP={tp} PP={pp} EP={ep}"

    plot_decomposed_timeline(
        events,
        trace_out / "decomposed_timeline.png",
        title=f"{prefix}: decomposed compute / comm",
        max_ms=max_plot_ms,
    )
    plot_traffic_volume_pct(
        events,
        trace_out / "traffic_volume_pct.png",
        title=f"{prefix}: category time share",
        parallel_label=parallel_label,
    )
    plot_classic_timeline(
        events,
        trace_out / "compute_comm_control_timeline.png",
        title=f"{prefix}: compute / comm / control",
    )

    expert_gb = plot_expert_traffic_volume(
        events,
        trace_out / "expert_traffic_gb.png",
        num_experts=num_experts,
    )

    n_ranks_use = max(n_ranks, pp * tp, 2)
    mat = build_traffic_matrix(events, n_ranks_use)
    plot_traffic_heatmap(
        mat,
        trace_out / "all2all_traffic_heatmap.png",
        title=f"{prefix}: comm traffic matrix (rank x rank)",
    )

    msg_stats = plot_message_stats(
        events,
        trace_out / "message_stats_prefill_decode.png",
        prefix=prefix,
    )

    comm_breakdown = plot_collective_ops_breakdown(
        events,
        trace_out / "collective_ops_breakdown.png",
        title=f"{prefix}: communication ops (count & GPU time)",
    )
    write_summary_json(trace_out / "collective_ops_breakdown.json", comm_breakdown)

    nccl_ops = collect_nccl_ops(events)
    write_collective_ops_table(nccl_ops, trace_out / "collective_ops.json")

    nocomm = nocomm_windows(events)
    if nocomm:
        plot_window_cdf(
            nocomm,
            trace_out / "nocomm_windows_cdf.png",
            xlabel="No-comm window (ms)",
            title=f"{prefix}: CDF of gaps without communication",
        )
    deltas = comm_delta(events)
    if deltas:
        plot_window_cdf(
            deltas,
            trace_out / "comm_start_delta_cdf.png",
            xlabel="Δt between comm starts (ms)",
            title=f"{prefix}: CDF of comm start deltas",
        )

    idle = gpu_idle_windows_ms(events)
    if idle:
        plot_window_cdf(
            idle,
            trace_out / "gpu_idle_windows_cdf.png",
            xlabel="GPU idle window (ms)",
            title=f"{prefix}: CDF of GPU idle gaps",
        )

    idle_ctx = plot_idle_context(events, trace_out, prefix=prefix, min_gap_us=1000)
    if idle_ctx.get("gap_count", 0):
        print(
            f"  idle context: {idle_ctx['gap_count']} gaps, "
            f"top transition by time: {idle_ctx['top_transitions_by_ms'][:1]}"
        )

    summary = {
        "trace": str(trace_path),
        "event_count": len(events),
        "duty_by_subcategory": duty_by_sub(events),
        "expert_traffic_gb_heuristic": expert_gb,
        "message_stats": msg_stats,
        "comm_ops_breakdown": comm_breakdown,
        "nccl_op_count": len(nccl_ops),
        "idle_context": idle_ctx,
        "job_metadata": job_meta,
        "parallel_label": parallel_label,
    }
    write_summary_json(trace_out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot vLLM Nsight Chrome traces")
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Job results directory (e.g. results/7651157)",
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
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    out_dir = (args.out_dir or job_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slurm = args.slurm_out
    if slurm is None:
        candidates = sorted(job_dir.glob("*.out"))
        slurm = candidates[0] if candidates else None
    job_meta = parse_job_metadata(slurm)

    traces = args.trace or []
    if not traces:
        exported = job_dir / "exported"
        if exported.is_dir():
            traces = sorted(exported.rglob("*.json"))
        traces.extend(_find_traces(job_dir))

    if not traces:
        print(
            "No trace files found (.jsonl or .json under job-dir).\n"
            "Export from Nsight GUI (JSONL) or: bash plotting_tools/export_nsys.sh ...",
            file=sys.stderr,
        )
        sys.exit(1)

    all_summaries = {}
    for tp in traces:
        if tp.stat().st_size == 0:
            print(f"Skipping empty trace: {tp}")
            continue
        prefix = tp.stem.replace(".", "_")
        print(f"Analyzing {tp} -> {out_dir / prefix}")
        all_summaries[prefix] = analyze_trace(
            tp.resolve(),
            out_dir,
            prefix=prefix,
            job_meta=job_meta,
            n_ranks=args.n_ranks,
            num_experts=args.num_experts,
            strict=args.strict_classify,
            max_plot_ms=args.max_plot_ms,
        )

    write_summary_json(out_dir / "job_summary.json", all_summaries)
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
