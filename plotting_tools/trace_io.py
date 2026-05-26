"""Load Chrome-trace JSON and optional job metadata from Slurm logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from plotting_tools.classify import classify_event
from plotting_tools.comm_nvtx import finalize_comm_nvtx_records
from plotting_tools.nsys_jsonl import load_nsys_jsonl


def load_trace(
    path: Path,
    *,
    strict: bool = False,
    report: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load trace events and inner comm NVTX message records.

    Returns ``(events, comm_nvtx_records)``. Comm records are authoritative for
    count/shape/logical tensor bytes; outer iteration NVTX is used for timeline
    phase attribution on GPU events.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        events, iteration_ranges, comm_records = load_nsys_jsonl(
            path, strict=strict, report=report
        )
        if comm_records:
            comm_records, comm_stats = finalize_comm_nvtx_records(
                comm_records, iteration_ranges
            )
            if report is not None:
                report.comm_nvtx_dedup_removed += comm_stats.get(
                    "removed_nested_send_recv", 0
                )
                report.comm_nvtx_phases_assigned += comm_stats.get(
                    "phases_assigned_from_iteration", 0
                )
            print(
                f"  comm NVTX: {comm_stats.get('raw_count', 0):,} raw -> "
                f"{comm_stats.get('deduped_count', 0):,} logical messages "
                f"(removed {comm_stats.get('removed_nested_send_recv', 0):,} "
                f"nested send/recv; assigned "
                f"{comm_stats.get('phases_assigned_from_iteration', 0):,} "
                f"unknown phases from iteration ranges; "
                f"{comm_stats.get('unknown_phase_remaining', 0):,} still unknown; "
                f"{comm_stats.get('comm_conclusive', 0):,} conclusive prefill/decode, "
                f"excluded mixed={comm_stats.get('comm_mixed', 0):,}, "
                f"nearest={comm_stats.get('comm_untrusted_nearest', 0):,})"
            )
        if iteration_ranges or comm_records:
            tag_phase(events, iteration_ranges, comm_records)
        return events, comm_records
    raw = load_chrome_trace(path)
    events = parse_duration_events(raw, strict=strict)
    return events, []


def load_chrome_trace(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        data = data["traceEvents"]
    return data


def parse_duration_events(
    raw: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    uncl: list[str] = []
    for e in raw:
        if e.get("ph") != "X" or "ts" not in e or "dur" not in e:
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        kind, sub = classify_event(name, cat, uncl if strict else None)
        events.append(
            {
                "name": name,
                "cat": cat,
                "ts": int(e["ts"]),
                "dur": int(e["dur"]),
                "end": int(e["ts"]) + int(e["dur"]),
                "kind": kind,
                "sub": sub,
                "args": e.get("args") or {},
                "pid": e.get("pid"),
                "tid": e.get("tid"),
            }
        )
    if strict and uncl:
        from plotting_tools.classify import summarize_unclassified

        summarize_unclassified(uncl)
        raise SystemExit(1)
    return sorted(events, key=lambda x: x["ts"])


def parse_nvtx_ranges(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Instant events (ph=i/B/E) and duration NVTX (ph=X) for phase tagging."""
    ranges: list[dict[str, Any]] = []
    for e in raw:
        ph = e.get("ph")
        name = (e.get("name") or "").lower()
        if ph == "X" and "ts" in e and "dur" in e:
            ranges.append(
                {
                    "name": e.get("name", ""),
                    "ts": int(e["ts"]),
                    "end": int(e["ts"]) + int(e["dur"]),
                }
            )
        elif ph in ("b", "B") and "ts" in e:
            ranges.append({"name": e.get("name", ""), "ts": int(e["ts"]), "end": None})
        elif ph in ("e", "E") and "ts" in e and ranges:
            for r in reversed(ranges):
                if r.get("end") is None:
                    r["end"] = int(e["ts"])
                    break
    return [r for r in ranges if r.get("end") is not None]


def _narrowest_overlap_phase(
    event: dict[str, Any],
    ranges: list[tuple[str, int, int, int]],
) -> str | None:
    best: str | None = None
    best_span: int | None = None
    for phase, start, end, span in ranges:
        if event["ts"] < end and event["end"] > start:
            if best_span is None or span < best_span:
                best = phase
                best_span = span
    return best


def tag_phase(
    events: list[dict[str, Any]],
    iteration_nvtx: list[dict[str, Any]],
    comm_nvtx_records: list[dict[str, Any]] | None = None,
) -> None:
    """
    Set ``event['phase']`` for timeline plots.

    Inner comm NVTX records win over outer iteration ranges when both overlap.
    """
    from plotting_tools.comm_nvtx import is_conclusive_comm_record

    comm_ranges: list[tuple[str, int, int, int]] = []
    for r in comm_nvtx_records or []:
        if not is_conclusive_comm_record(r):
            continue
        phase = (r.get("phase") or "").lower().strip()
        span = int(r["end"]) - int(r["ts"])
        comm_ranges.append((phase, int(r["ts"]), int(r["end"]), span))

    iteration_ranges: list[tuple[str, int, int, int]] = []
    for r in iteration_nvtx:
        if r.get("scope") == "comm":
            continue
        n = (r.get("name") or r.get("phase") or "").lower().strip()
        span = int(r["end"]) - int(r["ts"])
        if n in ("prefill", "decode", "mixed"):
            iteration_ranges.append((n, int(r["ts"]), int(r["end"]), span))
        elif n.startswith("iter|") and r.get("phase") in ("prefill", "decode", "mixed"):
            phase = str(r["phase"])
            iteration_ranges.append((phase, int(r["ts"]), int(r["end"]), span))
        elif n == "idle":
            continue
        elif "prefill" in n or "prompt" in n or "context" in n:
            iteration_ranges.append(("prefill", int(r["ts"]), int(r["end"]), span))
        elif "decode" in n or "generation" in n:
            iteration_ranges.append(("decode", int(r["ts"]), int(r["end"]), span))

    for e in events:
        e["phase"] = "unknown"
        phase = _narrowest_overlap_phase(e, comm_ranges)
        if phase is None:
            phase = _narrowest_overlap_phase(e, iteration_ranges)
        if phase is not None:
            e["phase"] = phase


def parse_job_metadata(slurm_out: Path | None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if slurm_out is None or not slurm_out.is_file():
        return meta
    text = slurm_out.read_text(errors="replace")
    m = re.search(
        r"MODEL_ID=(\S+).*?TP=(\d+).*?PP=(\d+).*?EP=(\d+)",
        text,
    )
    if m:
        meta["model"] = m.group(1)
        meta["tensor_parallel"] = int(m.group(2))
        meta["pipeline_parallel"] = int(m.group(3))
        meta["expert_parallel"] = int(m.group(4))

    if "tensor_parallel" not in meta:
        m = re.search(r"[_-]tp(\d+)", text, re.I)
        if m:
            meta["tensor_parallel"] = int(m.group(1))
    if "pipeline_parallel" not in meta:
        m = re.search(r"[_-]pp(\d+)", text, re.I)
        if m:
            meta["pipeline_parallel"] = int(m.group(1))
    if "expert_parallel" not in meta:
        m = re.search(r"[_-]ep(\d+)", text, re.I)
        if m:
            meta["expert_parallel"] = int(m.group(1))

    m = re.search(r"Successful requests:\s+(\d+)", text)
    if m:
        meta["successful_requests"] = int(m.group(1))
    m = re.search(r"Benchmark duration \(s\):\s+([\d.]+)", text)
    if m:
        meta["benchmark_duration_s"] = float(m.group(1))
    m = re.search(r"Mean TTFT \(ms\):\s+([\d.]+)", text)
    if m:
        meta["mean_ttft_ms"] = float(m.group(1))
    m = re.search(r"Mean TPOT \(ms\):\s+([\d.]+)", text)
    if m:
        meta["mean_tpot_ms"] = float(m.group(1))
    m = re.search(r"HEAD_NODE=(\S+)", text)
    if m:
        meta["head_node"] = m.group(1)
    m = re.search(r"WORKER_NODES=(\S+)", text)
    if m:
        meta["worker_nodes"] = [n for n in m.group(1).split() if n]
    m = re.search(r"SLURM_JOB_NODELIST=(\S+)", text)
    if m:
        meta["slurm_nodelist"] = m.group(1)
    m = re.search(r"SLURM_JOB_ID=(\S+)", text)
    if m:
        meta["job_id"] = m.group(1)

    # Benchmark and scheduler parameters. These are echoed by the local
    # serving scripts before the benchmark starts.
    for env_key, meta_key, caster in (
        ("NUM_PROMPTS", "num_prompts", int),
        ("SP", "sp", int),
        ("SD", "sd", int),
        ("MAX_MODEL_LEN", "max_model_len", int),
        ("MAX_NUM_SEQS", "max_num_seqs", int),
        ("MAX_NUM_BATCHED_TOKENS", "max_num_batched_tokens", int),
        ("BURSTINESS", "burstiness", float),
        ("REQUEST_RATE", "request_rate", float),
    ):
        m = re.search(rf"\b{env_key}=([0-9.]+)\b", text)
        if m:
            meta[meta_key] = caster(m.group(1))
    m = re.search(r"--custom-output-len\s+(\d+)", text)
    if m:
        meta["custom_output_len"] = int(m.group(1))
    m = re.search(r"--ignore-eos", text)
    if m:
        meta["ignore_eos"] = True

    # Model ID from script name or env
    if "model_id" not in meta:
        m = re.search(r"MODEL_ID=(\S+)", text)
        if m:
            meta["model_id"] = m.group(1)
        else:
            m = re.search(r"--model\s+(\S+)", text)
            if m:
                meta["model_id"] = m.group(1)

    # Failed requests
    m = re.search(r"Failed requests:\s+(\d+)", text)
    if m:
        meta["failed_requests"] = int(m.group(1))
    for label, meta_key in (
        ("Request throughput \\(req/s\\)", "request_throughput_rps"),
        ("Output token throughput \\(tok/s\\)", "output_token_throughput_tps"),
        ("Peak output token throughput \\(tok/s\\)",
         "peak_output_token_throughput_tps"),
        ("Total token throughput \\(tok/s\\)", "total_token_throughput_tps"),
        ("Total input tokens", "total_input_tokens"),
        ("Total generated tokens", "total_generated_tokens"),
        ("Peak concurrent requests", "peak_concurrent_requests"),
        ("Median TTFT \\(ms\\)", "median_ttft_ms"),
        ("P99 TTFT \\(ms\\)", "p99_ttft_ms"),
        ("Median TPOT \\(ms\\)", "median_tpot_ms"),
        ("P99 TPOT \\(ms\\)", "p99_tpot_ms"),
    ):
        m = re.search(rf"{label}:\s+([\d.]+)", text)
        if m:
            val = float(m.group(1))
            meta[meta_key] = int(val) if val.is_integer() else val

    return meta


def parse_iteration_log(slurm_out: Path | None) -> list[dict[str, Any]]:
    """Parse EngineCore iteration log lines from Slurm output.

    Returns a list of dicts with keys:
        iteration, timestamp_s, context_requests, context_tokens,
        generation_requests, generation_tokens, elapsed_ms
    """
    if slurm_out is None or not slurm_out.is_file():
        return []

    pattern = re.compile(
        r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[core\.py:\d+\] "
        r"Iteration\((\d+)\): "
        r"(\d+) context requests, (\d+) context tokens, "
        r"(\d+) generation requests, (\d+) generation tokens, "
        r"iteration elapsed time: ([\d.]+) ms"
    )

    iterations: list[dict[str, Any]] = []
    t0_s: float | None = None

    for line in slurm_out.open(errors="replace"):
        m = pattern.search(line)
        if not m:
            continue
        ts_str = m.group(1)
        parts = ts_str.split()
        h, mi, s = parts[1].split(":")
        time_of_day_s = int(h) * 3600 + int(mi) * 60 + int(s)
        if t0_s is None:
            t0_s = time_of_day_s

        iterations.append({
            "iteration": int(m.group(2)),
            "timestamp_s": time_of_day_s - t0_s,
            "context_requests": int(m.group(3)),
            "context_tokens": int(m.group(4)),
            "generation_requests": int(m.group(5)),
            "generation_tokens": int(m.group(6)),
            "elapsed_ms": float(m.group(7)),
        })

    return iterations


def pp_rank_order(job_meta: dict[str, Any]) -> list[str]:
    """PP stage order: rank 0 = head node, then workers in listed order."""
    head = job_meta.get("head_node")
    workers = job_meta.get("worker_nodes") or []
    if head:
        return [head, *workers]
    return workers


def infer_node_name(trace_path: Path, *, default: str | None = None) -> str | None:
    """Extract cluster node id from trace path (e.g. htc-g059)."""
    m = re.search(r"(htc-g\d+)", trace_path.as_posix(), re.I)
    if m:
        return m.group(1).lower()
    return default


def infer_local_rank(
    trace_path: Path,
    job_meta: dict[str, Any],
    *,
    default: int = 0,
) -> int:
    """Map trace path hostname (e.g. htc-g060) to PP rank using Slurm log order."""
    node = infer_node_name(trace_path)
    if not node:
        return default
    order = [n.lower() for n in pp_rank_order(job_meta)]
    if node in order:
        return order.index(node)
    return default


def infer_device_id(events: list[dict[str, Any]], *, default: int = 0) -> int:
    """CUDA device index from CUPTI rows (TP local rank on node)."""
    devs: set[int] = set()
    for e in events:
        dev = (e.get("args") or {}).get("device_id")
        if dev is not None:
            devs.add(int(dev))
    return min(devs) if devs else default


def infer_global_rank(
    trace_path: Path,
    job_meta: dict[str, Any],
    events: list[dict[str, Any]],
) -> int:
    """Global rank = PP_rank * TP + local device id."""
    tp = max(1, int(job_meta.get("tensor_parallel", 1)))
    pp_rank = infer_local_rank(trace_path, job_meta)
    tp_rank = infer_device_id(events)
    return pp_rank * tp + tp_rank


def pp_sendrecv_duration_us(events: list[dict[str, Any]]) -> int:
    return sum(
        e["dur"]
        for e in events
        if e.get("kind") == "comm"
        and "sendrecv" in (e.get("name") or "").lower()
    )


def pp_comm_reference_per_rank_us(
    stage_totals: dict[int, int],
    *,
    tp: int,
) -> float:
    if not stage_totals:
        return 0.0
    return max(stage_totals.values()) / max(tp, 1)


def adjust_duty_with_pp_balance(
    duty: dict[str, float],
    events: list[dict[str, Any]],
    *,
    pp_reference_us_per_rank: float,
) -> dict[str, float]:
    if not events or pp_reference_us_per_rank <= 0:
        return duty
    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    span = max(t1 - t0, 1)
    totals = {k: int(v * span) for k, v in duty.items()}
    local_pp = pp_sendrecv_duration_us(events)
    supplement = max(0.0, pp_reference_us_per_rank - local_pp)
    if supplement <= 0:
        return duty
    totals["network_p2p"] = totals.get("network_p2p", 0) + int(supplement)
    return {k: v / span for k, v in totals.items()}


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[list[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def trace_t0_us(events: list[dict[str, Any]]) -> int | None:
    """Earliest event start (µs) in a trace, or None if empty."""
    if not events:
        return None
    return min(e["ts"] for e in events)


def global_align_t0(event_lists: list[list[dict[str, Any]]]) -> int | None:
    """Shared timeline origin: min start time across all traces."""
    starts = [t for ev in event_lists if ev for t in [trace_t0_us(ev)] if t is not None]
    return min(starts) if starts else None


def sync_capture_t0(event_lists: list[list[dict[str, Any]]]) -> int | None:
    """
    Plot origin when every worker capture has started (max local t0).

    Removes leading empty time on early-start nodes (e.g. g059) relative to
    nodes whose Nsight session began later.
    """
    starts = [t for ev in event_lists if ev for t in [trace_t0_us(ev)] if t is not None]
    return max(starts) if starts else None


def clock_offset_ms(
    events: list[dict[str, Any]],
    *,
    time_origin_us: int,
) -> float | None:
    """Milliseconds from global origin to this trace's first event."""
    local = trace_t0_us(events)
    if local is None:
        return None
    return (local - time_origin_us) / 1000.0


def infer_active_segments_us(
    events: list[dict[str, Any]],
    *,
    density_threshold: int = 50,
    bin_size_us: int = 1_000_000,
    margin_bins: int = 1,
    max_gap_bins: int = 1,
) -> list[tuple[int, int]]:
    """Infer one or more active inference segments from event density.

    Uses network_collective events as the primary signal for "model is actively
    doing inference" since TP all-reduce only fires during forward passes.
    Falls back to compute density if no collectives are found.

    Returns a list of (start_us, end_us) dense inference islands. Keeping
    multiple segments is important when traces contain benchmark gaps between
    bursts, startup, shutdown, or Nsight finalization artifacts.
    """
    nccl_events = [
        e for e in events
        if e.get("sub") in ("network_collective", "network_p2p")
    ]

    anchor_events = nccl_events
    threshold = density_threshold
    if not nccl_events:
        anchor_events = [e for e in events if e.get("kind") == "compute"]
        threshold = max(density_threshold, 100)
    if not anchor_events:
        return []

    t0 = min(e["ts"] for e in events)

    bins: dict[int, int] = {}
    for e in anchor_events:
        b = (e["ts"] - t0) // bin_size_us
        bins[b] = bins.get(b, 0) + 1

    active_bins = [b for b, cnt in bins.items() if cnt >= threshold]
    if not active_bins:
        return []

    segments: list[tuple[int, int]] = []
    sorted_bins = sorted(active_bins)
    seg_start = sorted_bins[0]
    prev = sorted_bins[0]
    for b in sorted_bins[1:]:
        if b - prev <= max_gap_bins + 1:
            prev = b
            continue
        first_bin = max(0, seg_start - margin_bins)
        last_bin = prev + margin_bins
        segments.append((
            t0 + first_bin * bin_size_us,
            t0 + (last_bin + 1) * bin_size_us,
        ))
        seg_start = b
        prev = b

    first_bin = max(0, seg_start - margin_bins)
    last_bin = prev + margin_bins
    segments.append((
        t0 + first_bin * bin_size_us,
        t0 + (last_bin + 1) * bin_size_us,
    ))
    return segments


def infer_active_window_us(
    events: list[dict[str, Any]],
    *,
    density_threshold: int = 50,
    bin_size_us: int = 1_000_000,
    margin_bins: int = 1,
) -> tuple[int, int] | None:
    """Infer the outer active inference window from event density.

    This compatibility helper returns the envelope around all detected active
    segments. Prefer infer_active_segments_us for plots and quality gates.
    """
    segments = infer_active_segments_us(
        events,
        density_threshold=density_threshold,
        bin_size_us=bin_size_us,
        margin_bins=margin_bins,
    )
    if not segments:
        return None
    return (min(s for s, _ in segments), max(e for _, e in segments))


def trim_events_to_window(
    events: list[dict[str, Any]],
    window_us: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """Keep only events that overlap with the given (start_us, end_us) window."""
    if window_us is None:
        return list(events)
    start, end = window_us
    return [e for e in events if e["end"] > start and e["ts"] < end]


def trim_events_to_windows(
    events: list[dict[str, Any]],
    windows_us: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    """Keep events that overlap any active segment."""
    if not windows_us:
        return list(events)
    return [
        e for e in events
        if any(
            e["end"] > start and e["ts"] < end
            for start, end in windows_us
        )
    ]


def duty_by_sub(
    events: list[dict[str, Any]],
    *,
    window_us: tuple[int, int] | None = None,
) -> dict[str, float]:
    if not events:
        return {}
    if window_us is not None:
        span = max(window_us[1] - window_us[0], 1)
    else:
        t0 = min(e["ts"] for e in events)
        t1 = max(e["end"] for e in events)
        span = max(t1 - t0, 1)
    totals: dict[str, int] = {}
    for e in events:
        dur = e["dur"]
        if window_us is not None:
            start, end = window_us
            dur = max(0, min(e["end"], end) - max(e["ts"], start))
        totals[e["sub"]] = totals.get(e["sub"], 0) + dur
    return {k: v / span for k, v in totals.items()}


def duty_by_sub_windows(
    events: list[dict[str, Any]],
    windows_us: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> dict[str, float]:
    """Duty by subcategory over concatenated active windows."""
    if not events or not windows_us:
        return duty_by_sub(events)
    span = max(sum(end - start for start, end in windows_us), 1)
    totals: dict[str, int] = {}
    for e in events:
        dur = 0
        for start, end in windows_us:
            dur += max(0, min(e["end"], end) - max(e["ts"], start))
        if dur:
            totals[e["sub"]] = totals.get(e["sub"], 0) + dur
    return {k: v / span for k, v in totals.items()}
