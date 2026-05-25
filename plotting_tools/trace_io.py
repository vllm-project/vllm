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
    return meta


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


def clock_offset_ms(events: list[dict[str, Any]], *, time_origin_us: int) -> float | None:
    """Milliseconds from global origin to this trace's first event."""
    local = trace_t0_us(events)
    if local is None:
        return None
    return (local - time_origin_us) / 1000.0


def duty_by_sub(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {}
    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    span = max(t1 - t0, 1)
    totals: dict[str, int] = {}
    for e in events:
        totals[e["sub"]] = totals.get(e["sub"], 0) + e["dur"]
    return {k: v / span for k, v in totals.items()}
