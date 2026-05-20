"""Matplotlib plots for vLLM / MoE trace analysis."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from plotting_tools.classify import classify_comm_operation, comm_operation_label
from plotting_tools.trace_io import duty_by_sub, merge_intervals

PLOT_EXT = ".pdf"


def save_figure(fig, out_path: Path, *, bbox_tight: bool = True) -> None:
    """Write matplotlib figure (PDF by default) and close it."""
    out_path = Path(out_path)
    fmt = out_path.suffix.lstrip(".") or "pdf"
    kwargs: dict[str, Any] = {"dpi": 200, "format": fmt}
    if bbox_tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(out_path, **kwargs)
    plt.close(fig)


# White (no traffic) -> green -> dark blue (heavy traffic)
TRAFFIC_CMAP = LinearSegmentedColormap.from_list(
    "traffic_wgb",
    ["#ffffff", "#41ab5d", "#08306b"],
)

DECOMPOSED_LAYERS = (
    ("attention_comp", "Attention Comp", "#4c78a8"),
    ("gate_comp", "Gate Comp", "#f58518"),
    ("experts_comp", "Experts Comp", "#54a24b"),
    ("add_norm_comp", "Add and Norm Comp", "#b279a2"),
    ("other_compute", "Other Compute", "#9ecae9"),
    ("collective_comm", "Collective Comm", "#e45756"),
    ("control", "Control", "#bab0ac"),
)

DECOMPOSED_LAYERS_PP_SPLIT = (
    ("attention_comp", "Attention Comp", "#4c78a8"),
    ("gate_comp", "Gate Comp", "#f58518"),
    ("experts_comp", "Experts Comp", "#54a24b"),
    ("add_norm_comp", "Add and Norm Comp", "#b279a2"),
    ("other_compute", "Other Compute", "#9ecae9"),
    ("pp_comm", "PP Comm (SendRecv)", "#d62728"),
    ("local_comm", "Local Comm (memcpy)", "#ff9896"),
    ("control", "Control", "#bab0ac"),
)

SUBCATEGORY_LABELS = {k: label for k, label, _ in DECOMPOSED_LAYERS}
SUBCATEGORY_LABELS.update(
    {k: label for k, label, _ in DECOMPOSED_LAYERS_PP_SPLIT}
)


def _plot_t0(events: list[dict[str, Any]], *, time_origin_us: int | None = None) -> int:
    if time_origin_us is not None:
        return time_origin_us
    return min(e["ts"] for e in events)


def _timeline_interval_ms(
    e: dict[str, Any],
    t0_us: int,
) -> tuple[float, float] | None:
    """
    (start_ms, width_ms) relative to t0_us, clipped so start >= 0.

    Events entirely before t0_us are omitted (synced multi-node plots).
    """
    start_us = int(e["ts"])
    end_us = int(e.get("end", start_us + int(e["dur"])))
    if end_us <= t0_us:
        return None
    clip_start = max(start_us, t0_us)
    return ((clip_start - t0_us) / 1000.0, (end_us - clip_start) / 1000.0)


def _decomposed_plot_sub(e: dict[str, Any], *, pp_comm_split: bool) -> str | None:
    sub = e.get("sub")
    if not pp_comm_split:
        return sub
    if sub != "collective_comm":
        return sub
    op = classify_comm_operation(e.get("name", ""), e.get("cat", ""))
    if op == "point_to_point":
        return "pp_comm"
    return "local_comm"


def _decomposed_layers(pp_comm_split: bool) -> tuple[tuple[str, str, str], ...]:
    return DECOMPOSED_LAYERS_PP_SPLIT if pp_comm_split else DECOMPOSED_LAYERS


def _ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_decomposed_timeline(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Decomposed timeline",
    max_ms: float | None = None,
    time_origin_us: int | None = None,
    pp_comm_split: bool = False,
) -> None:
    """Horizontal stacked bands: attention / gate / experts / norm / comm / control."""
    if not events:
        return
    layers = _decomposed_layers(pp_comm_split)
    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    fig, ax = plt.subplots(figsize=(18, 5))
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}

    for sub, _label, color in layers:
        intervals: list[tuple[float, float]] = []
        for e in events:
            if _decomposed_plot_sub(e, pp_comm_split=pp_comm_split) != sub:
                continue
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            continue
        y = y_positions[sub]
        ax.broken_barh(intervals, (y, 0.85), facecolors=color, edgecolors="none")

    ax.set_yticks([i + 0.4 for i in range(len(layers))])
    ax.set_yticklabels([spec[1] for spec in layers])
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    if max_ms is not None:
        ax.set_xlim(0, max_ms)
    else:
        ax.set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, out_path)


def _gpu_active_intervals(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (e["ts"], e["end"])
        for e in events
        if e["kind"] == "compute"
        or (e["kind"] == "comm" and "memcpy" in e.get("name", "").lower())
        or (e["kind"] == "comm" and "nccl" in e.get("name", "").lower())
    ]


def gpu_idle_windows_ms(events: list[dict[str, Any]]) -> list[float]:
    if not events:
        return []
    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    active = merge_intervals(_gpu_active_intervals(events))
    gaps: list[float] = []
    cur = t0
    for s, e in active:
        if cur < s:
            gaps.append((s - cur) / 1000.0)
        cur = max(cur, e)
    if cur < t1:
        gaps.append((t1 - cur) / 1000.0)
    return [g for g in gaps if g > 0]


def gpu_idle_fraction(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    span = max(t1 - t0, 1)
    active = merge_intervals(_gpu_active_intervals(events))
    busy = sum(e - s for s, e in active)
    return max(0.0, (span - busy) / span)


def _short_name(name: str, width: int = 48) -> str:
    name = name.strip()
    if len(name) <= width:
        return name
    return name[: width - 3] + "..."


def _dominant_sub_in_window(
    events: list[dict[str, Any]],
    start: int,
    end: int,
) -> tuple[str, str, str]:
    """Return (kind, sub, short_name) with largest overlap duration in [start, end]."""
    totals: dict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for e in events:
        overlap = min(e["end"], end) - max(e["ts"], start)
        if overlap <= 0:
            continue
        sub = e.get("sub", e["kind"])
        totals[sub] += overlap
        names[sub] = _short_name(e.get("name", sub))
        kinds[sub] = e["kind"]
    if not totals:
        return ("none", "none", "none")
    sub = max(totals, key=totals.get)
    return (kinds[sub], sub, names[sub])


def analyze_idle_gaps(
    events: list[dict[str, Any]],
    *,
    min_gap_us: int = 10,
) -> list[dict[str, Any]]:
    """
    For each GPU-idle gap, record dominant activity in the busy slab before/after.

    Busy = GPU kernels + device memcpy/NCCL (same as gpu_idle_windows_ms).
  """
    if not events:
        return []

    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    busy = merge_intervals(_gpu_active_intervals(events))
    gaps: list[dict[str, Any]] = []

    def record(gap_start: int, gap_end: int, before_win: tuple[int, int], after_win: tuple[int, int]):
        gap_us = gap_end - gap_start
        if gap_us < min_gap_us:
            return
        bk, bs, bn = _dominant_sub_in_window(events, *before_win)
        ak, a_sub, an = _dominant_sub_in_window(events, *after_win)
        gaps.append(
            {
                "gap_start_us": gap_start,
                "gap_end_us": gap_end,
                "gap_ms": gap_us / 1000.0,
                "before_kind": bk,
                "before_sub": bs,
                "before_label": SUBCATEGORY_LABELS.get(bs, bs),
                "before_name": bn,
                "after_kind": ak,
                "after_sub": a_sub,
                "after_label": SUBCATEGORY_LABELS.get(a_sub, a_sub),
                "after_name": an,
                "transition": f"{SUBCATEGORY_LABELS.get(bs, bs)} → idle → {SUBCATEGORY_LABELS.get(a_sub, a_sub)}",
            }
        )

    if not busy:
        record(t0, t1, (t0, t1), (t0, t1))
        return gaps

    if busy[0][0] > t0:
        record(t0, busy[0][0], (t0, t0), busy[0])

    for i in range(len(busy) - 1):
        gap_start, gap_end = busy[i][1], busy[i + 1][0]
        record(gap_start, gap_end, busy[i], busy[i + 1])

    if busy[-1][1] < t1:
        record(busy[-1][1], t1, busy[-1], (t1, t1))

    return gaps


def summarize_idle_transitions(
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    by_transition: Counter[str] = Counter()
    time_by_transition: dict[str, float] = defaultdict(float)
    by_before: Counter[str] = Counter()
    by_after: Counter[str] = Counter()
    time_by_before: dict[str, float] = defaultdict(float)
    time_by_after: dict[str, float] = defaultdict(float)

    for g in gaps:
        tr = g["transition"]
        ms = g["gap_ms"]
        by_transition[tr] += 1
        time_by_transition[tr] += ms
        by_before[g["before_label"]] += 1
        time_by_before[g["before_label"]] += ms
        by_after[g["after_label"]] += 1
        time_by_after[g["after_label"]] += ms

    longest = sorted(gaps, key=lambda g: g["gap_ms"], reverse=True)[:30]
    return {
        "gap_count": len(gaps),
        "top_transitions_by_count": by_transition.most_common(20),
        "top_transitions_by_ms": sorted(
            time_by_transition.items(), key=lambda x: x[1], reverse=True
        )[:20],
        "idle_ms_after": sorted(time_by_before.items(), key=lambda x: x[1], reverse=True)[:15],
        "idle_ms_before": sorted(time_by_after.items(), key=lambda x: x[1], reverse=True)[:15],
        "longest_gaps": longest,
    }


def plot_idle_transition_bars(
    gaps: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    top_n: int = 12,
) -> None:
    """Bar chart: total idle ms per (busy_before → busy_after) transition."""
    time_by_tr: dict[str, float] = defaultdict(float)
    count_by_tr: dict[str, int] = defaultdict(int)
    for g in gaps:
        time_by_tr[g["transition"]] += g["gap_ms"]
        count_by_tr[g["transition"]] += 1

    items = sorted(time_by_tr.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        return

    labels = [k for k, _ in items]
    ms_vals = [v for _, v in items]
    counts = [count_by_tr[k] for k in labels]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(labels))))
    y = np.arange(len(labels))
    bars = ax.barh(y, ms_vals, color="#9ecae9")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Total idle time in gaps (ms)")
    ax.set_title(f"{title}\n(dominant GPU activity slab before idle → after)")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  n={c}",
            va="center",
            fontsize=8,
        )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_idle_transition_heatmap(
    gaps: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Heatmap: rows = activity before idle, cols = activity after idle (total idle ms)."""
    matrix: dict[tuple[str, str], float] = defaultdict(float)
    rows: set[str] = set()
    cols: set[str] = set()
    for g in gaps:
        b, a = g["before_label"], g["after_label"]
        rows.add(b)
        cols.add(a)
        matrix[(b, a)] += g["gap_ms"]

    row_labels = sorted(rows)
    col_labels = sorted(cols)
    if not row_labels or not col_labels:
        return

    data = np.zeros((len(row_labels), len(col_labels)))
    for i, b in enumerate(row_labels):
        for j, a in enumerate(col_labels):
            data[i, j] = matrix.get((b, a), 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.9), max(6, len(row_labels) * 0.5)))
    vmax = data.max() if data.size else 1.0
    im = ax.imshow(data, cmap=TRAFFIC_CMAP, vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("GPU activity after idle (dominant in next busy slab)")
    ax.set_ylabel("GPU activity before idle (dominant in prior busy slab)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Total idle time (ms)")
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_idle_context(
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    prefix: str,
    min_gap_us: int = 1000,
) -> dict[str, Any]:
    """Analyze and plot idle gap context; write idle_gaps.json."""
    gaps = analyze_idle_gaps(events, min_gap_us=min_gap_us)
    summary = summarize_idle_transitions(gaps)

    plot_idle_transition_bars(
        gaps,
        out_dir / f"idle_transitions_by_time{PLOT_EXT}",
        title=f"{prefix}: idle gaps by before→after activity",
    )
    plot_idle_transition_heatmap(
        gaps,
        out_dir / f"idle_transition_heatmap{PLOT_EXT}",
        title=f"{prefix}: idle time (ms) by before/after activity",
    )

    write_summary_json(
        out_dir / "idle_gaps.json",
        {
            "summary": summary,
            "gaps_sample": summary.get("longest_gaps", []),
            "note": "Full per-gap list omitted if >500 gaps; see longest_gaps in summary.",
        },
    )
    return summary


def plot_traffic_volume_pct(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    parallel_label: str,
) -> None:
    """Bar chart of duty-cycle % per decomposed category (may exceed 100% if events overlap)."""
    duty = duty_by_sub(events)
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    for sub, label, color in DECOMPOSED_LAYERS:
        v = duty.get(sub, 0.0) * 100.0
        if v <= 0:
            continue
        labels.append(label)
        vals.append(v)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Summed duration / trace span (%)")
    ax.set_title(f"{title}\n{parallel_label}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, out_path)


def _extract_comm_bytes(event: dict[str, Any]) -> int | None:
    args = event.get("args") or {}
    for key in ("bytes", "size", "nbytes", "Byte", "copy bytes"):
        if key in args and isinstance(args[key], (int, float)):
            return int(args[key])
    name = event.get("name", "")
    m = re.search(r"(\d+)\s*b", name.lower())
    if m:
        return int(m.group(1))
    return None


def comm_message_stats(
    events: list[dict[str, Any]],
    phase: str | None = None,
) -> dict[str, Any]:
    comms = [
        e
        for e in events
        if e["kind"] == "comm" and (phase is None or e.get("phase", "unknown") == phase)
    ]
    sizes: list[int] = []
    for e in comms:
        b = _extract_comm_bytes(e)
        if b is not None and b > 0:
            sizes.append(b)
    return {
        "count": len(comms),
        "sizes_bytes": sizes,
        "avg_bytes": float(np.mean(sizes)) if sizes else 0.0,
        "total_bytes": int(sum(sizes)),
    }


def plot_message_stats(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    """Prefill vs decode comm count and average message size."""
    stats = {
        "prefill": comm_message_stats(events, "prefill"),
        "decode": comm_message_stats(events, "decode"),
    }
    all_s = comm_message_stats(events)

    phases = ["prefill", "decode", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = [
        stats["prefill"]["count"],
        stats["decode"]["count"],
        all_s["count"],
    ]
    avgs_mb = [
        stats["prefill"]["avg_bytes"] / 1e6,
        stats["decode"]["avg_bytes"] / 1e6,
        all_s["avg_bytes"] / 1e6,
    ]

    axes[0].bar(phases, counts, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_ylabel("Event count")
    axes[0].set_title(f"{prefix}: comm event count")

    axes[1].bar(phases, avgs_mb, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_ylabel("Avg size (MB)")
    axes[1].set_title(f"{prefix}: avg comm message size")

    plt.tight_layout()
    save_figure(fig, out_path)
    return {"prefill": stats["prefill"], "decode": stats["decode"], "all": all_s}


def _peer_from_nccl_name(name: str) -> int | None:
    m = re.search(r"rank\s*[=:]?\s*(\d+)", name.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"peer\s*[=:]?\s*(\d+)", name.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"root\s*[=:]?\s*(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def _peer_device_from_name(name: str) -> int | None:
    m = re.search(r"device\s*[=:]?\s*(\d+)", name.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"gpu\s*[=:]?\s*(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def _comm_volume(event: dict[str, Any]) -> float:
    b = _extract_comm_bytes(event)
    if b is not None and b > 0:
        return float(b)
    return float(event["dur"])


def _pp_neighbor_ranks(local_rank: int, n_ranks: int) -> list[int]:
    out: list[int] = []
    if local_rank > 0:
        out.append(local_rank - 1)
    if local_rank < n_ranks - 1:
        out.append(local_rank + 1)
    return out


def _infer_pp_peer(local_rank: int, n_ranks: int, name: str) -> int | None:
    """Pipeline-parallel fallback when NCCL name has no peer rank."""
    peer = _peer_from_nccl_name(name)
    if peer is not None and 0 <= peer < n_ranks:
        return peer
    lower = name.lower()
    if n_ranks == 2:
        return 1 - local_rank
    neighbors = _pp_neighbor_ranks(local_rank, n_ranks)
    if not neighbors:
        return None
    if "sendrecv" in lower or "send" in lower or "recv" in lower:
        if "send" in lower and "recv" not in lower and local_rank < n_ranks - 1:
            return local_rank + 1
        if "recv" in lower and "send" not in lower and local_rank > 0:
            return local_rank - 1
    if len(neighbors) == 1:
        return neighbors[0]
    return neighbors[0]


def build_traffic_matrix(
    events: list[dict[str, Any]],
    n_ranks: int,
    *,
    local_rank: int = 0,
    pp_mode: bool = True,
) -> np.ndarray:
    """Rank-to-rank comm volume (bytes, else µs) from this trace's perspective."""
    mat = np.zeros((n_ranks, n_ranks), dtype=np.float64)
    for e in events:
        if e["kind"] != "comm":
            continue
        vol = _comm_volume(e)
        name = e.get("name", "")
        peer = _peer_from_nccl_name(name)
        if peer is None and pp_mode:
            peer = _infer_pp_peer(local_rank, n_ranks, name)
        if peer is None or peer < 0 or peer >= n_ranks:
            continue
        if peer == local_rank:
            continue
        mat[local_rank, peer] += vol
    return mat


def merge_rank_traffic_matrices(
    matrices: list[np.ndarray],
) -> np.ndarray:
    """Sum per-rank trace matrices into one job-level rank x rank view."""
    if not matrices:
        return np.zeros((1, 1), dtype=np.float64)
    n = max(m.shape[0] for m in matrices)
    out = np.zeros((n, n), dtype=np.float64)
    for m in matrices:
        out[: m.shape[0], : m.shape[1]] += m
    return out


def build_gpu_traffic_matrix(
    events: list[dict[str, Any]],
    *,
    n_gpus: int | None = None,
) -> np.ndarray:
    """
    GPU-to-GPU comm on this process: diagonal = local collectives;
    off-diagonal = P2P when peer GPU is known, else split across other GPUs.
    """
    devices: set[int] = set()
    for e in events:
        args = e.get("args") or {}
        if "device_id" in args:
            devices.add(int(args["device_id"]))
    if n_gpus is None:
        n_gpus = (max(devices) + 1) if devices else 1
    n_gpus = max(n_gpus, 1)
    mat = np.zeros((n_gpus, n_gpus), dtype=np.float64)

    for e in events:
        if e["kind"] != "comm":
            continue
        args = e.get("args") or {}
        src = int(args.get("device_id", 0))
        if src >= n_gpus:
            continue
        vol = _comm_volume(e)
        name = e.get("name", "")
        peer_dev = _peer_device_from_name(name)
        lower = name.lower()
        if peer_dev is not None and 0 <= peer_dev < n_gpus and peer_dev != src:
            mat[src, peer_dev] += vol
        elif any(k in lower for k in ("allreduce", "allgather", "reducescatter", "broadcast")):
            for dst in range(n_gpus):
                if dst != src:
                    mat[src, dst] += vol / max(n_gpus - 1, 1)
        elif "sendrecv" in lower or "send" in lower or "recv" in lower:
            for dst in range(n_gpus):
                if dst != src:
                    mat[src, dst] += vol / max(n_gpus - 1, 1)
        else:
            mat[src, src] += vol
    return mat


def plot_traffic_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    *,
    title: str,
    xlabel: str = "Destination rank",
    ylabel: str = "Source rank",
    colorbar_label: str = "Volume (bytes or µs proxy)",
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.9), max(5, matrix.shape[0] * 0.8)))
    vmax = float(matrix.max()) if matrix.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(matrix, cmap=TRAFFIC_CMAP, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    n0, n1 = matrix.shape
    ax.set_xticks(range(n1))
    ax.set_yticks(range(n0))
    ax.set_xticklabels([str(i) for i in range(n1)])
    ax.set_yticklabels([str(i) for i in range(n0)])
    if annotate:
        for i in range(n0):
            for j in range(n1):
                val = matrix[i, j]
                if val > 0:
                    txt = f"{val:.2e}" if val >= 1e4 else f"{val:.1f}"
                    color = "white" if val > 0.6 * vmax else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_expert_traffic_volume(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    num_experts: int = 128,
) -> float:
    """Estimate expert routing traffic from gate/MoE-related comm + compute duty."""
    expert_bytes = 0
    gate_dur = 0
    for e in events:
        s = f"{e['name']} {e['cat']}".lower()
        if e["sub"] == "gate_comp":
            gate_dur += e["dur"]
        if e["kind"] == "comm" and any(
            k in s for k in ("expert", "moe", "dispatch", "combine", "all2all", "all_to_all")
        ):
            expert_bytes += _extract_comm_bytes(e) or 0
    # Heuristic: gate duration correlates with routing metadata volume
    if expert_bytes == 0 and gate_dur > 0:
        expert_bytes = int(gate_dur * 100)  # rough µs->byte proxy for plotting
    total_b = expert_bytes / 1e9

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Expert routing (est.)"], [total_b], color="#54a24b")
    ax.set_ylabel("Volume (GB)")
    ax.set_title("Expert traffic (routing / MoE comm heuristic)")
    plt.tight_layout()
    save_figure(fig, out_path)
    return total_b


def comm_operation_breakdown(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Count and total duration (µs) per comm operation type."""
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "dur_us": 0, "bytes": 0}
    )
    for e in events:
        if e.get("kind") != "comm":
            continue
        op = classify_comm_operation(e.get("name", ""), e.get("cat", ""))
        if op is None:
            op = "other_comm"
        stats[op]["count"] += 1
        stats[op]["dur_us"] += e["dur"]
        b = _extract_comm_bytes(e)
        if b:
            stats[op]["bytes"] += b
    return dict(stats)


def plot_collective_ops_breakdown(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> dict[str, dict[str, float | int]]:
    """Bar chart: comm op type vs event count and vs total time (ms)."""
    stats = comm_operation_breakdown(events)
    plot_collective_ops_breakdown_stats(stats, out_path, title=title)
    return stats


def collect_nccl_ops(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for e in events:
        if e["kind"] != "comm":
            continue
        name = e.get("name", "")
        if "nccl" not in name.lower():
            continue
        shape = None
        args = e.get("args") or {}
        for k in ("shape", "dims", "tensor_shape"):
            if k in args:
                shape = args[k]
        ops.append(
            {
                "ts_ms": (e["ts"] - _plot_t0(events)) / 1000.0,
                "name": name,
                "dur_us": e["dur"],
                "bytes": _extract_comm_bytes(e),
                "shape": shape,
            }
        )
    return ops


def write_collective_ops_table(
    ops: list[dict[str, Any]],
    out_path: Path,
) -> None:
    with out_path.open("w") as f:
        json.dump(ops, f, indent=2)


def plot_window_cdf(
    vals: list[float],
    out_path: Path,
    *,
    xlabel: str,
    title: str,
) -> None:
    if not vals:
        return
    vals = sorted(vals)
    n = len(vals)
    cdf = [(i + 1) / n for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, cdf)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    save_figure(fig, out_path)


def nocomm_windows(events: list[dict[str, Any]]) -> list[float]:
    comms = [(e["ts"], e["end"]) for e in events if e["kind"] == "comm"]
    if not comms:
        return []
    comms = merge_intervals(comms)
    start = min(e["ts"] for e in events)
    end = max(e["end"] for e in events)
    windows: list[tuple[int, int]] = []
    cur = start
    for s, e in comms:
        if cur < s:
            windows.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        windows.append((cur, end))
    merged = merge_intervals(windows)
    return [(e - s) / 1000.0 for s, e in merged if e > s]


def comm_delta(events: list[dict[str, Any]]) -> list[float]:
    comms = sorted([e for e in events if e["kind"] == "comm"], key=lambda e: e["ts"])
    if len(comms) < 2:
        return []
    return [(comms[i + 1]["ts"] - comms[i]["ts"]) / 1000.0 for i in range(len(comms) - 1)]


def merge_comm_operation_breakdowns(
    breakdowns: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "dur_us": 0, "bytes": 0}
    )
    for stats in breakdowns:
        for op, vals in stats.items():
            merged[op]["count"] += vals.get("count", 0)
            merged[op]["dur_us"] += vals.get("dur_us", 0)
            merged[op]["bytes"] += vals.get("bytes", 0)
    return dict(merged)


def plot_collective_ops_breakdown_stats(
    stats: dict[str, dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Bar chart from pre-aggregated comm_operation_breakdown stats."""
    if not stats:
        return
    ops = sorted(stats.keys(), key=lambda k: stats[k]["dur_us"], reverse=True)
    labels = [comm_operation_label(o) for o in ops]
    counts = [stats[o]["count"] for o in ops]
    dur_ms = [stats[o]["dur_us"] / 1000.0 for o in ops]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ops))
    axes[0].bar(x, counts, color="#e45756")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Count by comm op")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, dur_ms, color="#4c78a8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Total time (ms)")
    axes[1].set_title("GPU time by comm op")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_duty_by_node(
    node_duty: dict[str, dict[str, float]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Grouped bar chart: category duty % per node."""
    if not node_duty:
        return
    nodes = sorted(node_duty)
    labels: list[str] = []
    for sub, label, _ in DECOMPOSED_LAYERS:
        if any(node_duty[n].get(sub, 0) > 0 for n in nodes):
            labels.append(label)
    if not labels:
        return

    sub_by_label = {label: sub for sub, label, _ in DECOMPOSED_LAYERS}
    x = np.arange(len(labels))
    width = 0.8 / max(len(nodes), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, node in enumerate(nodes):
        vals = [node_duty[node].get(sub_by_label[label], 0) * 100 for label in labels]
        offset = (i - (len(nodes) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=node)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=7, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Summed duration / trace span (%)")
    ax.set_title(title)
    ax.legend(title="Node")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_average_duty_pct(
    node_duty: dict[str, dict[str, float]],
    out_path: Path,
    *,
    title: str,
    parallel_label: str,
) -> None:
    """Mean category duty % across nodes."""
    if not node_duty:
        return
    nodes = list(node_duty)
    avg: dict[str, float] = {}
    for sub, label, _ in DECOMPOSED_LAYERS:
        vals = [node_duty[n].get(sub, 0) for n in nodes]
        if vals:
            avg[sub] = sum(vals) / len(vals)

    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    for sub, label, color in DECOMPOSED_LAYERS:
        v = avg.get(sub, 0) * 100
        if v <= 0:
            continue
        labels.append(label)
        vals.append(v)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean duty across nodes (%)")
    ax.set_title(f"{title}\n{parallel_label}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_multi_window_cdf(
    series: dict[str, list[float]],
    out_path: Path,
    *,
    xlabel: str,
    title: str,
) -> None:
    """Overlay CDFs for multiple nodes (or pooled if one key 'all')."""
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, vals in sorted(series.items()):
        if not vals:
            continue
        vals = sorted(vals)
        n = len(vals)
        cdf = [(i + 1) / n for i in range(n)]
        ax.plot(vals, cdf, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_expert_traffic_by_node(
    node_gb: dict[str, float],
    out_path: Path,
    *,
    title: str,
) -> None:
    if not node_gb:
        return
    nodes = sorted(node_gb)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(nodes, [node_gb[n] for n in nodes], color="#54a24b")
    ax.set_ylabel("Volume (GB)")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_multi_node_decomposed(
    node_events: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    title: str,
    max_ms: float | None = None,
    time_origin_us: int,
    pp_comm_split: bool = False,
) -> None:
    """Stacked decomposed timelines with a shared x-axis (global alignment)."""
    nodes = sorted(node_events)
    if not nodes:
        return
    layers = _decomposed_layers(pp_comm_split)
    fig, axes = plt.subplots(len(nodes), 1, sharex=True, figsize=(18, 2.8 * len(nodes)))
    if len(nodes) == 1:
        axes = [axes]
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}
    for ax, node in zip(axes, nodes):
        events = node_events[node]
        for sub, _label, color in layers:
            intervals: list[tuple[float, float]] = []
            for e in events:
                if _decomposed_plot_sub(e, pp_comm_split=pp_comm_split) != sub:
                    continue
                iv = _timeline_interval_ms(e, time_origin_us)
                if iv is not None:
                    intervals.append(iv)
            if not intervals:
                continue
            y = y_positions[sub]
            ax.broken_barh(intervals, (y, 0.85), facecolors=color, edgecolors="none")
        ax.set_yticks([i + 0.4 for i in range(len(layers))])
        ax.set_yticklabels([spec[1] for spec in layers], fontsize=8)
        ax.set_ylabel(node, rotation=0, ha="right", va="center")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Time (ms, global)")
    fig.suptitle(title)
    if max_ms is not None:
        axes[-1].set_xlim(0, max_ms)
    else:
        axes[-1].set_xlim(left=0)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_classic_timeline(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    time_origin_us: int | None = None,
) -> None:
    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    bands = {
        "compute": (2.0, "tab:blue"),
        "comm": (1.0, "tab:orange"),
        "control": (0.0, "tab:green"),
    }
    fig, ax = plt.subplots(figsize=(16, 4))
    for kind, (y, color) in bands.items():
        intervals: list[tuple[float, float]] = []
        for e in events:
            if e["kind"] != kind:
                continue
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                intervals.append(iv)
        if intervals:
            ax.broken_barh(intervals, (y, 0.7), facecolors=color, label=kind)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([0.35, 1.35, 2.35])
    ax.set_yticklabels(["control", "communication", "compute"])
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    save_figure(fig, out_path)


def write_summary_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
