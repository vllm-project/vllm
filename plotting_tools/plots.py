"""Matplotlib plots for vLLM / MoE trace analysis."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from plotting_tools.classify import (
    FABRIC_COMM_OPS,
    MOVEMENT_OPS,
    classify_comm_operation,
    comm_operation_label,
    is_fabric_comm_op,
    is_fabric_event,
    is_movement_op,
)
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

HEATMAP_BG = "#FAF9F6"
HEATMAP_NODE_HIGHLIGHT = "#FFF9E0"
TRAFFIC_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "traffic_navy_cream",
    ["#ffffcc", "#a1dab4", "#41b6c4", "#225ea8", "#08306b"],
)

DECOMPOSED_LAYERS = (
    ("attention_comp", "Attention Comp", "#4c78a8"),
    ("gate_comp", "Gate Comp", "#f58518"),
    ("moe_expert", "MoE Expert Comp", "#54a24b"),
    ("moe_routing", "MoE Routing Comp", "#2ca02c"),
    ("matmul_gemm", "Matmul/GEMM Comp", "#17becf"),
    ("kv_cache_write", "KV Cache Write", "#6baed6"),
    ("rotary_embedding", "Rotary Embedding", "#bcbddc"),
    ("add_norm_comp", "Add and Norm Comp", "#b279a2"),
    ("sampling_overhead", "Sampling Overhead", "#fdb462"),
    ("masking_indexing", "Masking / Indexing", "#8c6d31"),
    ("other_compute", "Other Compute", "#9ecae9"),
    ("network_collective", "Network Collective", "#e45756"),
    ("network_p2p", "Network P2P", "#d62728"),
    ("device_copy", "Device Memcpy", "#ff9896"),
    ("host_transfer", "Host Memcpy", "#c5b0d5"),
    ("control", "Control", "#bab0ac"),
)

DECOMPOSED_LAYERS_PP_SPLIT = (
    ("attention_comp", "Attention Comp", "#4c78a8"),
    ("gate_comp", "Gate Comp", "#f58518"),
    ("moe_expert", "MoE Expert Comp", "#54a24b"),
    ("moe_routing", "MoE Routing Comp", "#2ca02c"),
    ("matmul_gemm", "Matmul/GEMM Comp", "#17becf"),
    ("kv_cache_write", "KV Cache Write", "#6baed6"),
    ("rotary_embedding", "Rotary Embedding", "#bcbddc"),
    ("add_norm_comp", "Add and Norm Comp", "#b279a2"),
    ("sampling_overhead", "Sampling Overhead", "#fdb462"),
    ("masking_indexing", "Masking / Indexing", "#8c6d31"),
    ("other_compute", "Other Compute", "#9ecae9"),
    ("pp_comm", "PP Comm (SendRecv)", "#d62728"),
    ("network_collective", "TP Collective", "#e45756"),
    ("device_copy", "Device Memcpy", "#ff9896"),
    ("host_transfer", "Host Memcpy", "#c5b0d5"),
    ("control", "Control", "#bab0ac"),
)

_COMM_SUBS = frozenset({
    "network_collective",
    "network_p2p",
    "device_copy",
    "host_transfer",
    "collective_comm",
})

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
    if sub == "experts_comp":
        sub = "moe_expert"
    elif sub == "collective_comm":
        sub = "network_collective"
    if not pp_comm_split:
        return sub
    if sub not in _COMM_SUBS:
        return sub
    if sub == "network_p2p":
        return "pp_comm"
    if sub in ("device_copy", "host_transfer"):
        return "local_comm"
    op = classify_comm_operation(
        e.get("name", ""),
        e.get("cat", ""),
        args=e.get("args"),
    )
    if op == "point_to_point":
        return "pp_comm"
    if op in ("device_copy", "host_transfer"):
        return "local_comm"
    return sub if sub != "collective_comm" else "network_collective"


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


COMM_TIMELINE_LAYERS = (
    ("network_collective", "Network Collective", "#e45756"),
    ("network_p2p", "Network P2P", "#d62728"),
    ("device_copy", "Device Memcpy (D2D)", "#ff9896"),
    ("host_transfer", "Host Transfer (H2D/D2H)", "#c5b0d5"),
)


def plot_comm_timeline(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Communication-only timeline",
    max_ms: float | None = None,
    time_origin_us: int | None = None,
) -> None:
    """Timeline showing only communication events (fabric + local memcpy)."""
    comm_events = [
        e for e in events
        if e.get("kind") == "comm" or e.get("sub") in (
            "network_collective", "network_p2p", "device_copy", "host_transfer"
        )
    ]
    if not comm_events:
        return
    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    fig, ax = plt.subplots(figsize=(18, 3.5))
    layers = COMM_TIMELINE_LAYERS
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}

    for sub, _label, color in layers:
        intervals: list[tuple[float, float]] = []
        for e in comm_events:
            if e.get("sub") != sub:
                continue
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            continue
        min_w = _min_visible_width_ms(intervals)
        intervals = [(s, max(w, min_w)) for s, w in intervals]
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


FABRIC_COMM_TIMELINE_LAYERS = (
    ("network_collective", "Network Collective", "#e45756"),
    ("network_p2p", "Network P2P", "#d62728"),
)


def _min_visible_width_ms(
    intervals: list[tuple[float, float]],
    fig_width_in: float = 18.0,
) -> float:
    """Minimum bar width so events are at least ~1.5px wide on screen."""
    if not intervals:
        return 0.0
    max_t = max(start + width for start, width in intervals)
    min_t = min(start for start, _ in intervals)
    span = max_t - min_t
    if span <= 0:
        return 0.0
    dpi = 200
    pixels = fig_width_in * dpi
    return span / pixels * 1.5


def plot_fabric_comm_timeline(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Collective communication timeline",
    max_ms: float | None = None,
    time_origin_us: int | None = None,
) -> None:
    """Timeline showing only fabric communication (collectives + P2P, no memcpy)."""
    comm_events = [
        e for e in events
        if e.get("sub") in ("network_collective", "network_p2p")
    ]
    if not comm_events:
        return
    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    fig, ax = plt.subplots(figsize=(18, 2.5))
    layers = FABRIC_COMM_TIMELINE_LAYERS
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}

    for sub, _label, color in layers:
        intervals: list[tuple[float, float]] = []
        for e in comm_events:
            if e.get("sub") != sub:
                continue
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            continue
        min_w = _min_visible_width_ms(intervals)
        intervals = [(s, max(w, min_w)) for s, w in intervals]
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


def plot_fabric_comm_timeline_trimmed(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Collective communication timeline (trimmed)",
    max_ms: float | None = None,
    time_origin_us: int | None = None,
) -> None:
    """Like plot_fabric_comm_timeline but x-axis trimmed to first/last comm event."""
    comm_events = [
        e for e in events
        if e.get("sub") in ("network_collective", "network_p2p")
    ]
    if not comm_events:
        return
    first_ts = min(e["ts"] for e in comm_events)
    last_end = max(e["end"] for e in comm_events)
    t0 = first_ts
    span_ms = (last_end - first_ts) / 1000.0

    fig, ax = plt.subplots(figsize=(18, 2.5))
    layers = FABRIC_COMM_TIMELINE_LAYERS
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}

    for sub, _label, color in layers:
        intervals: list[tuple[float, float]] = []
        for e in comm_events:
            if e.get("sub") != sub:
                continue
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            continue
        min_w = _min_visible_width_ms(intervals)
        intervals = [(s, max(w, min_w)) for s, w in intervals]
        y = y_positions[sub]
        ax.broken_barh(intervals, (y, 0.85), facecolors=color, edgecolors="none")

    ax.set_yticks([i + 0.4 for i in range(len(layers))])
    ax.set_yticklabels([spec[1] for spec in layers])
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    if max_ms is not None and max_ms < span_ms:
        ax.set_xlim(0, max_ms)
    else:
        ax.set_xlim(0, span_ms * 1.02)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_fabric_comm_timeline_active(
    events: list[dict[str, Any]],
    active_segments_us: list[tuple[int, int]],
    out_path: Path,
    *,
    title: str = "Fabric communication timeline (active segments)",
    max_ms: float | None = None,
) -> None:
    """Fabric timeline with active segments concatenated on the x-axis."""
    if not events or not active_segments_us:
        return
    comm_events = [
        e for e in events
        if e.get("sub") in ("network_collective", "network_p2p")
    ]
    if not comm_events:
        return

    gap_ms = 10.0
    offsets: list[tuple[int, int, float, float]] = []
    cursor_ms = 0.0
    for start_us, end_us in active_segments_us:
        seg_ms = max((end_us - start_us) / 1000.0, 0.0)
        offsets.append((start_us, end_us, cursor_ms, seg_ms))
        cursor_ms += seg_ms + gap_ms
    total_ms = max(cursor_ms - gap_ms, 0.0)
    if total_ms <= 0:
        return

    fig, ax = plt.subplots(figsize=(18, 2.7))
    layers = FABRIC_COMM_TIMELINE_LAYERS
    y_positions = {spec[0]: i for i, spec in enumerate(layers)}

    for sub, _label, color in layers:
        intervals: list[tuple[float, float]] = []
        for e in comm_events:
            if e.get("sub") != sub:
                continue
            for start_us, end_us, offset_ms, _seg_ms in offsets:
                if e["end"] <= start_us or e["ts"] >= end_us:
                    continue
                clipped_start = max(e["ts"], start_us)
                clipped_end = min(e["end"], end_us)
                intervals.append((
                    offset_ms + (clipped_start - start_us) / 1000.0,
                    max((clipped_end - clipped_start) / 1000.0, 0.0),
                ))
        if not intervals:
            continue
        min_w = _min_visible_width_ms(intervals)
        intervals = [(s, max(w, min_w)) for s, w in intervals]
        y = y_positions[sub]
        ax.broken_barh(intervals, (y, 0.85), facecolors=color, edgecolors="none")

    for idx, (_start_us, _end_us, offset_ms, seg_ms) in enumerate(offsets):
        if idx > 0:
            ax.axvline(offset_ms - gap_ms / 2.0, color="0.5",
                       linestyle=":", linewidth=0.8)
        ax.text(
            offset_ms + seg_ms / 2.0,
            len(layers) + 0.05,
            f"segment {idx}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_yticks([i + 0.4 for i in range(len(layers))])
    ax.set_yticklabels([spec[1] for spec in layers])
    ax.set_xlabel("Active-segment time, gaps removed (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(0, min(total_ms, max_ms) if max_ms is not None else total_ms)
    ax.set_ylim(-0.2, len(layers) + 0.45)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_active_window_detection(
    events: list[dict[str, Any]],
    active_segments_us: list[tuple[int, int]],
    out_path: Path,
    *,
    title: str = "Active-window detection",
    density_threshold: int = 50,
    bin_size_us: int = 1_000_000,
    time_origin_us: int | None = None,
) -> None:
    """Plot fabric event density over the full capture with active overlays."""
    if not events:
        return
    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    t1 = max(e["end"] for e in events)
    if t1 <= t0:
        return
    bin_count = int(np.ceil((t1 - t0) / bin_size_us))
    if bin_count <= 0:
        return

    counts = np.zeros(bin_count, dtype=np.float64)
    for e in events:
        if e.get("sub") not in ("network_collective", "network_p2p"):
            continue
        if e["ts"] < t0:
            continue
        idx = int((e["ts"] - t0) // bin_size_us)
        if 0 <= idx < bin_count:
            counts[idx] += 1.0

    bin_s = bin_size_us / 1e6
    density = counts / bin_s
    x_ms = (np.arange(bin_count) * bin_size_us) / 1000.0
    width_ms = bin_size_us / 1000.0
    threshold_per_s = density_threshold / bin_s

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_ms, density, width=width_ms, align="edge",
           color="#4c78a8", alpha=0.75, label="fabric events/s")
    ax.axhline(threshold_per_s, color="#e45756", linestyle="--",
               linewidth=1.2, label=f"threshold ({threshold_per_s:.0f}/s)")
    for idx, (start_us, end_us) in enumerate(active_segments_us):
        start_ms = max(0.0, (start_us - t0) / 1000.0)
        end_ms = max(start_ms, (end_us - t0) / 1000.0)
        ax.axvspan(start_ms, end_ms, color="#54a24b", alpha=0.18,
                   label="active segment" if idx == 0 else None)
    ax.set_xlabel("Time from trace start (ms)")
    ax.set_ylabel("Fabric events per second")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_active_segments_timeline(
    active_segments_us: list[tuple[int, int]],
    out_path: Path,
    *,
    title: str = "Active segments",
    time_origin_us: int | None = None,
) -> None:
    """Show detected active segments on the full capture axis."""
    if not active_segments_us:
        return
    t0 = time_origin_us if time_origin_us is not None else min(
        start for start, _ in active_segments_us
    )
    intervals = [
        ((start - t0) / 1000.0, max((end - start) / 1000.0, 0.0))
        for start, end in active_segments_us
    ]
    fig, ax = plt.subplots(figsize=(14, 1.8))
    ax.broken_barh(intervals, (0, 0.8), facecolors="#54a24b",
                   edgecolors="#2f6f35", linewidth=0.8)
    for idx, (start_ms, width_ms) in enumerate(intervals):
        ax.text(start_ms + width_ms / 2.0, 0.4, str(idx),
                ha="center", va="center", fontsize=9)
    ax.set_yticks([0.4])
    ax.set_yticklabels(["active"])
    ax.set_xlabel("Time from trace start (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
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

    def record(
        gap_start: int,
        gap_end: int,
        before_win: tuple[int, int],
        after_win: tuple[int, int],
    ):
        gap_us = gap_end - gap_start
        if gap_us < min_gap_us:
            return
        bk, bs, bn = _dominant_sub_in_window(events, *before_win)
        ak, a_sub, an = _dominant_sub_in_window(events, *after_win)
        transition = (
            f"{SUBCATEGORY_LABELS.get(bs, bs)} → idle → "
            f"{SUBCATEGORY_LABELS.get(a_sub, a_sub)}"
        )
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
                "transition": transition,
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
        "idle_ms_after": sorted(
            time_by_before.items(), key=lambda x: x[1], reverse=True
        )[:15],
        "idle_ms_before": sorted(
            time_by_after.items(), key=lambda x: x[1], reverse=True
        )[:15],
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
    """Heatmap of idle time by before/after activity."""
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

    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 0.9),
                 max(6, len(row_labels) * 0.5))
    )
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
            "note": (
                "Full per-gap list omitted if >500 gaps; "
                "see longest_gaps in summary."
            ),
        },
    )
    return summary


def plot_traffic_volume_pct(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    parallel_label: str,
    duty: dict[str, float] | None = None,
) -> None:
    """Bar chart of duty-cycle percent per decomposed category."""
    if duty is None:
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
    """Legacy CUPTI-based stats (fallback when inner comm NVTX is missing)."""
    comms = [
        e
        for e in events
        if is_fabric_event(e)
        and (phase is None or e.get("phase", "unknown") == phase)
    ]
    sizes: list[int] = []
    for e in comms:
        b = _extract_comm_bytes(e)
        if b is not None and b > 0:
            sizes.append(b)
    return {
        "count": len(comms),
        "logical_tensor_bytes_list": sizes,
        "avg_logical_tensor_bytes": float(np.mean(sizes)) if sizes else 0.0,
        "total_logical_tensor_bytes": int(sum(sizes)),
        "bytes_label": "logical tensor bytes (CUPTI fallback)",
        "sizes_bytes": sizes,
        "avg_bytes": float(np.mean(sizes)) if sizes else 0.0,
        "total_bytes": int(sum(sizes)),
        "source": "cupit_events",
    }


def plot_message_stats(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    prefix: str,
    comm_nvtx_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prefill vs decode comm counts from conclusive inner comm NVTX only."""
    from plotting_tools.comm_nvtx import (
        BYTES_LABEL,
        comm_nvtx_message_stats,
        comm_phase_exclusion_counts,
        filter_conclusive_comm_records,
    )

    if comm_nvtx_records:
        conclusive = filter_conclusive_comm_records(comm_nvtx_records)
        stats = {
            "prefill": comm_nvtx_message_stats(
                conclusive, phase="prefill", conclusive_only=True
            ),
            "decode": comm_nvtx_message_stats(
                conclusive, phase="decode", conclusive_only=True
            ),
        }
        all_s = comm_nvtx_message_stats(conclusive, conclusive_only=True)
        all_s["source"] = "comm_nvtx_conclusive"
        stats["prefill"]["source"] = "comm_nvtx_conclusive"
        stats["decode"]["source"] = "comm_nvtx_conclusive"
        excluded = comm_phase_exclusion_counts(comm_nvtx_records)
    else:
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
        stats["prefill"]["avg_logical_tensor_bytes"] / 1e6,
        stats["decode"]["avg_logical_tensor_bytes"] / 1e6,
        all_s["avg_logical_tensor_bytes"] / 1e6,
    ]

    axes[0].bar(phases, counts, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_ylabel("Comm message count (conclusive inner NVTX)")
    title_suffix = ""
    if comm_nvtx_records:
        title_suffix = " [conclusive prefill/decode only]"
    axes[0].set_title(f"{prefix}: comm message count{title_suffix}")

    axes[1].bar(phases, avgs_mb, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_ylabel(f"Avg {BYTES_LABEL} (MB)")
    axes[1].set_title(f"{prefix}: avg {BYTES_LABEL}{title_suffix}")

    plt.tight_layout()
    save_figure(fig, out_path)
    out: dict[str, Any] = {
        "prefill": stats["prefill"],
        "decode": stats["decode"],
        "all": all_s,
    }
    if comm_nvtx_records:
        out["excluded_from_conclusions"] = excluded
    return out


def write_comm_nvtx_table(
    records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Write authoritative inner comm NVTX message records."""
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)


def plot_message_size_cdf(
    comm_nvtx_records: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Logical message size CDF",
) -> dict[str, Any]:
    """CDF of logical message bytes from inner communication NVTX records."""
    series: dict[str, list[int]] = defaultdict(list)
    for r in comm_nvtx_records:
        nbytes = r.get("logical_tensor_bytes")
        if nbytes is None:
            continue
        n = int(nbytes)
        if n <= 0:
            continue
        op = str(r.get("op") or "unknown")
        phase = str(r.get("phase") or "unknown")
        series[f"{op}:{phase}"].append(n)

    summary = {
        key: {
            "count": len(vals),
            "min_bytes": int(min(vals)),
            "median_bytes": float(np.median(vals)),
            "p95_bytes": float(np.percentile(vals, 95)),
            "max_bytes": int(max(vals)),
            "total_bytes": int(sum(vals)),
        }
        for key, vals in sorted(series.items())
    }
    if not series:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Logical message bytes unavailable\n"
            "Use comm NVTX markers or theoretical estimates.",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_title(title)
        plt.tight_layout()
        save_figure(fig, out_path)
        return {"series": {}, "bytes_available": False}

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, vals in sorted(series.items()):
        xs = np.sort(np.asarray(vals, dtype=np.float64))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", label=f"{key} (n={len(xs)})")
    ax.set_xscale("log")
    ax.set_xlabel("Logical message bytes")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, out_path)
    return {"series": summary, "bytes_available": True}


def plot_key_value_table(
    rows: list[tuple[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Write a compact two-column table figure."""
    if not rows:
        return
    height = max(2.0, 0.32 * len(rows) + 0.8)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.axis("off")
    cell_text = [[k, "" if v is None else str(v)] for k, v in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Field", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.42, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#dddddd")
    ax.set_title(title, pad=12)
    plt.tight_layout()
    save_figure(fig, out_path)


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


def _pp_stage_ranks(pp_rank: int, tp: int) -> list[int]:
    return list(range(pp_rank * tp, (pp_rank + 1) * tp))


def _tp_peer_ranks(local_rank: int, tp: int) -> list[int]:
    base = (local_rank // tp) * tp
    return [base + j for j in range(tp) if base + j != local_rank]


def _infer_pp_peer_ranks(
    local_rank: int,
    *,
    tp: int,
    pp: int,
    name: str,
) -> list[int]:
    peer = _peer_from_nccl_name(name)
    if peer is not None and peer != local_rank:
        return [peer]

    lower = name.lower()
    if not any(k in lower for k in ("sendrecv", "send", "recv")):
        return []

    pp_rank = local_rank // tp
    targets: list[int] = []
    if "send" in lower and "recv" not in lower:
        if pp_rank + 1 < pp:
            targets = _pp_stage_ranks(pp_rank + 1, tp)
    elif "recv" in lower and "send" not in lower:
        if pp_rank > 0:
            targets = _pp_stage_ranks(pp_rank - 1, tp)
    else:
        if pp_rank > 0:
            targets.extend(_pp_stage_ranks(pp_rank - 1, tp))
        if pp_rank + 1 < pp:
            targets.extend(_pp_stage_ranks(pp_rank + 1, tp))
    return [t for t in targets if t != local_rank]


def _infer_traffic_peers(
    local_rank: int,
    n_ranks: int,
    name: str,
    cat: str,
    *,
    tp: int = 1,
    pp: int = 1,
) -> list[int]:
    explicit = _peer_from_nccl_name(name)
    if explicit is not None and 0 <= explicit < n_ranks and explicit != local_rank:
        return [explicit]

    op = classify_comm_operation(name, cat)
    if op is None and cat.lower() == "kernel":
        op = classify_comm_operation(name, "kernel")

    if op in ("point_to_point",) or (
        op is None
        and any(k in name.lower() for k in ("sendrecv", "send", "recv"))
    ):
        if pp > 1:
            return _infer_pp_peer_ranks(local_rank, tp=tp, pp=pp, name=name)
        if n_ranks == 2:
            other = 1 - local_rank
            return [other] if other != local_rank else []

    if op in (
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
        "all_to_all",
        "nccl_other",
    ):
        if tp > 1:
            return _tp_peer_ranks(local_rank, tp)

    return []


def _is_traffic_comm_event(e: dict[str, Any]) -> bool:
    """Fabric / network traffic only (see classify.is_fabric_event)."""
    return is_fabric_event(e)


def build_traffic_matrix(
    events: list[dict[str, Any]],
    n_ranks: int,
    *,
    local_rank: int = 0,
    tp: int = 1,
    pp: int = 1,
) -> np.ndarray:
    """Rank-to-rank comm volume (bytes, else µs) from this trace's perspective."""
    mat = np.zeros((n_ranks, n_ranks), dtype=np.float64)
    for e in events:
        if not _is_traffic_comm_event(e):
            continue
        vol = _comm_volume(e)
        peers = _infer_traffic_peers(
            local_rank,
            n_ranks,
            e.get("name", ""),
            e.get("cat", ""),
            tp=tp,
            pp=pp,
        )
        if not peers:
            continue
        share = vol / len(peers)
        for peer in peers:
            if 0 <= peer < n_ranks and peer != local_rank:
                mat[local_rank, peer] += share
    return mat


def symmetrize_traffic_matrix(
    mat: np.ndarray,
    *,
    tp: int,
    pp: int,
) -> np.ndarray:
    """Balance one-sided profiler durations across TP pairs and PP stages."""
    out = mat.copy()
    for stage in range(pp):
        base = stage * tp
        for a in range(tp):
            for b in range(a + 1, tp):
                i, j = base + a, base + b
                if i >= out.shape[0] or j >= out.shape[1]:
                    continue
                v = max(out[i, j], out[j, i])
                out[i, j] = out[j, i] = v
    for stage_a in range(pp):
        for stage_b in range(pp):
            if stage_a == stage_b:
                continue
            for i in range(stage_a * tp, stage_a * tp + tp):
                for j in range(stage_b * tp, stage_b * tp + tp):
                    if i >= out.shape[0] or j >= out.shape[1]:
                        continue
                    v = max(out[i, j], out[j, i])
                    out[i, j] = out[j, i] = v
    return out


def merge_rank_traffic_matrices(
    matrices: list[np.ndarray],
    *,
    tp: int = 1,
    pp: int = 1,
    symmetrize: bool = True,
) -> np.ndarray:
    """Sum per-rank trace matrices into one job-level rank x rank view."""
    if not matrices:
        return np.zeros((1, 1), dtype=np.float64)
    n = max(m.shape[0] for m in matrices)
    out = np.zeros((n, n), dtype=np.float64)
    for m in matrices:
        out[: m.shape[0], : m.shape[1]] += m
    if symmetrize and tp >= 1 and pp >= 1:
        out = symmetrize_traffic_matrix(out, tp=tp, pp=pp)
    return out


def _rank_node_group(rank: int, *, tp: int, pp: int) -> int:
    _ = pp
    return rank // max(tp, 1)


def _heatmap_tick_labels(
    n: int,
    *,
    rank_node_names: dict[int, str] | None,
) -> list[str]:
    labels: list[str] = []
    for r in range(n):
        node = (rank_node_names or {}).get(r, "")
        labels.append(f"{r}\n{node}" if node else str(r))
    return labels


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
        elif any(
            k in lower
            for k in ("allreduce", "allgather", "reducescatter", "broadcast")
        ):
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
    tp: int = 1,
    pp: int = 1,
    rank_node_names: dict[int, str] | None = None,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(6, matrix.shape[1] * 0.9),
                 max(5, matrix.shape[0] * 0.8))
    )
    fig.patch.set_facecolor(HEATMAP_BG)
    ax.set_facecolor(HEATMAP_BG)

    n0, n1 = matrix.shape
    vmax = float(matrix.max()) if matrix.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    for i in range(n0):
        for j in range(n1):
            if _rank_node_group(i, tp=tp, pp=pp) == _rank_node_group(j, tp=tp, pp=pp):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=HEATMAP_NODE_HIGHLIGHT,
                        edgecolor="#e8e0c8",
                        linewidth=0.6,
                        zorder=0,
                    )
                )

    masked = np.ma.masked_where(matrix <= 0, matrix)
    im = ax.imshow(
        masked,
        cmap=TRAFFIC_HEATMAP_CMAP,
        vmin=0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(n1))
    ax.set_yticks(range(n0))
    ax.set_xticklabels(_heatmap_tick_labels(n1, rank_node_names=rank_node_names))
    ax.set_yticklabels(_heatmap_tick_labels(n0, rank_node_names=rank_node_names))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(8)

    if annotate:
        for i in range(n0):
            for j in range(n1):
                val = matrix[i, j]
                if val > 0:
                    txt = f"{val:.2e}" if val >= 1e4 else f"{val:.1f}"
                    color = "white" if val > 0.55 * vmax else "#1a1a1a"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        zorder=2,
                    )

    cbar = fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor("#cccccc")
    cbar.ax.set_facecolor(HEATMAP_BG)

    ax.legend(
        handles=[
            Patch(
                facecolor=HEATMAP_NODE_HIGHLIGHT,
                edgecolor="#e8e0c8",
                label="Same node (TP group)",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1.22, 1.0),
        fontsize=8,
        framealpha=0.95,
    )

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
            k in s
            for k in (
                "expert",
                "moe",
                "dispatch",
                "combine",
                "all2all",
                "all_to_all",
            )
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


def _classify_event_for_comm_breakdown(
    e: dict[str, Any],
) -> str | None:
    sub = e.get("sub")
    if sub in MOVEMENT_OPS:
        return sub

    kind = e.get("kind")
    name = e.get("name", "")
    cat = e.get("cat", "")
    args = e.get("args") or {}

    if kind == "control":
        op = classify_comm_operation(name, cat, args=args)
        return op if op in MOVEMENT_OPS else None

    if kind == "comm":
        return classify_comm_operation(name, cat, args=args)

    return None


def _short_comm_label(name: str, cat: str, width: int = 96) -> str:
    label = f"{cat}|{name}".strip("|")
    if len(label) <= width:
        return label
    return label[: width - 1] + "…"


CommBreakdownScope = Literal["fabric", "movement", "all"]


def comm_operation_breakdown(
    events: list[dict[str, Any]],
    *,
    scope: CommBreakdownScope = "all",
) -> tuple[dict[str, dict[str, float | int]], Counter[str]]:
    """
    Count duration/bytes per comm op bucket.

    scope:
      fabric   — is_fabric_event + op in FABRIC_COMM_OPS
      movement — device_copy (local) + host_transfer (not network)
      all      — full inventory
    """
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "dur_us": 0, "logical_tensor_bytes": 0, "bytes": 0}
    )
    unclassified: Counter[str] = Counter()
    for e in events:
        if scope == "fabric":
            if not is_fabric_event(e):
                continue
            op = classify_comm_operation(
                e.get("name", ""),
                e.get("cat", ""),
                args=e.get("args"),
            )
            if not is_fabric_comm_op(op):
                if op in ("unclassified_comm", "nccl_other"):
                    unclassified[
                        _short_comm_label(e.get("name", ""), e.get("cat", ""))
                    ] += 1
                continue
        elif scope == "movement":
            if is_fabric_event(e):
                continue
            op = _classify_event_for_comm_breakdown(e)
            if op is None or not is_movement_op(op):
                continue
        else:
            op = _classify_event_for_comm_breakdown(e)
            if op is None:
                continue
            if op == "unclassified_comm":
                unclassified[
                    _short_comm_label(e.get("name", ""), e.get("cat", ""))
                ] += 1
        stats[op]["count"] += 1
        stats[op]["dur_us"] += e["dur"]
        b = _extract_comm_bytes(e)
        if b:
            stats[op]["bytes"] += b
    return dict(stats), unclassified


def merge_comm_operation_breakdowns(
    breakdowns: list[dict[str, dict[str, float | int]]],
    *,
    include_ops: frozenset[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "dur_us": 0, "logical_tensor_bytes": 0, "bytes": 0}
    )
    for stats in breakdowns:
        for op, vals in stats.items():
            if include_ops is not None and op not in include_ops:
                continue
            merged[op]["count"] += vals.get("count", 0)
            merged[op]["dur_us"] += vals.get("dur_us", 0)
            merged[op]["bytes"] += vals.get("bytes", 0)
    return dict(merged)


def plot_fabric_comm_breakdown(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> tuple[dict[str, dict[str, float | int]], Counter[str]]:
    """Bar chart: fabric NCCL/custom-AR ops only."""
    stats, unclassified = comm_operation_breakdown(events, scope="fabric")
    plot_comm_breakdown_stats(stats, out_path, title=title)
    return stats, unclassified


def plot_data_movement_breakdown(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> tuple[dict[str, dict[str, float | int]], Counter[str]]:
    """Bar chart: local device memcpy + host transfer (not fabric)."""
    stats, unclassified = comm_operation_breakdown(events, scope="movement")
    plot_comm_breakdown_stats(stats, out_path, title=title)
    return stats, unclassified


def plot_collective_ops_breakdown(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> tuple[dict[str, dict[str, float | int]], Counter[str]]:
    """Backward-compatible alias: fabric-only breakdown."""
    return plot_fabric_comm_breakdown(events, out_path, title=title)


def collect_nccl_ops(
    events: list[dict[str, Any]],
    *,
    comm_nvtx_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Prefer inner comm NVTX records; fall back to CUPTI NCCL kernels."""
    if comm_nvtx_records:
        t0 = min(r["ts"] for r in comm_nvtx_records)
        ops: list[dict[str, Any]] = []
        for r in comm_nvtx_records:
            ops.append(
                {
                    "ts_ms": (r["ts"] - t0) / 1000.0,
                    "name": r.get("nvtx_name", r.get("op", "")),
                    "op": r.get("op"),
                    "phase": r.get("phase"),
                    "tensor_key": r.get("tensor_key"),
                    "shape": r.get("shape"),
                    "dur_us": r.get("dur_us"),
                    "logical_tensor_bytes": r.get("logical_tensor_bytes"),
                    "bytes_label": "logical tensor bytes",
                    "peer": r.get("peer"),
                    "source": "comm_nvtx",
                }
            )
        return ops

    ops = []
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
                "logical_tensor_bytes": _extract_comm_bytes(e),
                "bytes_label": "logical tensor bytes (CUPTI fallback)",
                "bytes": _extract_comm_bytes(e),
                "shape": shape,
                "source": "cupit_events",
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
    comms = [(e["ts"], e["end"]) for e in events if is_fabric_event(e)]
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
    comms = sorted(
        [e for e in events if is_fabric_event(e)],
        key=lambda e: e["ts"],
    )
    if len(comms) < 2:
        return []
    return [
        (comms[i + 1]["ts"] - comms[i]["ts"]) / 1000.0
        for i in range(len(comms) - 1)
    ]


def plot_comm_breakdown_stats(
    stats: dict[str, dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Bar chart from pre-aggregated comm op stats."""
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


def plot_nccl_log_ops_breakdown(
    stats: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """NCCL INFO log op count and byte-size breakdown."""
    if not stats:
        return
    ops = sorted(
        stats.keys(),
        key=lambda k: float(stats[k].get("algorithmic_rank_bytes_estimated", 0.0)),
        reverse=True,
    )
    labels = [comm_operation_label(o) for o in ops]
    counts = [int(stats[o].get("count", 0)) for o in ops]
    logical_mb = [float(stats[o].get("logical_bytes", 0.0)) / 1e6 for o in ops]
    algo_mb = [
        float(stats[o].get("algorithmic_rank_bytes_estimated", 0.0)) / 1e6
        for o in ops
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ops))
    axes[0].bar(x, counts, color="#e45756")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("NCCL log op records")
    axes[0].set_title("Count by NCCL op")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    width = 0.38
    axes[1].bar(x - width / 2, logical_mb, width, label="logical op MB")
    axes[1].bar(
        x + width / 2,
        algo_mb,
        width,
        label="algorithmic rank MB estimate",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("MB")
    axes[1].set_title("Bytes from NCCL INFO lines")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_collective_ops_breakdown_stats(
    stats: dict[str, dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Alias for plot_comm_breakdown_stats."""
    plot_comm_breakdown_stats(stats, out_path, title=title)


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


def plot_layer_slice_compute_comm_idle(
    events: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Layer-slice timeline",
    time_origin_us: int | None = None,
) -> None:
    """Three-band timeline for a small slice: compute / comm / idle.

    - Top band: compute kernels
    - Middle band: communication kernels/transfers
    - Bottom band: idle gaps between compute+comm activity
    """
    if not events:
        return

    t0 = _plot_t0(events, time_origin_us=time_origin_us)
    t_start = min(e["ts"] for e in events)
    t_end = max(e["end"] for e in events)

    compute_intervals: list[tuple[float, float]] = []
    comm_intervals: list[tuple[float, float]] = []
    active_us: list[tuple[int, int]] = []

    for e in events:
        if e["kind"] == "compute":
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                compute_intervals.append(iv)
                active_us.append((e["ts"], e["end"]))
        elif e["kind"] == "comm":
            iv = _timeline_interval_ms(e, t0)
            if iv is not None:
                comm_intervals.append(iv)
                active_us.append((e["ts"], e["end"]))

    merged_active = merge_intervals(active_us)
    idle_intervals_ms: list[tuple[float, float]] = []
    cur = t_start
    for s, e in merged_active:
        if cur < s:
            idle_intervals_ms.append(((cur - t0) / 1000.0, (s - cur) / 1000.0))
        cur = max(cur, e)
    if cur < t_end:
        idle_intervals_ms.append(((cur - t0) / 1000.0, (t_end - cur) / 1000.0))

    fig, ax = plt.subplots(figsize=(14, 3.2))
    if compute_intervals:
        ax.broken_barh(
            compute_intervals,
            (2.0, 0.75),
            facecolors="#4c78a8",
            edgecolors="none",
            label="Compute",
        )
    if comm_intervals:
        ax.broken_barh(
            comm_intervals,
            (1.0, 0.75),
            facecolors="#f58518",
            edgecolors="none",
            label="Comm",
        )
    if idle_intervals_ms:
        ax.broken_barh(
            idle_intervals_ms,
            (0.0, 0.75),
            facecolors="#b8b8b8",
            edgecolors="none",
            label="Idle",
        )

    ax.set_yticks([0.375, 1.375, 2.375])
    ax.set_yticklabels(["Idle", "Comm", "Compute"])
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_classification_histogram(
    counts: dict[str, int],
    out_path: Path,
    *,
    title: str,
    xlabel: str = "Count",
) -> None:
    """Horizontal bar chart of label → count (unclassified / skipped runtime)."""
    if not counts:
        return
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30]
    labels = [k[:70] + ("…" if len(k) > 70 else "") for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(items))))
    ax.barh(range(len(items)), vals, color="#e45756")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_request_timeline(
    iterations: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Request arrival & active batch over time",
) -> None:
    """Plot request arrivals and active batch size from EngineCore iteration log.

    Shows:
    - Cumulative request arrivals (step function)
    - Active batch size (context + generation requests per iteration)
    - Prefill vs decode breakdown stacked
    """
    if not iterations:
        return

    # Compute cumulative arrivals using elapsed_ms for sub-second precision
    cumulative_t_s: list[float] = []
    cumulative_arrivals: list[int] = []
    active_t_s: list[float] = []
    ctx_reqs: list[int] = []
    gen_reqs: list[int] = []

    running_t_ms = 0.0
    total_arrivals = 0
    for it in iterations:
        t_s = running_t_ms / 1000.0
        active_t_s.append(t_s)
        ctx_reqs.append(it["context_requests"])
        gen_reqs.append(it["generation_requests"])

        if it["context_requests"] > 0:
            total_arrivals += it["context_requests"]
            cumulative_t_s.append(t_s)
            cumulative_arrivals.append(total_arrivals)

        running_t_ms += it["elapsed_ms"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: cumulative arrivals
    if cumulative_t_s:
        ax1.step(cumulative_t_s, cumulative_arrivals, where="post",
                 color="#1f77b4", linewidth=1.5)
        ax1.set_ylabel("Cumulative requests arrived")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(total_arrivals, color="gray", linestyle="--", alpha=0.5,
                    label=f"total={total_arrivals}")
        ax1.legend(loc="lower right")

    # Bottom: active batch size (stacked)
    total_active = [c + g for c, g in zip(ctx_reqs, gen_reqs)]
    ax2.fill_between(active_t_s, 0, gen_reqs, alpha=0.6,
                     color="#2ca02c", label="Generation (decode)")
    ax2.fill_between(active_t_s, gen_reqs, total_active, alpha=0.6,
                     color="#ff7f0e", label="Context (prefill)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Active requests")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, out_path)


def plot_iteration_token_timeline(
    iterations: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Tokens per iteration",
    max_iters: int | None = None,
) -> None:
    """Plot prefill vs decode tokens per iteration (from EngineCore logs).

    Produces a single-panel plot with stacked filled regions:
      - Prefill tokens (context_tokens)
      - Decode tokens (generation_tokens)
    """
    if not iterations:
        return

    iters = iterations[:max_iters] if max_iters is not None else iterations
    xs = [it["iteration"] for it in iters]
    prefill = np.array([it["context_tokens"] for it in iters], dtype=float)
    decode = np.array([it["generation_tokens"] for it in iters], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(xs, 0, prefill, color="#7aa6d8", alpha=0.75, label="Prefill Tokens")
    ax.fill_between(
        xs,
        prefill,
        prefill + decode,
        color="#e07a7a",
        alpha=0.75,
        label="Decode Tokens",
    )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("# Tokens")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_prefill_zoom_panel(
    iterations: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Prefill-focused zoom",
    context_iters: int = 30,
) -> None:
    """Plot a zoomed token timeline around prefill-active iterations."""
    if not iterations:
        return

    x = np.array([int(it["iteration"]) for it in iterations], dtype=np.int64)
    ctx = np.array([int(it.get("context_tokens", 0)) for it in iterations], dtype=np.float64)
    gen = np.array(
        [int(it.get("generation_tokens", 0)) for it in iterations], dtype=np.float64
    )

    prefill_idx = np.where(ctx > 0)[0]
    if prefill_idx.size == 0:
        lo = 0
        hi = min(len(x), 300)
    else:
        lo = max(0, int(prefill_idx.min()) - context_iters)
        hi = min(len(x), int(prefill_idx.max()) + context_iters + 1)

    xz = x[lo:hi]
    ctxz = ctx[lo:hi]
    genz = gen[lo:hi]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(
        xz,
        0,
        genz,
        step="mid",
        alpha=0.55,
        color="#e07a7a",
        label="Decode Tokens",
    )
    ax.fill_between(
        xz,
        genz,
        genz + ctxz,
        step="mid",
        alpha=0.80,
        color="#7aa6d8",
        label="Prefill Tokens",
    )
    nz = np.where(ctxz > 0)[0]
    if nz.size > 0:
        ax.scatter(
            xz[nz],
            (genz + ctxz)[nz],
            s=9,
            color="#2f5f91",
            alpha=0.9,
            zorder=5,
            label="Prefill-active iters",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("# Tokens")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    plt.tight_layout()
    save_figure(fig, out_path)


def summarize_iteration_tokens(
    iterations: list[dict[str, Any]],
    *,
    max_num_batched_tokens: int | None = None,
) -> dict[str, Any]:
    """Summarize token counts processed per scheduler iteration."""
    if not iterations:
        return {}
    total_tokens = np.array([
        int(it.get("context_tokens", 0)) + int(it.get("generation_tokens", 0))
        for it in iterations
    ], dtype=np.float64)
    context_tokens = np.array([
        int(it.get("context_tokens", 0)) for it in iterations
    ], dtype=np.float64)
    generation_tokens = np.array([
        int(it.get("generation_tokens", 0)) for it in iterations
    ], dtype=np.float64)
    positive = total_tokens[total_tokens > 0]
    observed_max = float(total_tokens.max()) if total_tokens.size else 0.0
    observed_min = float(total_tokens.min()) if total_tokens.size else 0.0
    capacity = max_num_batched_tokens or int(observed_max)
    low_threshold = max(1.0, 0.25 * capacity) if capacity else 1.0
    low_mask = (total_tokens > 0) & (total_tokens < low_threshold)
    prefill_mask = context_tokens > 0
    decode_mask = generation_tokens > 0
    return {
        "iterations": len(iterations),
        "max_num_batched_tokens": max_num_batched_tokens,
        "total_tokens": {
            "min": observed_min,
            "median_positive": (
                float(np.median(positive)) if positive.size else 0.0
            ),
            "mean": float(total_tokens.mean()) if total_tokens.size else 0.0,
            "p95": (
                float(np.percentile(total_tokens, 95))
                if total_tokens.size else 0.0
            ),
            "max": observed_max,
        },
        "prefill_tokens_total": int(context_tokens.sum()),
        "decode_tokens_total": int(generation_tokens.sum()),
        "context_tokens_total": int(context_tokens.sum()),
        "generation_tokens_total": int(generation_tokens.sum()),
        "prefill_iterations": int(prefill_mask.sum()),
        "decode_iterations": int(decode_mask.sum()),
        "mixed_prefill_decode_iterations": int((prefill_mask & decode_mask).sum()),
        "low_token_threshold": float(low_threshold),
        "low_token_iterations": int(low_mask.sum()),
        "low_token_iteration_fraction": (
            float(low_mask.sum() / len(iterations)) if iterations else 0.0
        ),
    }


def plot_batch_tokens_per_iteration(
    iterations: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Tokens processed per scheduler iteration",
    max_num_batched_tokens: int | None = None,
) -> dict[str, Any]:
    """Plot total/context/generation tokens per vLLM scheduler iteration."""
    if not iterations:
        return {}

    x = np.array([int(it["iteration"]) for it in iterations], dtype=np.int64)
    ctx = np.array([
        int(it.get("context_tokens", 0)) for it in iterations
    ], dtype=np.float64)
    gen = np.array([
        int(it.get("generation_tokens", 0)) for it in iterations
    ], dtype=np.float64)
    total = ctx + gen
    summary = summarize_iteration_tokens(
        iterations,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    threshold = float(summary.get("low_token_threshold", 0.0))
    if threshold > 0:
        ax1.axhspan(
            0,
            threshold,
            color="#f2d7d5",
            alpha=0.28,
            zorder=0,
            label=f"<25% capacity ({threshold:.0f})",
        )
    ax1.fill_between(
        x,
        0,
        gen,
        step="mid",
        alpha=0.70,
        color="#2ca02c",
        zorder=2,
        label="decode / generation tokens",
    )
    ax1.fill_between(
        x,
        gen,
        total,
        step="mid",
        alpha=0.75,
        color="#ff7f0e",
        zorder=2,
        label="prefill / context tokens",
    )
    ax1.plot(
        x,
        total,
        color="#2f4858",
        linewidth=1.0,
        zorder=3,
        label="total tokens",
    )
    if max_num_batched_tokens:
        ax1.axhline(
            max_num_batched_tokens,
            color="#8c8c8c",
            linestyle="--",
            linewidth=1.0,
            zorder=3,
            label=f"max batched tokens={max_num_batched_tokens}",
        )
    ax1.set_ylabel("Tokens / iteration")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.fill_between(
        x,
        0,
        gen,
        step="mid",
        alpha=0.65,
        color="#2ca02c",
        label="decode / generation tokens",
    )
    ax2.fill_between(
        x,
        gen,
        gen + ctx,
        step="mid",
        alpha=0.65,
        color="#ff7f0e",
        label="prefill / context tokens",
    )
    ax2.set_xlabel("Scheduler iteration")
    ax2.set_ylabel("Token split")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save_figure(fig, out_path)
    return summary


VERBOSE_BAR_CATEGORIES = [
    ("AllReduce", "#c44e52"),
    ("AllGather", "#dd8452"),
    ("ReduceScatter", "#e8875a"),
    ("SendRecv (PP)", "#937860"),
    ("MoE Expert", "#4c72b0"),
    ("Attention", "#55a868"),
    ("MatMul/Projection", "#8172b2"),
    ("MoE Routing", "#64b5cd"),
    ("Gate", "#76b7b2"),
    ("Add/Norm (RMSNorm)", "#ccb974"),
    ("Rotary Embedding", "#c49c94"),
    ("KV Cache Write", "#f7b6d2"),
    ("Sampling", "#aec7e8"),
    ("Masking/Indexing", "#dbdb8d"),
    ("Other Compute", "#9edae5"),
    ("Device Memcpy (D2D)", "#d5bbcd"),
    ("Host Transfer (API)", "#e0c8a8"),
    ("Control (Kernel Launch)", "#b0b0b0"),
    ("Idle/Unaccounted", "#e8e8e8"),
]

_SUB_TO_VERBOSE_CAT: dict[str, str] = {
    "network_collective": "_split_by_name",
    "network_p2p": "SendRecv (PP)",
    "moe_expert": "MoE Expert",
    "attention_comp": "Attention",
    "matmul_gemm": "MatMul/Projection",
    "moe_routing": "MoE Routing",
    "gate_comp": "Gate",
    "add_norm_comp": "Add/Norm (RMSNorm)",
    "rotary_embedding": "Rotary Embedding",
    "kv_cache_write": "KV Cache Write",
    "sampling_overhead": "Sampling",
    "masking_indexing": "Masking/Indexing",
    "other_compute": "Other Compute",
    "device_copy": "Device Memcpy (D2D)",
    "host_transfer": "Host Transfer (API)",
    "control": "Control (Kernel Launch)",
}


def _map_events_to_bar_categories(
    events: list[dict[str, Any]],
) -> dict[str, float]:
    """Map classified events to verbose stacked-bar categories (seconds)."""
    cat_dur: dict[str, float] = {name: 0.0 for name, _ in VERBOSE_BAR_CATEGORIES}

    for e in events:
        dur_s = e["dur"] / 1e6
        sub = e.get("sub", "")
        name_lower = e.get("name", "").lower()

        mapping = _SUB_TO_VERBOSE_CAT.get(sub)
        if mapping == "_split_by_name":
            if "all_gather" in name_lower or "allgather" in name_lower:
                cat_dur["AllGather"] += dur_s
            elif "reduce_scatter" in name_lower or "reducescatter" in name_lower:
                cat_dur["ReduceScatter"] += dur_s
            else:
                cat_dur["AllReduce"] += dur_s
        elif mapping:
            cat_dur[mapping] += dur_s
        else:
            cat_dur["Idle/Unaccounted"] += dur_s

    return cat_dur


def plot_time_breakdown_bar(
    configs: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Time breakdown by operation category",
    ylabel: str = "Time (seconds)",
) -> None:
    """Stacked bar chart of time decomposition per parallelism configuration.

    Each entry in `configs` should have:
        - "label": str (x-axis label, e.g. "T4-P1")
        - "category_seconds": dict mapping category name -> seconds
    """
    if not configs:
        return

    n = len(configs)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n + 3), 6))

    x = np.arange(n)
    bar_width = 0.6 if n <= 4 else 0.8
    bottoms = np.zeros(n)

    for cat_name, color in VERBOSE_BAR_CATEGORIES:
        vals = np.array([c["category_seconds"].get(cat_name, 0.0) for c in configs])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bar_width, bottom=bottoms, label=cat_name, color=color,
               edgecolor="white", linewidth=0.3)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in configs], fontsize=9)
    ax.set_xlabel("Parallelism Configuration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, None)

    plt.tight_layout()
    save_figure(fig, out_path)


def extract_single_layer_events(
    active_events: list[dict[str, Any]],
    *,
    n_layers: int = 48,
    min_nccl_dur_us: int = 100,
) -> list[list[dict[str, Any]]]:
    """Extract individual transformer layer event groups from steady-state decode.

    Uses the alternating all-reduce pattern (2 per layer: after attention, after MoE)
    to segment events into per-layer groups. Targets non-CUDA-Graph iterations where
    kernel durations are accurately reported by the profiler.

    Each layer spans from one NCCL event start to the next-but-one NCCL event start,
    capturing both the attention and MoE/FFN sub-blocks plus their respective all-reduces.
    """
    nccl_all = sorted(
        [e for e in active_events if e.get("sub") == "network_collective"],
        key=lambda e: e["ts"],
    )
    if len(nccl_all) < n_layers * 2 + 10:
        return []

    # Find the longest run of consecutive non-graphed NCCL events (dur > threshold)
    # These have accurate timing visible to the profiler.
    best_run_start = 0
    best_run_len = 0
    cur_start = 0
    cur_len = 0
    for i, e in enumerate(nccl_all):
        if e["dur"] >= min_nccl_dur_us:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_run_len:
                best_run_len = cur_len
                best_run_start = cur_start
        else:
            cur_len = 0

    if best_run_len < n_layers * 2 + 4:
        # Fall back: use all events regardless of duration
        nccl_run = nccl_all
        offset = min(10, len(nccl_run) // 4)
    else:
        nccl_run = nccl_all[best_run_start:best_run_start + best_run_len]
        offset = 4

    layers: list[list[dict[str, Any]]] = []
    for i in range(offset, min(offset + n_layers * 2, len(nccl_run) - 2), 2):
        l_start = nccl_run[i]["ts"]
        l_end = nccl_run[i + 2]["ts"]
        layer_events = [
            e for e in active_events
            if e["ts"] >= l_start and e["end"] <= l_end
        ]
        if layer_events:
            layers.append(layer_events)

    return layers


def plot_single_layer_breakdown(
    layer_events_list: list[list[dict[str, Any]]],
    out_path: Path,
    *,
    title: str = "Single transformer layer breakdown (decode)",
) -> dict[str, Any]:
    """Plot time breakdown for one representative transformer layer.

    Shows a horizontal stacked bar with comm/compute/other breakdown,
    annotated with the per-category microsecond values.
    Returns the average layer stats dict.
    """
    if not layer_events_list:
        return {}

    from collections import defaultdict

    all_dur_by_cat: dict[str, float] = defaultdict(float)
    all_wall_us: float = 0.0
    n = len(layer_events_list)

    for layer in layer_events_list:
        lt0 = min(e["ts"] for e in layer)
        lt1 = max(e["end"] for e in layer)
        all_wall_us += (lt1 - lt0)
        for e in layer:
            sub = e.get("sub", "other")
            all_dur_by_cat[sub] += e["dur"]

    avg_wall_us = all_wall_us / n
    avg_dur: dict[str, float] = {k: v / n for k, v in all_dur_by_cat.items()}

    # Map to display categories
    LAYER_CATEGORIES = [
        ("AllReduce", "#f4a460", ["network_collective", "network_p2p"]),
        ("Attention", "#4878a8", ["attention_comp"]),
        ("MoE Expert", "#7ec8c8", ["moe_expert"]),
        ("MatMul/Proj", "#5a9e5a", ["matmul_gemm"]),
        ("Norm/Gate/Routing", "#e8d860", ["add_norm_comp", "gate_comp", "moe_routing",
                                           "rotary_embedding", "kv_cache_write"]),
        ("Memory", "#c5b0d5", ["host_transfer", "device_copy"]),
        ("Other/Control", "#a08060", ["control", "sampling_overhead", "other_compute",
                                       "masking_indexing"]),
    ]

    cat_vals: list[tuple[str, str, float]] = []
    for cat_name, color, subs in LAYER_CATEGORIES:
        val = sum(avg_dur.get(s, 0.0) for s in subs)
        if val > 0:
            cat_vals.append((cat_name, color, val))

    idle_us = avg_wall_us - sum(v for _, _, v in cat_vals)
    if idle_us > 0:
        cat_vals.append(("Idle/Gap", "#e0e0e0", idle_us))

    fig, ax = plt.subplots(figsize=(12, 3))

    left = 0.0
    for cat_name, color, val in cat_vals:
        width = val / avg_wall_us
        ax.barh(0, width, left=left, color=color, edgecolor="white", linewidth=0.5,
                label=f"{cat_name} ({val:.0f}μs, {val/avg_wall_us*100:.1f}%)")
        if width > 0.04:
            ax.text(left + width / 2, 0, f"{val:.0f}μs",
                    ha="center", va="center", fontsize=7, fontweight="bold")
        left += width

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Fraction of layer wall time ({avg_wall_us:.0f} μs)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)

    plt.tight_layout()
    save_figure(fig, out_path)

    stats = {
        "avg_layer_wall_us": round(avg_wall_us, 1),
        "avg_layer_wall_ms": round(avg_wall_us / 1000, 3),
        "n_layers_sampled": n,
        "category_us": {name: round(val, 1) for name, _, val in cat_vals},
        "category_pct": {name: round(val / avg_wall_us * 100, 1) for name, _, val in cat_vals},
        "utilization_pct": round((avg_wall_us - idle_us) / avg_wall_us * 100, 1)
                          if idle_us > 0 else 100.0,
    }
    return stats


def write_summary_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
