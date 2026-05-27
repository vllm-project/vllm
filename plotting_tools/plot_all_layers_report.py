from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plotting_tools.nsys_jsonl import load_nsys_jsonl
from plotting_tools.plots import extract_single_layer_events
from plotting_tools.trace_io import infer_active_window_us, trim_events_to_window


def _layer_breakdown_us(layer_events: list[dict]) -> dict[str, float]:
    """Per-layer breakdown in microseconds."""
    totals = defaultdict(float)
    for e in layer_events:
        sub = e.get("sub", "other")
        totals[sub] += float(e["dur"])

    out = {
        "AllReduce": totals.get("network_collective", 0.0),
        "Comm P2P": totals.get("network_p2p", 0.0),
        "Attention": totals.get("attention_comp", 0.0),
        "MoE Expert": totals.get("moe_expert", 0.0),
        "MatMul/Proj": totals.get("matmul_gemm", 0.0),
        "Norm/Gate/Routing": (
            totals.get("add_norm_comp", 0.0)
            + totals.get("gate_comp", 0.0)
            + totals.get("moe_routing", 0.0)
            + totals.get("rotary_embedding", 0.0)
            + totals.get("kv_cache_write", 0.0)
        ),
        "Memory": totals.get("host_transfer", 0.0) + totals.get("device_copy", 0.0),
        "Other/Control": totals.get("control", 0.0) + totals.get("other_compute", 0.0),
    }
    return out


def _draw_layer_page(fig, layer_events: list[dict], layer_idx: int) -> None:
    layer_events = sorted(layer_events, key=lambda e: e["ts"])
    t0 = min(e["ts"] for e in layer_events)
    t1 = max(e["end"] for e in layer_events)
    span_us = max(1, t1 - t0)

    # ---- Top: 3-band timeline (compute / comm / idle) ----
    ax1 = fig.add_subplot(2, 1, 1)
    compute_intervals = []
    comm_intervals = []
    active = []
    for e in layer_events:
        start_ms = (e["ts"] - t0) / 1000.0
        dur_ms = max((e["end"] - e["ts"]) / 1000.0, 0.0)
        if e["kind"] == "compute":
            compute_intervals.append((start_ms, dur_ms))
            active.append((e["ts"], e["end"]))
        elif e["kind"] == "comm":
            comm_intervals.append((start_ms, dur_ms))
            active.append((e["ts"], e["end"]))

    active = sorted(active)
    merged = []
    for s, e in active:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    idle_intervals = []
    cur = t0
    for s, e in merged:
        if cur < s:
            idle_intervals.append(((cur - t0) / 1000.0, (s - cur) / 1000.0))
        cur = max(cur, e)
    if cur < t1:
        idle_intervals.append(((cur - t0) / 1000.0, (t1 - cur) / 1000.0))

    if compute_intervals:
        ax1.broken_barh(compute_intervals, (2.0, 0.75), facecolors="#4c78a8", edgecolors="none")
    if comm_intervals:
        ax1.broken_barh(comm_intervals, (1.0, 0.75), facecolors="#f58518", edgecolors="none")
    if idle_intervals:
        ax1.broken_barh(idle_intervals, (0.0, 0.75), facecolors="#b8b8b8", edgecolors="none")
    ax1.set_yticks([0.375, 1.375, 2.375])
    ax1.set_yticklabels(["Idle", "Comm", "Compute"])
    ax1.set_xlim(0, span_us / 1000.0)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax1.set_title(f"Layer {layer_idx} timeline ({span_us/1000.0:.3f} ms)")
    ax1.set_xlabel("Time (ms)")

    # ---- Bottom: stacked breakdown bar ----
    ax2 = fig.add_subplot(2, 1, 2)
    breakdown = _layer_breakdown_us(layer_events)
    wall_us = float(span_us)
    colors = {
        "AllReduce": "#c44e52",
        "Comm P2P": "#dd8452",
        "Attention": "#55a868",
        "MoE Expert": "#4c72b0",
        "MatMul/Proj": "#8172b2",
        "Norm/Gate/Routing": "#ccb974",
        "Memory": "#d5bbcd",
        "Other/Control": "#b0b0b0",
        "Idle/Gap": "#e8e8e8",
    }

    left = 0.0
    non_idle_us = sum(breakdown.values())
    for k in [
        "AllReduce",
        "Comm P2P",
        "Attention",
        "MoE Expert",
        "MatMul/Proj",
        "Norm/Gate/Routing",
        "Memory",
        "Other/Control",
    ]:
        v = breakdown[k]
        if v <= 0:
            continue
        w = v / wall_us
        ax2.barh(0, w, left=left, color=colors[k], edgecolor="white", linewidth=0.4, label=k)
        if w > 0.045:
            ax2.text(left + w / 2, 0, f"{v:.0f}μs", ha="center", va="center", fontsize=7)
        left += w

    idle_us = max(wall_us - non_idle_us, 0.0)
    if idle_us > 0:
        w = idle_us / wall_us
        ax2.barh(0, w, left=left, color=colors["Idle/Gap"], edgecolor="white", linewidth=0.4, label="Idle/Gap")
        if w > 0.045:
            ax2.text(left + w / 2, 0, f"{idle_us:.0f}μs", ha="center", va="center", fontsize=7)

    ax2.set_xlim(0, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel(f"Fraction of layer wall time ({wall_us/1000.0:.3f} ms)")
    ax2.set_title("Per-layer time breakdown")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=7, frameon=False)

    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one multi-page PDF with all layer timelines + per-layer breakdowns."
    )
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-layers", type=int, default=48)
    args = parser.parse_args()

    events = load_nsys_jsonl(args.trace_jsonl, strict=False, report=None)[0]
    active_window = infer_active_window_us(events)
    active_events = trim_events_to_window(events, active_window) if active_window else events
    layers = extract_single_layer_events(active_events, n_layers=args.n_layers)
    if not layers:
        raise RuntimeError("No layers extracted.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        for i, layer_events in enumerate(layers):
            fig = plt.figure(figsize=(13, 7))
            _draw_layer_page(fig, layer_events, i)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {len(layers)} layer pages -> {args.out}")


if __name__ == "__main__":
    main()

