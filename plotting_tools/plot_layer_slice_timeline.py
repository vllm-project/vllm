from __future__ import annotations

import argparse
from pathlib import Path

from plotting_tools.nsys_jsonl import load_nsys_jsonl
from plotting_tools.plots import (
    extract_single_layer_events,
    plot_layer_slice_compute_comm_idle,
)
from plotting_tools.trace_io import infer_active_window_us, trim_events_to_window


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one 3-band timeline (compute/comm/idle) for a small layer slice."
        )
    )
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Index into extracted layer slices (default: 0)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=48,
        help="How many layer slices to extract from active window (default: 48)",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    events = load_nsys_jsonl(args.trace_jsonl, strict=False, report=None)[0]
    active_window = infer_active_window_us(events)
    active_events = (
        trim_events_to_window(events, active_window)
        if active_window is not None
        else events
    )

    layers = extract_single_layer_events(active_events, n_layers=args.n_layers)
    if not layers:
        raise RuntimeError("No layer slices extracted from this trace.")

    idx = min(max(args.layer_index, 0), len(layers) - 1)
    layer_events = layers[idx]
    lt0 = min(e["ts"] for e in layer_events)
    lt1 = max(e["end"] for e in layer_events)
    duration_ms = (lt1 - lt0) / 1000.0
    title = args.title or f"Layer slice {idx} timeline ({duration_ms:.2f} ms)"

    plot_layer_slice_compute_comm_idle(
        layer_events,
        args.out,
        title=title,
        time_origin_us=lt0,
    )


if __name__ == "__main__":
    main()

