# SPDX-License-Identifier: Apache-2.0
"""
Plot token cache hit rate vs cache capacity (GiB).

Sweeps a logarithmically-spaced range of cache capacities, runs the simulator
at each point, and produces a matplotlib figure showing how token hit rate
scales with available memory.

Usage::

    python -m lmcache.tools.cache_simulator.plot_hit_rate \\
        -i /path/to/lookup_hashes/ \\
        --min-capacity-gib 1 \\
        --max-capacity-gib 512 \\
        --points 30 \\
        -o hit_rate_vs_capacity.png
"""

# Standard
from pathlib import Path
import argparse
import math
import sys

# First Party
from lmcache.tools.cache_simulator.simulator import (
    compute_kv_bytes_per_chunk,
    load_lookup_events,
    simulate,
)

_GIB = 2**30


def capacity_range_bytes(
    min_gib: float,
    max_gib: float,
    num_points: int,
) -> list[int]:
    """
    Return *num_points* byte capacities log-spaced between *min_gib* and
    *max_gib* GiB.
    """
    log_min = math.log10(min_gib * _GIB)
    log_max = math.log10(max_gib * _GIB)
    step = (log_max - log_min) / max(num_points - 1, 1)
    return sorted({round(10 ** (log_min + i * step)) for i in range(num_points)})


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all ``sweep`` CLI flags onto *parser*.

    Called by both the module ``main()`` and by
    :class:`~lmcache.cli.commands.tool.ToolCommand` so that flag definitions
    live in exactly one place.

    Args:
        parser: The ``ArgumentParser`` (or sub-parser) to add flags to.
    """
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more lookup-hash JSONL files or directories",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of events to process (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Filter events by model_name (exact match)",
    )
    parser.add_argument(
        "--min-capacity-gib",
        type=float,
        default=0.5,
        metavar="GiB",
        help="Minimum cache capacity to sweep (default: 0.5 GiB)",
    )
    parser.add_argument(
        "--max-capacity-gib",
        type=float,
        default=500.0,
        metavar="GiB",
        help="Maximum cache capacity to sweep (default: 500 GiB)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=30,
        metavar="N",
        help="Number of log-spaced capacity samples (default: 30)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear x-axis (default: log scale)",
    )
    parser.add_argument(
        "--kv-bytes-per-chunk",
        type=int,
        default=None,
        metavar="BYTES",
        help=(
            "Bytes consumed by one cached chunk.  "
            "Auto-computed from the first event's shapes/dtypes if omitted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="hit_rate_vs_capacity.png",
        metavar="FILE",
        help="Output image path (default: hit_rate_vs_capacity.png)",
    )


def run_sweep(args: argparse.Namespace) -> None:
    """Execute the sweep workflow from a parsed argument namespace.

    Loads events, resolves ``kv_bytes_per_chunk``, sweeps across a log-spaced
    range of cache capacities, prints a results table, and saves a hit-rate vs
    capacity PNG.  Called by both the module ``main()`` and by
    :class:`~lmcache.cli.commands.tool.ToolCommand`.

    Args:
        args: Parsed CLI arguments.  Must have the attributes registered by
            :func:`add_sweep_arguments`.
    """
    paths = [Path(p) for p in args.input]
    print(f"Loading lookup events from {[str(p) for p in paths]} …")
    events = load_lookup_events(paths, model=args.model, max_samples=args.max_samples)
    print(f"Loaded {len(events):,} event(s)\n")

    if not events:
        print("No events to process.")
        sys.exit(0)

    kv_bpc = args.kv_bytes_per_chunk
    if kv_bpc is None:
        kv_bpc = compute_kv_bytes_per_chunk(events[0])
        if kv_bpc == 0:
            print(
                "Error: could not determine kv_bytes_per_chunk from the first event "
                "(shapes/dtypes are empty).  Pass --kv-bytes-per-chunk explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected kv_bytes_per_chunk = {kv_bpc:,} bytes")

    chunk_size = events[0].get("chunk_size", "?")
    model_label = args.model or "all models"

    capacities_bytes = capacity_range_bytes(
        args.min_capacity_gib, args.max_capacity_gib, args.points
    )
    hit_rates: list[float] = []

    scale_label = "linear" if args.linear else "log"
    print(
        f"Sweeping {len(capacities_bytes)} capacity values "
        f"({args.min_capacity_gib:.2f} – {args.max_capacity_gib:.2f} GiB), "
        f"chunk_size = {chunk_size} tokens, model = {model_label}\n"
    )
    print(f"{'Capacity (GiB)':>18}  {'Hit rate':>10}")
    print("-" * 32)

    for cap_bytes in capacities_bytes:
        cap_gib = cap_bytes / _GIB
        res = simulate(events, cap_bytes, kv_bpc, fast=True)
        rate = res["token_hit_rate"]
        hit_rates.append(rate)
        print(f"{cap_gib:>18.3f}  {rate:>9.2%}")

    # ── Plot ────────────────────────────────────────────────────────────────
    x_values = [c / _GIB for c in capacities_bytes]

    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415 — lazy import to avoid hard dependency

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        x_values,
        [r * 100 for r in hit_rates],
        marker="o",
        linewidth=2,
        markersize=4,
    )

    if not args.linear:
        ax.set_xscale("log")

    ax.set_xlabel("Cache capacity (GiB)", fontsize=12)
    ax.set_ylabel("Token hit rate (%)", fontsize=12)
    ax.set_title(
        f"Token cache hit rate vs capacity\n"
        f"(chunk_size = {chunk_size} tokens, {len(events):,} requests, "
        f"model = {model_label}, {scale_label} scale)",
        fontsize=11,
    )
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to '{args.output}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot token cache hit rate vs cache capacity from lookup-hash JSONL logs"
        )
    )
    add_sweep_arguments(parser)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
