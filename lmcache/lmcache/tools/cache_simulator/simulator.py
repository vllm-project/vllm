# SPDX-License-Identifier: Apache-2.0
"""
Cache hit-rate simulator driven by LMCache lookup-hash JSONL logs.

The simulator replays ``MP_LOOKUP`` events recorded by
:class:`~lmcache.v1.mp_observability.subscribers.logging.lookup_hash.LookupHashLoggingSubscriber`.
Each event contains the ordered list of *full-chunk* hashes that were looked up
for a single request, together with the sequence length and chunk size.

**Token cache hit rate** (the primary metric) is defined as::

    token_hit_rate = total_hit_tokens / total_tokens

where:

* ``total_tokens`` = sum of ``seq_len`` across all requests (includes tail tokens
  that do not fill a complete chunk — these are *always* a miss because LMCache
  only caches complete chunks).
* ``total_hit_tokens`` = number of tokens covered by a *continuous prefix* of
  cache-hit chunks at the start of each request, i.e.
  ``hit_prefix_chunks × chunk_size``.

Running the simulator prints a text report **and** saves a multi-panel PNG with
seven statistical charts.

Usage (module mode)::

    python3 -m lmcache.tools.cache_simulator.simulator \\
        -i /path/to/lookup_hashes/ \\
        --cache-capacity-gib 64 \\
        -o stats.png
"""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys
import warnings

# First Party
from lmcache.tools.cache_simulator.lru_cache import LRUCache, LRUCacheFast

# ---------------------------------------------------------------------------
# Dtype → bytes mapping
# ---------------------------------------------------------------------------

_DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}

_GIB = 2**30


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_kv_bytes_per_chunk(event: dict[str, Any]) -> int:
    """
    Compute the number of KV-cache bytes that one chunk occupies.

    The value is derived from the ``shapes`` and ``dtypes`` fields of a single
    lookup event.  Each ``(shape, dtype)`` pair represents one tensor stored
    per chunk (e.g. key and value tensors for all layers); their byte sizes are
    summed.

    Returns 0 if ``shapes`` or ``dtypes`` is empty (caller must handle this).
    """
    shapes = event.get("shapes", [])
    dtypes = event.get("dtypes", [])
    if not shapes or not dtypes:
        return 0
    total = 0
    for shape, dt in zip(shapes, dtypes, strict=False):
        elem_bytes = _DTYPE_BYTES.get(dt, 0)
        if elem_bytes == 0:
            warnings.warn(
                f"Unknown dtype '{dt}' — treating as 0 bytes per element.",
                UserWarning,
                stacklevel=2,
            )
        total += math.prod(shape) * elem_bytes
    return total


def load_lookup_events(
    paths: list[Path],
    model: str | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load and return lookup events from one or more JSONL files or directories.

    Parameters
    ----------
    paths:
        Each element may be a ``.jsonl`` file or a directory.  Directories are
        globbed for ``lookup_hashes_*.jsonl`` files.
    model:
        If given, only events whose ``model_name`` exactly matches this string
        are returned.
    max_samples:
        If given, truncate the final sorted list to this many events.

    Returns
    -------
    list[dict]
        Events sorted by ``timestamp`` ascending.
    """
    all_events: list[dict[str, Any]] = []

    for p in paths:
        files: list[Path]
        if p.is_dir():
            files = sorted(p.glob("lookup_hashes_*.jsonl"))
            if not files:
                warnings.warn(
                    f"Directory '{p}' contains no lookup_hashes_*.jsonl files.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            files = [p]

        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError as exc:
                            warnings.warn(
                                f"{f}:{lineno}: skipping malformed JSON — {exc}",
                                UserWarning,
                                stacklevel=2,
                            )
                            continue
                        if model is not None and event.get("model_name") != model:
                            continue
                        all_events.append(event)
            except OSError as exc:
                warnings.warn(
                    f"Could not open '{f}': {exc}",
                    UserWarning,
                    stacklevel=2,
                )

    all_events.sort(key=lambda e: e.get("timestamp", 0.0))

    if max_samples is not None and max_samples > 0:
        all_events = all_events[:max_samples]

    return all_events


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    events: list[dict[str, Any]],
    cache_capacity_bytes: int,
    kv_bytes_per_chunk: int,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Replay lookup events through an LRU cache and compute token hit-rate
    statistics.

    Parameters
    ----------
    events:
        Lookup events as returned by :func:`load_lookup_events`.
    cache_capacity_bytes:
        Total cache capacity in bytes.
    kv_bytes_per_chunk:
        Bytes consumed by one cached chunk.
    fast:
        If ``True``, use :class:`~lmcache.tools.cache_simulator.lru_cache.LRUCacheFast`
        and skip per-chunk statistics (faster for capacity sweeps).

    Returns
    -------
    dict
        Simulation results (see source for field list).
    """
    if kv_bytes_per_chunk <= 0:
        raise ValueError(
            "kv_bytes_per_chunk must be > 0.  "
            "Either pass --kv-bytes-per-chunk or ensure the JSONL records "
            "contain non-empty 'shapes' and 'dtypes' fields."
        )

    cache_capacity_chunks = max(1, cache_capacity_bytes // kv_bytes_per_chunk)

    cache: LRUCacheFast | LRUCache
    if fast:
        cache = LRUCacheFast(cache_capacity_chunks)
    else:
        cache = LRUCache(cache_capacity_chunks)

    # ── Aggregates ──────────────────────────────────────────────────────────
    total_requests = 0
    total_tokens = 0
    total_hit_tokens = 0

    # ── Per-request (skipped in fast mode) ──────────────────────────────────
    per_request_token_hit_rates: list[float] = []
    hit_prefix_lengths: list[int] = []
    rolling_token_hit_rate: list[float] = []
    input_lengths: list[int] = []

    # ── Chunk-level (skipped in fast mode) ──────────────────────────────────
    chunk_reuse_counts: dict[str, int] = defaultdict(int)
    chunk_last_seen: dict[str, int] = {}
    global_span_distribution: list[int] = []
    cache_position_distribution: list[int] = []
    global_chunk_index = 0

    for event in events:
        hashes: list[str] = event.get("chunk_hashes", [])
        seq_len: int = event.get("seq_len", 0)
        chunk_sz: int = event.get("chunk_size", 1)

        if not hashes and seq_len == 0:
            continue

        # ── Prefix hit count ────────────────────────────────────────────────
        hit_prefix = 0
        for h in hashes:
            if cache.contains(h):
                hit_prefix += 1
            else:
                break

        # ── Token accounting ────────────────────────────────────────────────
        # Tail tokens (seq_len - len(hashes)*chunk_sz) are always a miss.
        hit_tokens = hit_prefix * chunk_sz
        request_tokens = seq_len  # includes tail tokens

        total_requests += 1
        total_tokens += request_tokens
        total_hit_tokens += hit_tokens

        if not fast:
            input_lengths.append(seq_len)
            per_request_token_hit_rates.append(
                hit_tokens / request_tokens if request_tokens > 0 else 0.0
            )
            hit_prefix_lengths.append(hit_prefix)
            rolling_token_hit_rate.append(
                total_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            # Per-hit-chunk statistics
            for i, h in enumerate(hashes[:hit_prefix]):
                chunk_reuse_counts[h] += 1
                if h in chunk_last_seen:
                    global_span_distribution.append(
                        global_chunk_index + i - chunk_last_seen[h]
                    )
                if isinstance(cache, LRUCache):
                    cache_position_distribution.append(cache.position(h))

        # ── Update cache ────────────────────────────────────────────────────
        for i, h in enumerate(hashes):
            if not fast:
                chunk_last_seen[h] = global_chunk_index + i
            if i < hit_prefix:
                cache.access(h)
            else:
                cache.insert(h)

        if not fast:
            global_chunk_index += len(hashes)

    token_hit_rate = total_hit_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        # ── Aggregates ──────────────────────────────────────────────────────
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_hit_tokens": total_hit_tokens,
        "total_miss_tokens": total_tokens - total_hit_tokens,
        "token_hit_rate": token_hit_rate,
        "eviction_count": cache.eviction_count,
        "cache_size_at_end_chunks": len(cache),
        "cache_capacity_chunks": cache_capacity_chunks,
        "cache_capacity_bytes": cache_capacity_bytes,
        "kv_bytes_per_chunk": kv_bytes_per_chunk,
        # ── Per-request ─────────────────────────────────────────────────────
        "per_request_token_hit_rates": per_request_token_hit_rates,
        "hit_prefix_lengths": hit_prefix_lengths,
        "input_lengths": input_lengths,
        "rolling_token_hit_rate": rolling_token_hit_rate,
        # ── Chunk-level ─────────────────────────────────────────────────────
        "chunk_reuse_counts": dict(chunk_reuse_counts),
        "global_span_distribution": global_span_distribution,
        "cache_position_distribution": cache_position_distribution,
    }


# ---------------------------------------------------------------------------
# Reporting — text
# ---------------------------------------------------------------------------


def _percentiles(values: list[float], pcts: list[int]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    result = {}
    for p in pcts:
        idx = min(int(p / 100 * n), n - 1)
        result[f"p{p}"] = s[idx]
    return result


def print_statistics(results: dict[str, Any]) -> None:
    sep = "=" * 60

    gib = results["cache_capacity_bytes"] / _GIB
    print(sep)
    print("Aggregate")
    print(sep)
    print(f"  Requests processed  : {results['total_requests']:,}")
    print(f"  Total tokens        : {results['total_tokens']:,}")
    print(f"  Hit tokens          : {results['total_hit_tokens']:,}")
    print(f"  Miss tokens         : {results['total_miss_tokens']:,}")
    print(f"  Token hit rate      : {results['token_hit_rate']:.2%}")
    print(
        f"  Cache capacity      : {gib:.2f} GiB  "
        f"({results['cache_capacity_chunks']:,} chunks × "
        f"{results['kv_bytes_per_chunk']:,} bytes/chunk)"
    )
    print(
        f"  Cache occupancy     : {results['cache_size_at_end_chunks']:,} / "
        f"{results['cache_capacity_chunks']:,} chunks"
    )

    rates = results["per_request_token_hit_rates"]
    if rates:
        zero_hit = sum(1 for r in rates if r == 0.0)
        full_hit = sum(1 for r in rates if r == 1.0)
        pcts = _percentiles(rates, [25, 50, 75, 90, 99])
        print()
        print(sep)
        print("Stat 1 — Per-request token hit rate distribution")
        print(sep)
        print(
            f"  Requests with 0% hit rate   : "
            f"{zero_hit:,} ({zero_hit / len(rates):.1%})"
        )
        print(
            f"  Requests with 100% hit rate : "
            f"{full_hit:,} ({full_hit / len(rates):.1%})"
        )
        print(f"  Mean                        : {sum(rates) / len(rates):.2%}")
        for k, v in pcts.items():
            print(f"  {k:4s}                        : {v:.2%}")

    lengths = results["hit_prefix_lengths"]
    if lengths:
        pcts_len = _percentiles([float(x) for x in lengths], [25, 50, 75, 90, 99])
        print()
        print(sep)
        print("Stat 2 — Hit prefix length per request (chunks)")
        print(sep)
        print(f"  Mean               : {sum(lengths) / len(lengths):.1f}")
        for k, v in pcts_len.items():
            print(f"  {k:4s}               : {v:.0f}")

    reuse = sorted(results["chunk_reuse_counts"].values())
    if reuse:
        pcts_reuse = _percentiles([float(x) for x in reuse], [25, 50, 75, 90, 99])
        print()
        print(sep)
        print("Stat 3 — Chunk reuse count distribution")
        print(sep)
        print(f"  Unique chunks hit at least once : {len(reuse):,}")
        print(f"  Mean reuse count                : {sum(reuse) / len(reuse):.1f}")
        print(f"  Max reuse count                 : {reuse[-1]:,}")
        for k, v in pcts_reuse.items():
            print(f"  {k:4s}                            : {v:.0f}")

    rolling = results["rolling_token_hit_rate"]
    if rolling:
        print()
        print(sep)
        print("Stat 4 — Rolling (cumulative) token hit rate over time")
        print(sep)
        n = len(rolling)
        for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            idx = max(0, min(int(n * frac) - 1, n - 1))
            print(f"  After request {idx + 1:>6,} : {rolling[idx]:.2%}")

    print()
    print(sep)
    print("Stat 5 — Evictions")
    print(sep)
    print(f"  Total evictions    : {results['eviction_count']:,}")

    spans = results["global_span_distribution"]
    if spans:
        pcts_span = _percentiles([float(x) for x in spans], [25, 50, 75, 90, 99])
        print()
        print(sep)
        print("Stat 6 — Global span distribution (chunks between last store and hit)")
        print(sep)
        print(f"  Total hit chunks   : {len(spans):,}")
        print(f"  Mean span          : {sum(spans) / len(spans):.1f}")
        print(f"  Max span           : {max(spans):,}")
        for k, v in pcts_span.items():
            print(f"  {k:4s}               : {v:.0f}")

    positions = results["cache_position_distribution"]
    if positions:
        pcts_pos = _percentiles([float(x) for x in positions], [25, 50, 75, 90, 99])
        print()
        print(sep)
        print("Stat 7 — Cache position at hit (0 = MRU, max = LRU)")
        print(sep)
        print(f"  Mean position      : {sum(positions) / len(positions):.1f}")
        print(f"  Max position       : {max(positions):,}")
        for k, v in pcts_pos.items():
            print(f"  {k:4s}               : {v:.0f}")

    print(sep)


# ---------------------------------------------------------------------------
# Reporting — charts
# ---------------------------------------------------------------------------


def plot_statistics(
    results: dict[str, Any], events: list[dict[str, Any]], output: str
) -> None:
    """
    Render and save a 2×4 multi-panel figure with seven statistical charts.

    Parameters
    ----------
    results:
        Output of :func:`simulate` with ``fast=False``.
    events:
        The event list used to produce *results* (used for chunk_size label).
    output:
        Output file path (PNG).
    """
    cap_gib = results["cache_capacity_bytes"] / _GIB
    chunk_size = events[0].get("chunk_size", "?") if events else "?"
    n_req = results["total_requests"]

    per_request_hit_rates = [r * 100 for r in results["per_request_token_hit_rates"]]
    hit_prefix_lengths = results["hit_prefix_lengths"]
    reuse_counts = sorted(results["chunk_reuse_counts"].values())
    rolling = [r * 100 for r in results["rolling_token_hit_rate"]]
    input_lengths = results["input_lengths"]
    global_spans = results["global_span_distribution"]
    cache_positions = results["cache_position_distribution"]

    # Third Party
    import matplotlib.pyplot as plt  # noqa: PLC0415 — lazy import to avoid hard dependency

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"Cache simulation statistics  "
        f"(chunk_size={chunk_size} tokens,  capacity={cap_gib:.1f} GiB,  "
        f"{n_req:,} requests,  token hit rate={results['token_hit_rate']:.2%})",
        fontsize=12,
    )

    # ------------------------------------------------------------------
    # Plot 1 — Per-request token hit rate (non-zero requests only)
    # Two small pies: left = requests hit/miss, right = tokens hit/miss
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    nonzero = [r for r in per_request_hit_rates if r > 0]
    n_zero = len(per_request_hit_rates) - len(nonzero)
    ax.hist(nonzero, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Token hit rate (%) — zero-hit requests excluded")
    ax.set_ylabel("Number of requests")
    ax.set_title("1. Per-request token hit rate")

    # Left pie — requests
    ax_pie = ax.inset_axes([0.01, 0.52, 0.24, 0.42])
    ax_pie.patch.set_alpha(0)
    wedges, _, _ = ax_pie.pie(
        [len(nonzero), n_zero],
        labels=["hit", "miss"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 5},
        colors=["#4C72B0", "#DD8452"],
    )
    for w in wedges:
        w.set_alpha(0.6)
    ax_pie.set_title("requests", fontsize=5, pad=2)
    ax_pie.text(
        0.5,
        -0.08,
        "Fraction of requests\nwith ≥1 chunk hit",
        transform=ax_pie.transAxes,
        fontsize=5,
        ha="center",
        va="top",
        color="dimgray",
    )

    # Right pie — tokens
    ax_pie2 = ax.inset_axes([0.27, 0.52, 0.24, 0.42])
    ax_pie2.patch.set_alpha(0)
    wedges2, _, _ = ax_pie2.pie(
        [results["total_hit_tokens"], results["total_miss_tokens"]],
        labels=["hit", "miss"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 5},
        colors=["#4C72B0", "#DD8452"],
    )
    for w in wedges2:
        w.set_alpha(0.6)
    ax_pie2.set_title("tokens", fontsize=5, pad=2)
    ax_pie2.text(
        0.5,
        -0.08,
        "Fraction of tokens\nserved from cache",
        transform=ax_pie2.transAxes,
        fontsize=5,
        ha="center",
        va="top",
        color="dimgray",
    )

    # ------------------------------------------------------------------
    # Plot 1b — Zoom into 97–100% hit rate
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    n_full = sum(1 for r in per_request_hit_rates if r == 100)
    high = [r for r in nonzero if r >= 97]
    ax.hist(high, bins=20, edgecolor="black", linewidth=0.4)
    ax.set_xlim(97, 100)
    ax.set_xlabel("Token hit rate (%) — 97–100% zoom")
    ax.set_ylabel("Number of requests")
    ax.set_title("1b. Per-request token hit rate (97–100%)")
    ax.text(
        0.03,
        0.95,
        f"100% hit: {n_full:,} requests",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    # ------------------------------------------------------------------
    # Plot 2 — Hit prefix length per request (clean histogram, no pie)
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    nonzero_prefix = [n for n in hit_prefix_lengths if n > 0]
    ax.hist(nonzero_prefix, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Hit prefix length (chunks) — zero-hit requests excluded")
    ax.set_ylabel("Number of requests")
    ax.set_title("2. Hit prefix length per request")

    # Plot 3 — Chunk reuse count
    # ------------------------------------------------------------------
    ax = axes[0, 3]
    if reuse_counts:
        cap = min(max(reuse_counts), 100)
        ax.hist(
            [r for r in reuse_counts if r <= cap],
            bins=range(1, cap + 2),
            edgecolor="black",
            linewidth=0.4,
        )
        if max(reuse_counts) > cap:
            n_above = sum(1 for r in reuse_counts if r > cap)
            pct_above = n_above / len(reuse_counts) * 100
            ax.text(
                0.97,
                0.95,
                f"max={max(reuse_counts):,}\n"
                f"{n_above:,} chunks ({pct_above:.1f}%) above cap",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )
    ax.set_xlabel("Times a chunk was hit (capped at 100)")
    ax.set_ylabel("Number of unique chunks")
    ax.set_title("3. Chunk reuse count")

    # ------------------------------------------------------------------
    # Plot 4 — Rolling token hit rate over time
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(range(1, len(rolling) + 1), rolling, linewidth=1.5)
    ax.set_xlabel("Request index")
    ax.set_ylabel("Cumulative token hit rate (%)")
    ax.set_title("4. Rolling token hit rate over time")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------
    # Plot 5 — Input length distribution
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.hist(input_lengths, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Input length (tokens / seq_len)")
    ax.set_ylabel("Number of requests")
    ax.set_title("5. Input length per request")

    # ------------------------------------------------------------------
    # Plot 6 — Global span distribution
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    if global_spans:
        ax.hist(global_spans, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Global span (chunks between last store and hit)")
    ax.set_ylabel("Number of hit chunks")
    ax.set_title("6. Global span distribution")

    # ------------------------------------------------------------------
    # Plot 7 — Cache position at hit time
    # ------------------------------------------------------------------
    ax = axes[1, 3]
    if cache_positions:
        ax.hist(cache_positions, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Cache position (0 = MRU, max = LRU)")
    ax.set_ylabel("Number of hit chunks")
    ax.set_title("7. Cache position at hit")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"\nStats plot saved to '{output}'")


# ---------------------------------------------------------------------------
# CLI helpers — shared between the module entry point and lmcache tool
# ---------------------------------------------------------------------------


def add_simulate_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all ``simulate`` CLI flags onto *parser*.

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
        "--cache-capacity-gib",
        type=float,
        required=True,
        metavar="GiB",
        help="Cache capacity in gibibytes",
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
        "--model",
        default=None,
        metavar="NAME",
        help="Filter events by model_name (exact match)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cache_stats.png",
        metavar="FILE",
        help="Output image path (default: cache_stats.png)",
    )


def run_simulate(args: argparse.Namespace) -> None:
    """Execute the simulate workflow from a parsed argument namespace.

    Loads events, resolves ``kv_bytes_per_chunk``, runs the simulator, prints
    a text report, and saves a statistics PNG.  Called by both the module
    ``main()`` and by :class:`~lmcache.cli.commands.tool.ToolCommand`.

    Args:
        args: Parsed CLI arguments.  Must have the attributes registered by
            :func:`add_simulate_arguments`.
    """
    paths = [Path(p) for p in args.input]
    print(f"Loading lookup events from {[str(p) for p in paths]} …")
    events = load_lookup_events(paths, model=args.model, max_samples=args.max_samples)
    print(f"Loaded {len(events):,} event(s)")

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

    capacity_bytes = int(args.cache_capacity_gib * _GIB)

    print("\nSimulation parameters:")
    print(
        f"  Cache capacity     : {args.cache_capacity_gib:.2f} GiB "
        f"({capacity_bytes:,} bytes)"
    )
    print(f"  KV bytes/chunk     : {kv_bpc:,}")
    chunk_sz = events[0].get("chunk_size", "?")
    print(f"  Chunk size         : {chunk_sz} tokens")
    if args.model:
        print(f"  Model filter       : {args.model}")
    print()

    results = simulate(events, capacity_bytes, kv_bpc)
    print_statistics(results)
    plot_statistics(results, events, args.output)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m lmcache.tools.cache_simulator.simulator``.

    Parses command-line arguments and delegates to :func:`run_simulate`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate LRU token cache hit rate from lookup-hash JSONL logs. "
            "Prints a text report and saves a multi-panel statistics chart."
        )
    )
    add_simulate_arguments(parser)
    args = parser.parse_args()
    run_simulate(args)


if __name__ == "__main__":
    main()
