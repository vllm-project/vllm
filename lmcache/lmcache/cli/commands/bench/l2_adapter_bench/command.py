# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench l2`` subcommand implementation.

This module provides argument registration via :func:`add_l2_arguments`
and the execution orchestrator :func:`run_l2_adapter_bench` for the L2
adapter benchmark.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import os
import sys

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand
    from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_l2_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench l2`` arguments to *parser*.

    Args:
        parser: The ``ArgumentParser`` for the L2 bench subcommand.
    """

    parser.add_argument(
        "--l2-adapter",
        dest="l2_adapter",
        action="append",
        default=None,
        type=str,
        metavar="JSON",
        help=(
            'L2 adapter spec as JSON with a "type" field and adapter-'
            'specific configs, e.g. \'{"type":"fs","base_path":"/tmp/'
            "bench\"}'. If not provided, falls back to L2_ADAPTER_JSON "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=32,
        help="Keys per submit (default: 32).",
    )
    parser.add_argument(
        "--in-flight",
        type=int,
        default=1,
        help=(
            "In-flight submits per round. Each round issues this many "
            "submits sequentially from a single producer thread, then "
            "waits for all of them (default: 1)."
        ),
    )
    parser.add_argument(
        "--data-size-kb",
        type=int,
        default=256,
        help="Data size per key in KB (default: 256).",
    )
    parser.add_argument(
        "--l1-align-bytes",
        type=int,
        default=1,
        help=(
            "Alignment in bytes for benchmark L1 buffers. "
            "Use 4096 when benchmarking O_DIRECT backends. Default: 1."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Measurement rounds per operation (default: 1).",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Warmup rounds before measurement (default: 1).",
    )
    parser.add_argument(
        "--lookup-max-hit-rate",
        type=float,
        default=0.0,
        help=(
            "Upper bound on the lookup hit rate, in [0, 1]. The "
            "benchmark will request floor(N * rate) keys from the "
            "potentially-existing range and (N - hit) keys from a "
            "guaranteed-non-existent range, where N is the total "
            "number of lookup keys (rounds * in_flight * num_keys). "
            "The actual hit rate may be lower if those keys were "
            "never stored. Default: 0.0."
        ),
    )
    # Round-trip verification is OFF by default because it needs both
    # store and load object batches resident at the same time.
    # Use --no-skip-verify to enable verification.
    parser.add_argument(
        "--skip-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip round-trip data verification (default). "
            "Pass --no-skip-verify to enable verification."
        ),
    )
    parser.add_argument(
        "--only",
        choices=["lookup", "store", "load"],
        default=None,
        help="Run only the specified operation (default: run all).",
    )
    parser.add_argument(
        "--flamegraph",
        choices=["on", "off"],
        default="off",
        help=(
            "Capture a flame graph of the measured phases if turned on. The benchmark "
            "profiles itself and renders an SVG"
        ),
    )
    parser.add_argument(
        "--flamegraph-mode",
        default="on-cpu",
        metavar="MODE[,MODE...]",
        help=(
            "What to sample when profiling is on (default: on-cpu). Pass "
            "several comma-separated to profile one benchmark run per mode. "
            "Modes: on-cpu, off-cpu, wakeup, offwake (perf/bcc), wall, gil "
            "(py-spy). See the 'lmcache tool flamegraph' docs for detail."
        ),
    )
    parser.add_argument(
        "--flamegraph-output",
        default="",
        metavar="PATH",
        help=(
            "SVG output path for '--flamegraph on'. Default: "
            "/tmp/lmcache_bench_flames/<adapter>.<mode>.svg."
        ),
    )
    parser.add_argument(
        "--flamegraph-scripts-dir",
        default="",
        metavar="DIR",
        help=(
            "Directory with the FlameGraph scripts (flamegraph.pl, "
            "stackcollapse-perf.pl); default ~/FlameGraph (cloned there on "
            "first use). Unused by --flamegraph-mode wall / gil, which "
            "render their own SVG."
        ),
    )


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_l2_adapter_bench(command: "BaseCommand", args: argparse.Namespace) -> None:
    """Run the L2 adapter benchmark.

    Args:
        command: The owning :class:`BaseCommand` instance, used only
            to obtain a configured :class:`Metrics` object via
            ``command.create_metrics``.
        args: Parsed CLI arguments from the ``bench l2`` subparser.
    """
    # Lazy imports: keep CLI loadable without torch / native deps.
    # First Party
    from lmcache.cli.commands.bench.l2_adapter_bench.data import (
        create_l1_memory_desc,
        make_aligned_tensor,
        make_memory_objects,
        make_object_keys,
        verify_round_trip,
    )
    from lmcache.cli.commands.bench.l2_adapter_bench.runner import (
        bench_load,
        bench_lookup,
        bench_store,
    )
    from lmcache.cli.profiling import (
        PY_SPY_MODES,
        FlameProfiler,
        ProfileError,
        check_profiling_deps,
        default_output_path,
        resolve_flamegraph_dir,
    )
    from lmcache.v1.distributed.l2_adapters import create_l2_adapter
    from lmcache.v1.distributed.l2_adapters.config import (
        parse_args_to_l2_adapters_config,
    )

    kb = 1024
    mb = 1024 * 1024
    data_size = args.data_size_kb * kb
    l1_align_bytes = int(args.l1_align_bytes)
    if l1_align_bytes <= 0:
        print("Error: --l1-align-bytes must be positive", file=sys.stderr)
        sys.exit(2)
    if data_size % l1_align_bytes != 0:
        print(
            "Error: --data-size-kb must produce a payload size that is "
            "a multiple of --l1-align-bytes",
            file=sys.stderr,
        )
        sys.exit(2)
    in_flight = args.in_flight
    num_keys = args.num_keys
    rounds = args.rounds
    warmup = args.warmup_rounds
    total_rounds = warmup + rounds
    max_hit_rate = max(0.0, min(1.0, args.lookup_max_hit_rate))
    quiet = getattr(args, "quiet", False)

    # Keys per round (one in-flight wave) and total measured keys per
    # operation. Warmup rounds extend the consumed idx range.
    keys_per_round = in_flight * num_keys
    total_run_keys = total_rounds * keys_per_round  # warmup + measured

    def log(msg: str) -> None:
        # Per-round progress log; suppressed by --quiet.
        if not quiet:
            print(msg)

    # Resolve L2 adapter JSON: CLI arg takes priority, then env var
    l2_adapter_specs = args.l2_adapter
    if not l2_adapter_specs:
        env_json = os.environ.get("L2_ADAPTER_JSON")
        if env_json:
            l2_adapter_specs = [env_json]
        else:
            print(
                "Error: No L2 adapter configuration provided.\n"
                "Use --l2-adapter JSON or set L2_ADAPTER_JSON "
                "environment variable.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Parse adapter config using the standard LMCache mechanism
    ns = argparse.Namespace(l2_adapter=l2_adapter_specs)
    try:
        l2_cfg = parse_args_to_l2_adapters_config(ns)
    except (ValueError, KeyError) as e:
        print(f"Error parsing L2 adapter config: {e}", file=sys.stderr)
        sys.exit(2)

    if not l2_cfg.adapters:
        print("Error: no L2 adapter configs parsed", file=sys.stderr)
        sys.exit(2)

    # Use the first adapter config for benchmarking
    adapter_cfg = l2_cfg.adapters[0]

    # Backing L1 memory buffer for adapters that need an L1 desc.
    # Sized for one in-flight wave of store + load buffers.
    l1_buffer = make_aligned_tensor(2 * keys_per_round * data_size, l1_align_bytes)
    l1_memory_desc = create_l1_memory_desc(l1_buffer, align_bytes=l1_align_bytes)

    # Resolve and validate the flame-graph toolchain up front, before any
    # adapter (and its worker threads) is created. A user who explicitly
    # passes ``--flamegraph on`` gets a fast, actionable failure if a
    # required tool is missing, rather than a benchmark that silently runs
    # unprofiled. When ``--flamegraph`` is off (default), none of this runs
    # and the benchmark behaves exactly as before.
    flamegraph_on = getattr(args, "flamegraph", "off") == "on"
    flamegraph_dir = ""
    if flamegraph_on:
        try:
            check_profiling_deps(args.flamegraph_mode)
            # py-spy renders its own SVG; only the perf/bcc modes need
            # the FlameGraph scripts, so do not clone them otherwise.
            if args.flamegraph_mode not in PY_SPY_MODES:
                flamegraph_dir = resolve_flamegraph_dir(
                    args.flamegraph_scripts_dir, log
                )
        except ProfileError as e:
            print(
                "Error: --flamegraph on was requested but the profiling "
                f"toolchain is unavailable:\n  {e}",
                file=sys.stderr,
            )
            sys.exit(2)

    log("\n[Init] Creating adapter...")
    try:
        adapter = create_l2_adapter(adapter_cfg, l1_memory_desc=l1_memory_desc)
        log(f"[Init] Adapter created successfully ({type(adapter).__name__}).\n")
    except Exception as e:
        print(f"[Init] Failed to create adapter: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional self-profiling: record a flame graph of the measured
    # phases. Built here, after the adapter (and its worker threads)
    # exist, so the recorder attaches to a fully-started process. The
    # toolchain was already validated above, so a ProfileError here is a
    # late failure (e.g. the output directory is not writable); since the
    # user explicitly requested a flame graph we fail loudly rather than
    # downgrade, closing the adapter we just opened first.
    profiler: FlameProfiler | None = None
    if flamegraph_on:
        adapter_name = type(adapter).__name__
        try:
            profiler = FlameProfiler(
                mode=args.flamegraph_mode,
                output=(
                    args.flamegraph_output
                    or default_output_path(adapter_name, args.flamegraph_mode)
                ),
                flamegraph_dir=flamegraph_dir,
                pid=os.getpid(),
                title=f"{args.flamegraph_mode} ({adapter_name})",
            )
        except ProfileError as e:
            print(
                f"Error: cannot start flame-graph profiling: {e}",
                file=sys.stderr,
            )
            try:
                adapter.close()
            except Exception:
                pass
            sys.exit(2)

    # ------------------------------------------------------------------
    # Idx layout
    # ------------------------------------------------------------------
    # All ops live in the same idx universe so that ``--only store``
    # followed by ``--only load`` (or lookup) with the same flags hits
    # the exact same keys.
    #
    # Round r (0-indexed, warmup rounds first) consumes the idx slice
    #   [r * keys_per_round, (r+1) * keys_per_round)
    # split into ``in_flight`` contiguous batches of ``num_keys`` each.
    #
    # Lookup additionally splits each round into a hit-portion (drawn
    # from the same idx range as store/load) and a miss-portion drawn
    # from a guaranteed-non-existent range starting at
    # ``total_run_keys``.
    # ------------------------------------------------------------------

    def _build_round_keys(r: int) -> list[list]:
        """Build per-submit key batches for round *r* (store/load)."""
        base = r * keys_per_round
        return [
            make_object_keys(num_keys, key_offset=base + i * num_keys)
            for i in range(in_flight)
        ]

    def _build_round_objs(base_offset: int, fill_offset: int = 0) -> list[list]:
        """Build per-submit object batches backed by the registered L1 buffer.

        Some adapters register the L1 buffer passed through ``L1MemoryDesc``
        during initialization. The benchmark objects must therefore be views
        into that same buffer rather than independent tensors allocated
        elsewhere.

        ``fill_offset`` lets load buffers start with a pattern that differs
        from store buffers, so round-trip verification catches silent no-op
        loads that nevertheless report success.
        """
        return [
            make_memory_objects(
                l1_buffer,
                num_keys,
                data_size,
                base_offset + i * num_keys * data_size,
                fill_offset=fill_offset,
            )
            for i in range(in_flight)
        ]

    # Lookup hit/miss split per round.
    per_round_hit = int(keys_per_round * max_hit_rate)
    per_round_miss = keys_per_round - per_round_hit
    # Total expected hit count over measured rounds only.
    expected_hit_count = per_round_hit * rounds
    # Origin of the guaranteed-miss idx range.
    miss_origin = total_run_keys

    def _build_lookup_round_keys(r: int) -> list[list]:
        """Build per-submit lookup key batches for round *r*.

        Hit slice for round r:
          [r * per_round_hit, (r+1) * per_round_hit)
        Miss slice for round r (disjoint from any store/load idx):
          [miss_origin + r * per_round_miss,
           miss_origin + (r+1) * per_round_miss)

        The combined ``keys_per_round`` keys are concatenated then
        split into ``in_flight`` chunks of ``num_keys`` each.
        """
        hit_base = r * per_round_hit
        miss_base = miss_origin + r * per_round_miss
        keys_round: list = []
        keys_round.extend(make_object_keys(per_round_hit, key_offset=hit_base))
        keys_round.extend(make_object_keys(per_round_miss, key_offset=miss_base))
        # Split into in_flight equal-sized batches of num_keys.
        return [keys_round[i * num_keys : (i + 1) * num_keys] for i in range(in_flight)]

    # Per-direction object batches for store / load.
    #
    # Allocation strategy:
    # * Lazy: only allocate when the corresponding direction is
    #   actually exercised. With ``--only store`` we never touch
    #   load buffers (and vice versa), saving
    #   ``in_flight * num_keys * data_size`` bytes of host memory.
    # * Cross-round reuse: once allocated, the same batches are
    #   fed into every round; only the keys change per round. The
    #   L2 adapter does not care about object identity across
    #   rounds, and re-allocating these buffers each round would
    #   just be wasted work.
    store_obj_batches: list[list] | None = None
    load_obj_batches: list[list] | None = None

    def _store_objs(_r: int) -> list[list]:
        nonlocal store_obj_batches
        if store_obj_batches is None:
            store_obj_batches = _build_round_objs(0)
        return store_obj_batches

    def _load_objs(_r: int) -> list[list]:
        nonlocal load_obj_batches
        if load_obj_batches is None:
            load_obj_batches = _build_round_objs(
                keys_per_round * data_size,
                fill_offset=1,
            )
        return load_obj_batches

    results: list = []
    failed = False

    # Track the very last measured store round so we can verify it
    # against the matching load round (round-trip integrity check).
    last_store_round_keys: list[list] | None = None
    last_load_round_keys: list[list] | None = None

    try:
        if profiler is not None:
            # Pre-build the payload buffers before the recorder starts so
            # the one-time tensor allocation + fill (benchmark harness work,
            # not the adapter) is kept out of the flame graph. The batches
            # are reused across rounds, so this is the only build; warming
            # it here keeps the recording focused on adapter I/O.
            if args.only is None or args.only == "store":
                _store_objs(0)
            if args.only is None or args.only == "load":
                _load_objs(0)
            profiler.start(log)

        # ---- Store ----
        if args.only is None or args.only == "store":
            log(f"[Store] Running {warmup} warmup + {rounds} measurement rounds...")
            all_store = bench_store(
                adapter,
                in_flight=in_flight,
                num_keys=num_keys,
                data_size=data_size,
                rounds=total_rounds,
                keys_for_round=_build_round_keys,
                objs_for_round=_store_objs,
                log=log,
            )
            results.append(_strip_warmup(all_store, warmup))
            # Last measured store round is total_rounds - 1.
            last_store_round_keys = _build_round_keys(total_rounds - 1)
            log("")

        # ---- Lookup ----
        if args.only is None or args.only == "lookup":
            log(f"[Lookup] Running {warmup} warmup + {rounds} measurement rounds...")
            all_lookup = bench_lookup(
                adapter,
                in_flight=in_flight,
                num_keys=num_keys,
                rounds=total_rounds,
                keys_for_round=_build_lookup_round_keys,
                log=log,
                expected_max_hit_rate=max_hit_rate,
                expected_hit_count=expected_hit_count,
            )
            results.append(_strip_warmup(all_lookup, warmup))
            log("")

        # ---- Load ----
        if args.only is None or args.only == "load":
            log(f"[Load] Running {warmup} warmup + {rounds} measurement rounds...")
            all_load = bench_load(
                adapter,
                in_flight=in_flight,
                num_keys=num_keys,
                data_size=data_size,
                rounds=total_rounds,
                keys_for_round=_build_round_keys,
                objs_for_round=_load_objs,
                log=log,
            )
            results.append(_strip_warmup(all_load, warmup))
            last_load_round_keys = _build_round_keys(total_rounds - 1)
            log("")

        # Stop profiling before verification / summary so the flame
        # graph reflects only the measured store/lookup/load work.
        if profiler is not None:
            profiler.stop(log)

        # ---- Round-trip verification (last measured round only) ----
        if (
            not args.skip_verify
            and store_obj_batches is not None
            and load_obj_batches is not None
            and last_store_round_keys is not None
            and last_load_round_keys is not None
        ):
            # Sanity: store and load used the same key idx range for
            # the last measured round, and load buffers now hold what
            # the adapter returned. Compare against the byte pattern
            # written by the store object batch (i & 0xFF, where i is
            # position within the batch).
            log(
                "[Verify] Checking store -> load data integrity for last "
                "measured round..."
            )
            flat_keys = [k for kl in last_load_round_keys for k in kl]
            flat_store = [o for ol in store_obj_batches for o in ol]
            flat_load = [o for ol in load_obj_batches for o in ol]
            ok = verify_round_trip(flat_keys, flat_store, flat_load, log)
            if not ok:
                failed = True
            log("")

        # ---- Summary via metrics system ----
        _emit_l2_adapter_metrics(
            command=command,
            args=args,
            l2_adapter_json=l2_adapter_specs[0],
            keys_per_round=keys_per_round,
            data_per_round_mb=(keys_per_round * data_size) / mb,
            results=results,
        )
    finally:
        # Idempotent: a no-op if profiling already stopped on the normal
        # path; tears the recorder down if a phase raised.
        if profiler is not None:
            profiler.stop(log)
        log("[Cleanup] Closing adapter...")
        try:
            adapter.close()
        except Exception as e:
            print(f"[Cleanup] adapter.close() failed: {e}", file=sys.stderr)
        log("[Cleanup] Done.")

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_warmup(result: "BenchResult", warmup: int) -> "BenchResult":
    """Drop the leading *warmup* rounds from a BenchResult."""
    # First Party
    from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult

    # Adjust the expected hit count proportionally for the kept rounds.
    kept_rounds = max(0, len(result.round_durations) - warmup)
    total_rounds = max(1, len(result.round_durations))
    scaled_expected_hit = int(result.expected_hit_count * kept_rounds / total_rounds)

    return BenchResult(
        operation=result.operation,
        in_flight=result.in_flight,
        num_keys=result.num_keys,
        data_size_bytes=result.data_size_bytes,
        round_durations=result.round_durations[warmup:],
        success_counts=result.success_counts[warmup:],
        expected_max_hit_rate=result.expected_max_hit_rate,
        expected_hit_count=scaled_expected_hit,
    )


def _emit_l2_adapter_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    l2_adapter_json: str,
    keys_per_round: int,
    data_per_round_mb: float,
    results: list,
) -> None:
    """Emit L2 adapter benchmark summary using the CLI metrics system."""
    title = "L2 Adapter Benchmark Result"
    metrics = command.create_metrics(title, args, width=64)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("l2_adapter_json", "L2 adapter JSON", l2_adapter_json)
    cfg_section.add("num_keys", "Keys / submit", args.num_keys)
    cfg_section.add("in_flight", "In-flight / round", args.in_flight)
    cfg_section.add("keys_per_round", "Keys / round", keys_per_round)
    cfg_section.add(
        "data_size_kb",
        "Data size / key (KB)",
        args.data_size_kb,
    )
    cfg_section.add(
        "data_per_round_mb",
        "Data / round (MB)",
        round(data_per_round_mb, 2),
    )
    cfg_section.add("measurement_rounds", "Measurement rounds", args.rounds)
    cfg_section.add("warmup_rounds", "Warmup rounds", args.warmup_rounds)
    # Only meaningful when lookup is actually executed; matches the
    # original banner log behaviour.
    if args.only is None or args.only == "lookup":
        cfg_section.add(
            "lookup_max_hit_rate",
            "Lookup max hit rate",
            round(args.lookup_max_hit_rate, 4),
        )

    for idx, r in enumerate(results):
        section_id = f"op_{idx}"
        section = metrics.add_section(section_id, r.operation)
        section.add("operation", "Operation", r.operation)
        section.add("rounds", "Rounds", len(r.round_durations))
        section.add("keys_per_round", "Keys / round", r.keys_per_round)
        section.add("total_keys", "Total keys", r.total_keys)
        section.add("total_success", "Total success", r.total_success)
        section.add(
            "duration_avg_ms",
            "Duration avg (ms)",
            round(r.avg_duration * 1000, 2),
        )
        section.add(
            "duration_min_ms",
            "Duration min (ms)",
            round(r.min_duration * 1000, 2),
        )
        section.add(
            "duration_max_ms",
            "Duration max (ms)",
            round(r.max_duration * 1000, 2),
        )
        section.add(
            "duration_p50_ms",
            "Duration p50 (ms)",
            round(r.p50_duration * 1000, 2),
        )
        section.add(
            "duration_p99_ms",
            "Duration p99 (ms)",
            round(r.p99_duration * 1000, 2),
        )
        section.add(
            "duration_std_ms",
            "Duration std (ms)",
            round(r.std_duration * 1000, 2),
        )
        section.add(
            "throughput_avg_mbps",
            "Throughput avg (MB/s)",
            round(r.avg_throughput_mbps, 2),
        )
        section.add(
            "throughput_min_mbps",
            "Throughput min (MB/s)",
            round(r.min_throughput_mbps, 2),
        )
        section.add(
            "throughput_max_mbps",
            "Throughput max (MB/s)",
            round(r.max_throughput_mbps, 2),
        )
        section.add(
            "ops_per_sec_avg",
            "Avg ops/s",
            round(r.avg_ops_per_sec, 2),
        )
        section.add(
            "latency_per_key_ms",
            "Avg latency / key (ms)",
            round(r.avg_latency_per_key_ms, 3),
        )
        if r.expected_max_hit_rate > 0 or r.expected_hit_count > 0:
            section.add(
                "expected_max_hit_rate",
                "Expected max hit rate",
                round(r.expected_max_hit_rate, 4),
            )
            section.add(
                "expected_hit_count",
                "Expected hit keys",
                r.expected_hit_count,
            )
            section.add(
                "actual_hit_rate",
                "Actual hit rate",
                round(r.actual_hit_rate, 4),
            )

    metrics.emit()
