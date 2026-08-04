# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace replay`` — replay a trace file against a StorageManager."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import json
import os
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.metrics import Metrics, StreamHandler, get_formatter
from lmcache.logging import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.trace._driver import ReplayResult
    from lmcache.cli.commands.trace._stats import ReplayStatsCollector


class ReplayCommand(BaseCommand):
    """Replay a trace file against a fresh StorageManager."""

    def name(self) -> str:
        return "replay"

    def help(self) -> str:
        return "Replay a trace file against a fresh StorageManager."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_replay_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        run_trace_replay(args)


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache trace replay`` arguments to *parser*.

    Args:
        parser: The ``ArgumentParser`` for the replay subcommand.
    """
    parser.add_argument(
        "trace_path",
        metavar="FILE",
        help="Path to a .lct trace file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print one line per replayed record.",
    )
    parser.add_argument(
        "--jsonl-out",
        default=None,
        metavar="PATH",
        help=(
            "Write one JSON object per replayed record to PATH "
            "(qualname, latency_ms, failed).  Useful for post-hoc "
            "analysis."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help=(
            "Directory for aggregated CSV/JSON summary output "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip the aggregated CSV summary export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also export an aggregated JSON summary.",
    )

    try:
        # First Party
        from lmcache.v1.distributed.config import add_storage_manager_args
        from lmcache.v1.mp_observability.config import add_observability_args

        add_storage_manager_args(parser)
        add_observability_args(parser)
    except ImportError as e:
        logger.warning("lmcache trace replay import error, error is %s", e)


def run_trace_replay(args: argparse.Namespace) -> None:
    """Construct a StorageManager from *args* and drive replay.

    Produces three kinds of output:

    * Per-record stream: every dispatch is logged at INFO with its
      progress (``[N/total]``), qualname, and latency.
      ``--verbose`` additionally mirrors each record to stdout,
      and ``--jsonl-out PATH`` writes one JSON object per record
      to ``PATH`` for post-hoc analysis.
    * Aggregated per-qualname summary: CSV (unless ``--no-csv``)
      and JSON (with ``--json``) written under ``--output-dir``.
    * Terminal metrics table (unless ``--quiet``) using the shared
      :class:`~lmcache.cli.metrics.Metrics` renderer.

    Args:
        args: Parsed CLI arguments.
    """
    # First Party
    from lmcache.cli.commands.trace._driver import StorageReplayDriver
    from lmcache.v1.distributed.config import StorageManagerConfig, parse_args_to_config
    from lmcache.v1.mp_observability.config import parse_args_to_observability_config
    from lmcache.v1.mp_observability.trace.reader import TraceReader

    sm_config: StorageManagerConfig = parse_args_to_config(args)

    # ``--trace-level`` / ``--trace-output`` belong to the recording
    # surface.  They are still registered on the parser so the flag set
    # stays in lock-step with ``lmcache server``, but they have no
    # meaning here — any value a caller passes is silently clobbered to
    # ``None`` so the replay-side ``ObservabilityConfig`` never tries to
    # start a recorder.
    args.trace_level = None
    args.trace_output = None
    obs_config = parse_args_to_observability_config(args)

    # Create output directories *before* replay starts.  A replay
    # can run for minutes; surfacing a bad ``--output-dir`` or
    # unwritable ``--jsonl-out`` parent now avoids silently losing
    # the summary/stream after the work has already happened.
    os.makedirs(args.output_dir, exist_ok=True)
    if args.jsonl_out:
        jsonl_parent = os.path.dirname(os.path.abspath(args.jsonl_out))
        if jsonl_parent:
            os.makedirs(jsonl_parent, exist_ok=True)

    # ANSI: bold + yellow for the banner text, reset at the end.
    bold = "\033[1;33m"
    reset = "\033[0m"
    bar = "=" * 78
    logger.warning(
        "\n%s%s\n"
        "  !! REPLAY ENVIRONMENT MISMATCH MAY CAUSE RETRIEVE MISSES !!\n"
        "%s%s\n"
        "  * Replay uses the *replay-side* StorageManager config, which\n"
        "    may differ from the config recorded in the trace.\n"
        "  * Replay runs on a host whose performance may differ from\n"
        "    the recording host.\n"
        "  * StorageManager reads/writes are async — an L2 load that\n"
        "    had finished at record time may not have finished yet at\n"
        "    replay time, so the matching retrieve can miss.\n"
        "\n"
        "  Treat retrieve-miss counts as a signal about the replay\n"
        "  environment, not as a defect in the trace.\n"
        "%s%s",
        bold,
        bar,
        bar,
        reset,
        bar,
        reset,
    )

    # Pre-scan to count total records so progress logs can show
    # [N/total].  The reader streams frames, so counting is cheap
    # relative to replay (which actually dispatches StorageManager
    # calls).
    with TraceReader(args.trace_path) as r:
        total_records = sum(1 for _ in r.records())
    logger.info(
        "trace replay: file=%s records=%d",
        args.trace_path,
        total_records,
    )

    jsonl_fh = open(args.jsonl_out, "w") if args.jsonl_out else None
    verbose = args.verbose
    counter = {"n": 0}

    def _on_record(qualname: str, latency_s: float, failed: bool) -> None:
        counter["n"] += 1
        status = "FAIL" if failed else "OK"
        logger.info(
            "[%d/%d] %s %s (%.3fms)",
            counter["n"],
            total_records,
            status,
            qualname,
            latency_s * 1000.0,
        )
        if verbose:
            print(
                f"  [{counter['n']}/{total_records}]  "
                f"{status:<4}  {latency_s * 1000:8.3f}ms  {qualname}"
            )
        if jsonl_fh is not None:
            jsonl_fh.write(
                json.dumps(
                    {
                        "qualname": qualname,
                        "latency_ms": latency_s * 1000.0,
                        "failed": failed,
                    }
                )
                + "\n"
            )

    try:
        with StorageReplayDriver(
            sm_config, args.trace_path, obs_config=obs_config
        ) as driver:
            result = driver.run(on_record=_on_record)
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()

    if not args.no_csv:
        csv_path = os.path.join(args.output_dir, "trace_replay_ops.csv")
        result.stats.export_csv(csv_path)
        logger.info("CSV written to %s", csv_path)
    if args.json:
        json_path = os.path.join(args.output_dir, "trace_replay_summary.json")
        result.stats.export_json(json_path)
        logger.info("JSON written to %s", json_path)

    if not args.quiet:
        _emit_replay_metrics(result.stats, result)

    if result.records_failed > 0:
        sys.exit(1)


def _emit_replay_metrics(
    stats: "ReplayStatsCollector",
    result: "ReplayResult",
) -> None:
    """Print the replay summary using the shared :class:`Metrics` renderer.

    Args:
        stats: The stats collector populated during replay.
        result: The full :class:`ReplayResult` — used for the
            replayed/skipped/failed totals and digest comparison.
    """
    metrics = Metrics(title="Trace Replay Result")
    metrics.add_handler(StreamHandler(get_formatter("terminal", width=64)))

    overall = metrics.add_section("overall", "Overall")
    overall.add("level", "Trace level", result.header_level)
    overall.add("replayed", "Records replayed", result.records_replayed)
    overall.add("skipped", "Records skipped", result.records_skipped)
    overall.add("failed", "Records failed", result.records_failed)
    overall.add(
        "duration",
        "Replay duration (s)",
        round(stats.total_duration_s(), 3),
    )
    header_digest = result.header_digest
    replay_digest = result.replay_config_digest
    if header_digest and replay_digest and header_digest != replay_digest:
        overall.add(
            "digest",
            "Config digest",
            f"MISMATCH (rec={header_digest[:8]}, run={replay_digest[:8]})",
        )
    elif header_digest:
        overall.add("digest", "Config digest", f"match ({header_digest[:8]})")

    summary = stats.summary()
    if summary:
        ops_section = metrics.add_section("ops", "Per-Op Latency (ms)")
        for qn in sorted(summary):
            s = summary[qn]
            short = _short_op_name(qn)
            ops_section.add(f"{short}_count", f"{short} count", s.count)
            ops_section.add(
                f"{short}_mean",
                f"{short} mean",
                round(s.mean_ms, 3),
            )
            ops_section.add(
                f"{short}_p50",
                f"{short} p50",
                round(s.p50_ms, 3),
            )
            ops_section.add(
                f"{short}_p99",
                f"{short} p99",
                round(s.p99_ms, 3),
            )

    metrics.emit()


def _short_op_name(qualname: str) -> str:
    """Return a compact, human-readable label for a traced qualname.

    Plain methods collapse to the method name: the table has limited
    column width and the fully-qualified path is verbose.

    Context-manager handlers (``__enter__`` / ``__exit__``) instead
    collapse to ``<owning_method>.enter`` / ``<owning_method>.exit``,
    so the reader can tell *which* context manager the pair belongs
    to — the bare ``__enter__`` / ``__exit__`` label is useless when
    multiple context-manager-returning methods are traced.

    Args:
        qualname: Dotted qualname recorded by the tracer, e.g.
            ``lmcache.v1.distributed.storage_manager.StorageManager.read_prefetched_results.__enter__``.

    Returns:
        A short label suitable as a metrics row prefix.
    """
    parts = qualname.split(".")
    last = parts[-1]
    if last in ("__enter__", "__exit__") and len(parts) >= 2:
        return f"{parts[-2]}.{last.strip('_')}"
    return last
