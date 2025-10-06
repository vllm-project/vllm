# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a single vLLM lite-profiler log in tabular form.

The script consumes the JSONL records emitted by the script `vllm.lite_profiler`
It expects log lines containing JSON payloads with a ``metrics`` dictionary
whose values are ``{"ns": int}``.

Usage examples::

    # Use the log file pointed to by VLLM_LITE_PROFILER_LOG_PATH
    python -m tools.lite_profiler_report

    # Provide an explicit path
    python -m tools.lite_profiler_report /tmp/vllm-lite-profiler.log
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TextIO


def _extract_event_ns(filenames: Sequence[str]) -> dict[str, list[int]]:
    """Collect the nanosecond timings for every scope in ``filenames``."""

    all_event_ns: dict[str, list[int]] = defaultdict(list)
    for filename in filenames:
        try:
            with open(filename, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metrics = payload.get("metrics")
                    if not isinstance(metrics, dict):
                        continue
                    for event, meta in metrics.items():
                        if isinstance(meta, dict) and "ns" in meta:
                            all_event_ns[event].append(int(meta["ns"]))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Lite-profiler log not found: {filename}") from None
    return all_event_ns


def _sum_events(event_ns: dict[str, list[int]]) -> dict[str, int]:
    return {event: sum(values) for event, values in event_ns.items()}


def _format_duration_ns(value_ns: int, total_ns: int) -> str:
    seconds = value_ns / 1_000_000_000 if value_ns else 0.0
    percent = (value_ns * 100.0 / total_ns) if total_ns else 0.0
    return f"{seconds:.2f}s ({percent:.2f}%)"


def _render_table(title: str, headers: Sequence[str],
                  rows: Iterable[Sequence[str]], *, stream: TextIO) -> None:
    table = [list(headers)] + [list(row) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    print(f"\n{title}", file=stream)
    print("-" * sum(widths) + "-" * (len(widths) - 1), file=stream)

    def _fmt(row: Sequence[str]) -> str:
        return " ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(table[0]), file=stream)
    print(" ".join("-" * w for w in widths), file=stream)
    for row in table[1:]:
        print(_fmt(row), file=stream)


TOP_EVENTS = [
    # Input processing
    "Input:Process",
    "Step:Schedule",
    # Model execution
    "Step:Model",
    # Output processing
    "Step:Output",
]

MODEL_EVENTS = [
    "Model:UpdateState",
    "Model:PrepareInput",
    "Model:Forward",
    "Model:Postprocess",
    "Model:Sample",
    "Model:Bookkeep",
    "Model:EPLB",
]


def _compute_table_rows(
    name: str,
    event_ns_sum: dict[str, int],
    events: Sequence[str],
) -> list[str]:
    total_ns = sum(event_ns_sum.get(event, 0) for event in events)
    cells = [name]
    for event in events:
        cells.append(_format_duration_ns(event_ns_sum.get(event, 0), total_ns))
    total_seconds = total_ns / 1_000_000_000 if total_ns else 0.0
    cells.append(f"{total_seconds:.2f}s")
    return cells


def _print_breakdown_tables(name: str, event_ns_sum: dict[str, int], *,
                            stream: TextIO) -> None:
    for title, events in (
        ("Top-level pipeline events", TOP_EVENTS),
        ("Model events breakdown (only includes the main key events)",
         MODEL_EVENTS),
    ):
        headers = ["Log", *events, "TOTAL"]
        rows = [_compute_table_rows(name, event_ns_sum, events)]
        _render_table(title, headers, rows, stream=stream)


def summarize_log(log_path: str, *, stream: TextIO) -> None:
    event_ns = _extract_event_ns([log_path])
    event_ns_sum = _sum_events(event_ns)
    _print_breakdown_tables(os.path.basename(log_path),
                            event_ns_sum,
                            stream=stream)
