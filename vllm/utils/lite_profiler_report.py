# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a single vLLM lite-profiler log in tabular form.

The script consumes the pipe-separated records emitted by `vllm.lite_profiler`
It expects log lines in the format: "<scope_name>|<elapsed_microseconds>"
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TextIO


def _extract_event_ns(filename: str) -> dict[str, list[int]]:
    """Collect the nanosecond timings for every scope in ``filenames``."""

    all_event_ns: dict[str, list[int]] = defaultdict(list)
    try:
        with open(filename, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Parse the new format: "scope_name|elapsed_microseconds"
                if "|" in line:
                    try:
                        scope_name, elapsed_us_str = line.split("|", 1)
                        elapsed_us = int(elapsed_us_str)
                        # Convert microseconds to nanoseconds for compatibility
                        elapsed_ns = elapsed_us * 1000
                        all_event_ns[scope_name].append(elapsed_ns)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Lite-profiler log not found: {filename}") from None
    return all_event_ns


def _sum_events(event_ns: dict[str, list[int]]) -> dict[str, int]:
    return {event: sum(values) for event, values in event_ns.items()}


def _format_duration_ns(value_ns: int, total_ns: int) -> str:
    seconds = value_ns / 1e9 if value_ns else 0.0
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
    "Model:Draft",
]


def _compute_table_rows(
    event_ns_sum: dict[str, int],
    events: Sequence[str],
) -> list[str]:
    total_ns = sum(event_ns_sum.get(event, 0) for event in events)
    cells = []
    for event in events:
        cells.append(_format_duration_ns(event_ns_sum.get(event, 0), total_ns))
    total_seconds = total_ns / 1_000_000_000 if total_ns else 0.0
    cells.append(f"{total_seconds:.2f}s")
    return cells


def _print_breakdown_tables(event_ns_sum: dict[str, int], *,
                            stream: TextIO) -> None:
    for title, events in (
        ("Top-level pipeline events", TOP_EVENTS),
        ("Model events breakdown (only includes the main key events)",
         MODEL_EVENTS),
    ):
        headers = [*events, "TOTAL"]
        rows = [_compute_table_rows(event_ns_sum, events)]
        _render_table(title, headers, rows, stream=stream)


def summarize_log(log_path: str, *, stream: TextIO) -> None:
    event_ns = _extract_event_ns(log_path)
    event_ns_sum = _sum_events(event_ns)
    _print_breakdown_tables(event_ns_sum, stream=stream)
