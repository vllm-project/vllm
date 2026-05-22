"""Parse Nsight Systems GUI/CLI JSONL export (not Chrome trace JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plotting_tools.classify import CONTROL_PATTERNS, classify_event
from plotting_tools.classify_report import ClassificationReport

# CUPTI cudaMemcpyKind
_MEMCPY_KIND = {
    0: "memcpy unknown",
    1: "memcpy htod",
    2: "memcpy dtoh",
    3: "memcpy dtod",
    4: "memcpy default",
    5: "memcpy host",
    6: "memcpy device",
}

_RUNTIME_COMM_HINTS = (
    "nccl",
    "cudamemcpy",
    "cudamemcpyasync",
    "cudamemcpy2d",
)


def _resolve_name(row: dict[str, Any], strings: dict[int, str]) -> str:
    for key in ("demangledName", "shortName", "mangledName", "nameId"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, int):
            return strings.get(val, f"id:{val}")
        return str(val)
    return row.get("table", "unknown")


def _ns_to_us(ts_ns: int) -> int:
    return ts_ns // 1000


def _skipped_table_label(
    row: dict[str, Any],
    strings: dict[int, str],
    table: str | None,
) -> str:
    """Resolve a human-readable label for skipped non-CUPTI rows."""
    for key in ("text", "name", "message"):
        val = row.get(key)
        if val:
            return str(val)[:500]
    name_id = row.get("nameId")
    if name_id is not None:
        try:
            return strings.get(int(name_id), f"nameId:{name_id}")[:500]
        except (TypeError, ValueError):
            pass
    return str(table or "unknown")


def load_nsys_jsonl(
    path: Path,
    *,
    strict: bool = False,
    report: ClassificationReport | None = None,
) -> list[dict[str, Any]]:
    """
    Stream Nsight JSONL and return classified duration events.

    Timestamps are in microseconds (Chrome-trace compatible).
    """
    strings: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    uncl: list[str] = []
    lines = 0
    if report is None:
        report = ClassificationReport(trace_path=str(path))

    with path.open() as f:
        for line in f:
            lines += 1
            report.jsonl_lines = lines
            if lines % 1_000_000 == 0:
                print(f"  ... scanned {lines:,} lines, {len(events):,} events")

            line = line.strip()
            if not line:
                report.skipped_rows["empty_line"] += 1
                continue
            row = json.loads(line)
            table = row.get("table")

            if table == "StringIds":
                strings[int(row["id"])] = row["value"]
                continue

            start = row.get("start")
            end = row.get("end")
            if start is None or end is None:
                report.skipped_rows["no_start_end"] += 1
                continue
            dur_ns = int(end) - int(start)
            if dur_ns <= 0:
                report.skipped_rows["zero_duration"] += 1
                continue
            ts = _ns_to_us(int(start))
            dur = dur_ns // 1000

            extra: dict[str, Any] = {}
            if row.get("deviceId") is not None:
                extra["device_id"] = int(row["deviceId"])
            if row.get("streamId") is not None:
                extra["stream_id"] = int(row["streamId"])
            if table == "CUPTI_ACTIVITY_KIND_KERNEL":
                name = _resolve_name(row, strings)
                cat = "kernel"
            elif table == "CUPTI_ACTIVITY_KIND_MEMCPY":
                kind = int(row.get("copyKind", 0))
                name = _MEMCPY_KIND.get(kind, "memcpy")
                cat = "memcpy"
                extra["copy_kind"] = kind
                if row.get("bytes") is not None:
                    extra["bytes"] = row["bytes"]
            elif table == "CUPTI_ACTIVITY_KIND_RUNTIME":
                name = strings.get(int(row["nameId"]), "")
                lower = name.lower()
                if "pytorch profiler" in lower:
                    report.skipped_rows["profiler"] += 1
                    continue
                comm_hit = any(h in lower for h in _RUNTIME_COMM_HINTS)
                control_hit = any(h in lower for h in CONTROL_PATTERNS)
                if not comm_hit and not control_hit:
                    report.skipped_rows["runtime_unmatched"] += 1
                    report.skipped_runtime_names[name[:500] or "(empty)"] += 1
                    continue
                cat = "runtime"
            else:
                report.skipped_rows[f"table:{table}"] += 1
                skip_name = _skipped_table_label(row, strings, table)
                if table == "OSRT_API":
                    report.skipped_osrt_names[skip_name] += 1
                elif table == "NVTX_EVENTS":
                    report.skipped_nvtx_names[skip_name] += 1
                continue

            kind, sub = classify_event(name, cat, uncl if strict else None)
            events.append(
                {
                    "name": name,
                    "cat": cat,
                    "ts": ts,
                    "dur": dur,
                    "end": ts + dur,
                    "kind": kind,
                    "sub": sub,
                    "args": extra,
                }
            )
            report.record_parsed_event(name, cat, kind, sub, args=extra)

    if strict and uncl:
        from plotting_tools.classify import summarize_unclassified

        summarize_unclassified(uncl)
        raise SystemExit(1)

    print(f"  parsed {len(events):,} events from {lines:,} JSONL lines")
    return sorted(events, key=lambda x: x["ts"])
