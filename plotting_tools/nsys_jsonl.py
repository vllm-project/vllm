"""Parse Nsight Systems GUI/CLI JSONL export (not Chrome trace JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plotting_tools.classify import CONTROL_PATTERNS, classify_event
from plotting_tools.classify_report import ClassificationReport
from plotting_tools.comm_nvtx import parse_comm_nvtx_label, parse_iter_nvtx_label

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

# vLLM iteration_nvtx_range() outer step labels (VLLM_ITERATION_NVTX=1)
_ITERATION_NVTX_NAMES = frozenset({"prefill", "decode", "mixed", "idle"})


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


def _resolve_nvtx_text(row: dict[str, Any], strings: dict[int, str]) -> str:
    text = row.get("text")
    if text is not None and not isinstance(text, int):
        return str(text).strip()
    tid = row.get("textId")
    if tid is not None:
        try:
            return strings.get(int(tid), "").strip()
        except (TypeError, ValueError):
            pass
    return _skipped_table_label(row, strings, "NVTX_EVENTS")


def _is_iteration_nvtx_name(name: str) -> bool:
    if name.startswith("iter|"):
        return True
    return name.lower().strip() in _ITERATION_NVTX_NAMES


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Stream Nsight JSONL and return (events, iteration NVTX ranges, comm NVTX records).

    Timestamps are in microseconds (Chrome-trace compatible).

    - **iteration_nvtx_ranges**: outer prefill/decode/mixed/idle (timeline attribution)
    - **comm_nvtx_records**: inner send/recv/collective markers (message records)
    """
    strings: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    iteration_nvtx_ranges: list[dict[str, Any]] = []
    comm_nvtx_records: list[dict[str, Any]] = []
    uncl: list[str] = []
    lines = 0
    if report is None:
        report = ClassificationReport(trace_path=str(path))

    with path.open() as f:
        for line in f:
            lines += 1
            report.jsonl_lines = lines
            if lines % 1_000_000 == 0:
                print(
                    f"  ... scanned {lines:,} lines, {len(events):,} events, "
                    f"{len(iteration_nvtx_ranges):,} iteration + "
                    f"{len(comm_nvtx_records):,} comm NVTX records"
                )

            line = line.strip()
            if not line:
                report.skipped_rows["empty_line"] += 1
                continue
            row = json.loads(line)
            table = row.get("table")

            if table == "StringIds":
                strings[int(row["id"])] = row["value"]
                continue

            if table == "NVTX_EVENTS":
                name = _resolve_nvtx_text(row, strings)
                start = row.get("start")
                end = row.get("end")
                if start is None or end is None:
                    report.skipped_rows["nvtx_no_duration"] += 1
                    continue
                dur_ns = int(end) - int(start)
                if dur_ns <= 0:
                    report.skipped_rows["nvtx_zero_duration"] += 1
                    continue
                if _is_iteration_nvtx_name(name):
                    ts_us = _ns_to_us(int(start))
                    end_us = _ns_to_us(int(end))
                    if parsed_iter := parse_iter_nvtx_label(name):
                        label = parsed_iter["phase"]
                        report.iteration_nvtx_ranges[label] += 1
                        iteration_nvtx_ranges.append(
                            {
                                **parsed_iter,
                                "ts": ts_us,
                                "end": end_us,
                            }
                        )
                    else:
                        label = name.lower().strip()
                        report.iteration_nvtx_ranges[label] += 1
                        iteration_nvtx_ranges.append(
                            {
                                "name": label,
                                "scope": "iteration",
                                "ts": ts_us,
                                "end": end_us,
                            }
                        )
                elif parsed := parse_comm_nvtx_label(name):
                    ts_us = _ns_to_us(int(start))
                    end_us = _ns_to_us(int(end))
                    record = {
                        **parsed,
                        "ts": ts_us,
                        "end": end_us,
                        "dur_us": end_us - ts_us,
                        "scope": "comm",
                    }
                    comm_nvtx_records.append(record)
                    op_key = parsed["op"]
                    if parsed.get("tensor_key"):
                        op_key = f"{op_key}:{parsed['tensor_key']}"
                    report.comm_nvtx_records[op_key] += 1
                else:
                    report.skipped_nvtx_names[name[:500] or "(empty)"] += 1
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
                continue

            kind, sub = classify_event(
                name, cat, uncl if strict else None, args=extra
            )
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

    iteration_nvtx_ranges.sort(key=lambda r: r["ts"])
    comm_nvtx_records.sort(key=lambda r: r["ts"])
    print(
        f"  parsed {len(events):,} events, {len(iteration_nvtx_ranges):,} iteration "
        f"NVTX ranges, {len(comm_nvtx_records):,} comm message records "
        f"from {lines:,} JSONL lines"
    )
    return (
        sorted(events, key=lambda x: x["ts"]),
        iteration_nvtx_ranges,
        comm_nvtx_records,
    )
