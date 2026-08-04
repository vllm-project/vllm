#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable
import json
import re
from pathlib import Path
from typing import Any

_BPFTRACE_EVENT_RE = re.compile(
    r"\bevent=(?P<event>\S+)\s+"
    r"ts=(?P<ts>\d+)\s+"
    r"request_id=(?P<request_id>\S*)\s+"
    r"extra=(?P<extra>.*)$"
)

_PROFILE_CORE_ORDER = {
    "offline": (
        "request_arrival",
        "request_id_assigned",
        "request_engine_admitted",
        "request_finish",
    ),
    "online": (
        "request_arrival",
        "request_id_mapping",
        "request_id_assigned",
        "request_engine_admitted",
        "request_first_output",
        "request_finish",
    ),
}
_PROFILE_CHOICES = ("auto", "offline", "online")
_BPFTRACE_EVIDENCE_CLASS = "ENGINEERING_USDT_BPFTRACE_POC"
_SIDECAR_EVIDENCE_CLASS = "AUXILIARY_SIDECAR_TRANSPORT_ORACLE"


def _load_bpftrace_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = _BPFTRACE_EVENT_RE.search(line)
        if match is None:
            continue

        extra_text = match.group("extra").strip()
        try:
            extra = json.loads(extra_text) if extra_text else {}
        except json.JSONDecodeError:
            extra = {"raw": extra_text}

        events.append(
            {
                "source": "usdt",
                "line_no": line_no,
                "event": match.group("event"),
                "request_id": match.group("request_id") or None,
                "ts_ns": int(match.group("ts")),
                "extra": extra,
            }
        )
    return events


def _load_sidecar_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(
            {
                "source": "sidecar",
                "line_no": line_no,
                "event": payload.get("event"),
                "request_id": payload.get("request_id"),
                "ts_ns": payload.get("monotonic_ns") or payload.get("ts_ns"),
                "wall_ts_ns": payload.get("ts_ns"),
                "extra": payload.get("extra", {}),
            }
        )
    return events


def _build_alias_map(events: Iterable[dict[str, Any]]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for event in events:
        if event.get("event") != "request_id_mapping":
            continue
        extra = event.get("extra") or {}
        external = extra.get("external_request_id")
        internal = extra.get("internal_request_id")
        if external is None or internal is None:
            continue
        external_id = str(external)
        internal_id = str(internal)
        alias_map[external_id] = external_id
        alias_map[internal_id] = external_id
    return alias_map


def _canonical_request_id(
    event: dict[str, Any], alias_map: dict[str, str]
) -> str:
    raw_id = event.get("request_id")
    if raw_id is not None:
        return alias_map.get(str(raw_id), str(raw_id))

    extra = event.get("extra") or {}
    for key in ("external_request_id", "internal_request_id"):
        value = extra.get(key)
        if value is not None:
            return alias_map.get(str(value), str(value))
    return "<none>"


def _group_by_canonical_request(
    events: Iterable[dict[str, Any]],
    alias_map: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_canonical_request_id(event, alias_map)].append(event)
    for request_events in grouped.values():
        request_events.sort(key=lambda event: event.get("ts_ns") or 0)
    return dict(grouped)


def _duration_ms(first_ts: int | None, last_ts: int | None) -> float | None:
    if first_ts is None or last_ts is None:
        return None
    return (last_ts - first_ts) / 1_000_000


def _counter_to_list(counter: Counter[str]) -> list[str]:
    items = []
    for event_name, count in sorted(counter.items()):
        items.extend([event_name] * count)
    return items


def _event_sequence(events: list[dict[str, Any]]) -> list[str]:
    return [str(event.get("event")) for event in events if event.get("event")]


def _raw_request_ids(events: Iterable[dict[str, Any]]) -> list[str]:
    raw_ids = []
    seen = set()
    for event in events:
        candidates = [event.get("request_id")]
        extra = event.get("extra") or {}
        candidates.append(extra.get("external_request_id"))
        candidates.append(extra.get("internal_request_id"))
        for candidate in candidates:
            if candidate is None:
                continue
            raw_id = str(candidate)
            if raw_id not in seen:
                raw_ids.append(raw_id)
                seen.add(raw_id)
    return raw_ids


def _infer_profile(events: Iterable[dict[str, Any]], requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile

    for event in events:
        if event.get("event") in {"request_id_mapping", "request_first_output"}:
            return "online"
        extra = event.get("extra") or {}
        if extra.get("path") == "online":
            return "online"
    return "offline"


def _core_event_order_ok(event_sequence: list[str], profile: str) -> bool:
    required_order = _PROFILE_CORE_ORDER[profile]
    cursor = 0
    for event_name in event_sequence:
        if cursor < len(required_order) and event_name == required_order[cursor]:
            cursor += 1
    return cursor == len(required_order)


def _timestamp_delta_summary(
    usdt_events: list[dict[str, Any]], sidecar_events: list[dict[str, Any]]
) -> dict[str, float | int | None]:
    by_event_usdt: dict[str, list[int]] = defaultdict(list)
    by_event_sidecar: dict[str, list[int]] = defaultdict(list)
    for event in usdt_events:
        event_name = event.get("event")
        if event_name is not None and event.get("ts_ns") is not None:
            by_event_usdt[str(event_name)].append(int(event["ts_ns"]))
    for event in sidecar_events:
        event_name = event.get("event")
        if event_name is not None and event.get("ts_ns") is not None:
            by_event_sidecar[str(event_name)].append(int(event["ts_ns"]))

    deltas_ms = []
    for event_name, usdt_timestamps in by_event_usdt.items():
        sidecar_timestamps = by_event_sidecar.get(event_name, [])
        for usdt_ts, sidecar_ts in zip(usdt_timestamps, sidecar_timestamps):
            deltas_ms.append((usdt_ts - sidecar_ts) / 1_000_000)

    if not deltas_ms:
        return {"count": 0, "min": None, "max": None, "avg": None}

    return {
        "count": len(deltas_ms),
        "min": min(deltas_ms),
        "max": max(deltas_ms),
        "avg": sum(deltas_ms) / len(deltas_ms),
    }


def _summarize_request(
    canonical_request_id: str,
    usdt_events: list[dict[str, Any]],
    sidecar_events: list[dict[str, Any]] | None,
    requested_profile: str,
) -> dict[str, Any]:
    timestamps = [event.get("ts_ns") for event in usdt_events]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None
    usdt_event_sequence = _event_sequence(usdt_events)
    sidecar_event_sequence = (
        _event_sequence(sidecar_events) if sidecar_events is not None else []
    )
    raw_request_ids = _raw_request_ids(
        usdt_events + (sidecar_events if sidecar_events is not None else [])
    )
    profile = _infer_profile(
        usdt_events + (sidecar_events if sidecar_events is not None else []),
        requested_profile,
    )

    summary = {
        "request_id": canonical_request_id,
        "canonical_request_id": canonical_request_id,
        "raw_request_ids": raw_request_ids,
        "profile": profile,
        "bpftrace_evidence_class": _BPFTRACE_EVIDENCE_CLASS,
        "has_request_id_assigned": "request_id_assigned" in usdt_event_sequence,
        "has_engine_admitted": "request_engine_admitted" in usdt_event_sequence,
        "has_first_output": "request_first_output" in usdt_event_sequence,
        "has_finish": "request_finish" in usdt_event_sequence,
        "has_abort": "request_abort" in usdt_event_sequence,
        "event_count": len(usdt_events),
        "first_event_ts": first_ts,
        "last_event_ts": last_ts,
        "lifecycle_duration_ms": _duration_ms(first_ts, last_ts),
        "event_sequence": usdt_event_sequence,
        "usdt_event_sequence": usdt_event_sequence,
        "sidecar_event_sequence": sidecar_event_sequence,
        "core_event_order_ok": _core_event_order_ok(usdt_event_sequence, profile),
    }

    if sidecar_events is not None:
        usdt_counter = Counter(
            str(event.get("event")) for event in usdt_events if event.get("event")
        )
        sidecar_counter = Counter(
            str(event.get("event"))
            for event in sidecar_events
            if event.get("event")
        )
        summary.update(
            {
                "sidecar_event_count": len(sidecar_events),
                "usdt_event_count": len(usdt_events),
                "sidecar_evidence_class": _SIDECAR_EVIDENCE_CLASS,
                "missing_in_usdt": _counter_to_list(sidecar_counter - usdt_counter),
                "extra_in_usdt": _counter_to_list(usdt_counter - sidecar_counter),
                "timestamp_delta_pairing": (
                    "same_canonical_request_and_event_order"
                ),
                "timestamp_delta_ms_summary": _timestamp_delta_summary(
                    usdt_events, sidecar_events
                ),
            }
        )

    return summary


def summarize(
    usdt_events: list[dict[str, Any]],
    sidecar_events: list[dict[str, Any]] | None,
    profile: str = "auto",
) -> list[dict[str, Any]]:
    if profile not in _PROFILE_CHOICES:
        raise ValueError(f"unsupported profile: {profile}")
    alias_map = _build_alias_map(usdt_events + (sidecar_events or []))
    usdt_by_request = _group_by_canonical_request(usdt_events, alias_map)
    sidecar_by_request = _group_by_canonical_request(sidecar_events or [], alias_map)
    request_ids = set(usdt_by_request)
    if sidecar_events is not None:
        request_ids.update(sidecar_by_request)

    return [
        _summarize_request(
            request_id,
            usdt_by_request.get(request_id, []),
            (
                sidecar_by_request.get(request_id, [])
                if sidecar_events is not None
                else None
            ),
            profile,
        )
        for request_id in sorted(request_ids)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze vLLM L0 USDT bpftrace output by request_id."
    )
    parser.add_argument("bpftrace_output", type=Path)
    parser.add_argument("--sidecar-jsonl", type=Path)
    parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default="auto",
        help=(
            "Lifecycle profile to validate. 'auto' uses online-only events "
            "or event metadata to select a profile."
        ),
    )
    args = parser.parse_args()

    usdt_events = _load_bpftrace_events(args.bpftrace_output)
    sidecar_events = (
        _load_sidecar_events(args.sidecar_jsonl) if args.sidecar_jsonl else None
    )
    print(
        json.dumps(
            summarize(usdt_events, sidecar_events, profile=args.profile),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
