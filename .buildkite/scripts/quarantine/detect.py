# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flaky test detection and quarantine management.

Analyzes test result histories grouped by hardware backend and
identifies tests whose failure rate indicates flakiness rather than
a consistent bug. Pure logic — no I/O or network calls.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

MIN_RUNS = 5
FLAKY_THRESHOLD_LOW = 0.1
FLAKY_THRESHOLD_HIGH = 0.9
ANALYSIS_WINDOW_DAYS = 7
REINSTATEMENT_PASSES = 5

VALID_BACKENDS = frozenset({"cuda", "rocm", "intel", "cpu", "other"})

# Longest prefix match: order matters for overlapping prefixes.
# "cpu-small" must match "cpu" (not miss), "intel_gpu" must match "intel".
_BACKEND_PREFIXES: list[tuple[str, str]] = [
    ("h200", "cuda"),
    ("h100", "cuda"),
    ("b200", "cuda"),
    ("a100", "cuda"),
    ("l4", "cuda"),
    ("gh200", "cuda"),
    ("mi250", "rocm"),
    ("mi300", "rocm"),
    ("mi325", "rocm"),
    ("mi355", "rocm"),
    ("amd_cpu", "rocm"),
    ("amd-cpu", "rocm"),
    ("zen5", "rocm"),
    ("intel", "intel"),
    ("cpu", "cpu"),
    ("arm_cpu", "cpu"),
    ("arm-cpu", "cpu"),
    ("ascend", "other"),
    ("dgx", "other"),
]


@dataclasses.dataclass(frozen=True)
class TestResult:
    __test__ = False

    test_id: str
    backend: str
    passed: bool
    build_number: int
    timestamp: datetime


@dataclasses.dataclass(frozen=True)
class QuarantineEntry:
    test_id: str
    backend: str
    quarantined_at: str
    fail_rate: float
    total_runs: int
    reason: str


def normalize_backend(device: str) -> str:
    if not device:
        raise ValueError("device must be a non-empty string")
    device_lower = device.lower()
    for prefix, backend in _BACKEND_PREFIXES:
        if device_lower.startswith(prefix):
            return backend
    return "other"


def parse_test_result(raw: dict) -> TestResult:
    required = {"test_id", "backend", "passed", "build_number", "timestamp"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")
    backend = raw["backend"]
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend!r}, must be one of {sorted(VALID_BACKENDS)}"
        )
    return TestResult(
        test_id=str(raw["test_id"]),
        backend=backend,
        passed=bool(raw["passed"]),
        build_number=int(raw["build_number"]),
        timestamp=datetime.fromisoformat(str(raw["timestamp"])),
    )


def detect_flaky_tests(
    results: list[TestResult],
    now: datetime | None = None,
) -> list[QuarantineEntry]:
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ANALYSIS_WINDOW_DAYS)

    groups: dict[tuple[str, str], list[TestResult]] = {}
    for r in results:
        if r.timestamp < cutoff:
            continue
        groups.setdefault((r.test_id, r.backend), []).append(r)

    entries: list[QuarantineEntry] = []
    for (test_id, backend), group in sorted(groups.items()):
        total = len(group)
        if total < MIN_RUNS:
            continue
        failures = sum(1 for r in group if not r.passed)
        fail_rate = failures / total
        if FLAKY_THRESHOLD_LOW < fail_rate < FLAKY_THRESHOLD_HIGH:
            entries.append(
                QuarantineEntry(
                    test_id=test_id,
                    backend=backend,
                    quarantined_at=now.isoformat(),
                    fail_rate=round(fail_rate, 4),
                    total_runs=total,
                    reason=(
                        f"Fail rate {fail_rate:.0%} over {total} runs on {backend}"
                    ),
                )
            )
    return entries


def check_reinstatement(
    entry: QuarantineEntry,
    results: list[TestResult],
    now: datetime | None = None,
) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ANALYSIS_WINDOW_DAYS)

    relevant = sorted(
        (
            r
            for r in results
            if r.test_id == entry.test_id
            and r.backend == entry.backend
            and r.timestamp >= cutoff
        ),
        key=lambda r: r.build_number,
    )
    if len(relevant) < REINSTATEMENT_PASSES:
        return False
    return all(r.passed for r in relevant[-REINSTATEMENT_PASSES:])


def update_quarantine_list(
    current_entries: list[QuarantineEntry],
    results: list[TestResult],
    now: datetime | None = None,
) -> list[QuarantineEntry]:
    if now is None:
        now = datetime.now(timezone.utc)

    new_flaky = detect_flaky_tests(results, now)
    existing = {(e.test_id, e.backend): e for e in current_entries}
    new = {(e.test_id, e.backend): e for e in new_flaky}

    updated: list[QuarantineEntry] = []
    for key, entry in existing.items():
        if check_reinstatement(entry, results, now):
            continue
        updated.append(entry)

    for key, entry in new.items():
        if key not in existing:
            updated.append(entry)

    return sorted(updated, key=lambda e: (e.test_id, e.backend))
