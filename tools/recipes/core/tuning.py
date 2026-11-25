# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Workload hints and shared runtime tuning result types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadHints:
    input_tokens: int | None = None
    output_tokens: int | None = None
    concurrency: int | None = None
    ttft_sla_ms: float | None = None
    tpot_sla_ms: float | None = None
    target_qps: float | None = None

    def validate(self) -> None:
        for name in ("input_tokens", "output_tokens", "concurrency"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        for name in ("ttft_sla_ms", "tpot_sla_ms", "target_qps"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(slots=True)
class TuningResult:
    """Runtime overrides and benchmark-only seeds produced by tuning policies."""

    overrides: dict[str, Any] = field(default_factory=dict)
    sweep_overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


Policy = Callable[
    [
        dict[str, Any],
        Any,
        WorkloadHints,
        TuningResult,
    ],
    None,
]


def record_override(
    result: TuningResult,
    key: str,
    value: Any,
    reason: str,
) -> None:
    result.overrides[key] = value
    result.notes.append(f"{key}={value!r}: {reason}")


def record_sweep_override(
    result: TuningResult,
    key: str,
    value: Any,
    reason: str,
) -> None:
    """Record an explicit benchmark seed without changing deployment config."""
    result.sweep_overrides[key] = value
    result.notes.append(f"sweep {key}={value!r}: {reason}")
