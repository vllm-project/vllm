# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional runtime refinement for configs produced from vLLM Recipes.

The recipe remains the baseline. Each policy may return an override only when
enough information is available. Keep policies independent so the tuning
algorithms can evolve without changing the converter or hardware detector.

The policies in this draft are intentionally simple heuristics, not benchmark-
proven universal optima.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from hardware_detection import HardwareInfo

_MIN_BATCHED_TOKENS = 2048
_MAX_BATCHED_TOKENS = 32768
_MAX_PARALLEL_PREFILLS = 8


@dataclass(frozen=True)
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


@dataclass
class TuningResult:
    overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


Policy = Callable[
    [dict[str, Any], HardwareInfo | None, WorkloadHints, TuningResult],
    None,
]


def _record_override(
    result: TuningResult,
    key: str,
    value: Any,
    reason: str,
) -> None:
    result.overrides[key] = value
    result.notes.append(f"{key}={value!r}: {reason}")


def _round_up_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _resolve_tensor_parallel_size(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del config, workload
    if hardware is None or hardware.numa_node_count < 1:
        return

    # Draft policy: use the largest power-of-two TP that does not exceed the
    # effective NUMA-node count. This deliberately avoids hard-coding unusual
    # non-power-of-two topologies into the converter.
    candidate = 1 << (hardware.numa_node_count.bit_length() - 1)
    _record_override(
        result,
        "tensor-parallel-size",
        candidate,
        "derived from the effective NUMA-node count",
    )


def _resolve_gpu_memory_utilization(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del workload
    if hardware is None or not hardware.numa_nodes:
        return

    reserve_fraction = 0.10
    safe_fractions = [
        max(
            0.0,
            node.available_memory_bytes / node.total_memory_bytes
            - reserve_fraction,
        )
        for node in hardware.numa_nodes
        if node.total_memory_bytes > 0
    ]
    if not safe_fractions:
        return

    safe_fraction = round(min(0.90, min(safe_fractions)), 2)
    if safe_fraction <= 0:
        return

    recipe_value = config.get("gpu-memory-utilization")
    if isinstance(recipe_value, (int, float)):
        candidate = min(float(recipe_value), safe_fraction)
    else:
        candidate = min(0.80, safe_fraction)

    _record_override(
        result,
        "gpu-memory-utilization",
        round(candidate, 2),
        "constrained by effective per-NUMA available memory",
    )


def _resolve_max_num_seqs(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del config, hardware
    if workload.concurrency is None:
        return

    _record_override(
        result,
        "max-num-seqs",
        workload.concurrency,
        "derived from the optional target concurrency",
    )


def _resolve_max_num_batched_tokens(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del config, hardware
    if workload.input_tokens is None or workload.concurrency is None:
        return

    # Draft workload heuristic. Keep this policy isolated because benchmarking
    # may later replace the constants/formula without changing the CLI.
    active_prefills = min(workload.concurrency, _MAX_PARALLEL_PREFILLS)
    requested_budget = workload.input_tokens * active_prefills
    candidate = _round_up_power_of_two(requested_budget)
    candidate = max(_MIN_BATCHED_TOKENS, min(candidate, _MAX_BATCHED_TOKENS))

    _record_override(
        result,
        "max-num-batched-tokens",
        candidate,
        (
            "derived from input-token length and target concurrency "
            f"(active_prefills={active_prefills})"
        ),
    )


def _resolve_data_parallel_size(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del config, hardware
    if workload.target_qps is None:
        return

    result.notes.append(
        "data-parallel-size: target QPS supplied; kept recipe value because "
        "per-replica capacity is not known yet"
    )


DEFAULT_POLICIES: tuple[Policy, ...] = (
    _resolve_tensor_parallel_size,
    _resolve_gpu_memory_utilization,
    _resolve_max_num_seqs,
    _resolve_max_num_batched_tokens,
    _resolve_data_parallel_size,
)

# Select tuning behavior from the hardware requested by the Recipes rendering,
# not from devices physically present in the host. A GPU server still exposes
# its host CPU topology, so host inspection alone cannot determine deployment
# intent.
HARDWARE_TUNING_POLICIES: dict[str, tuple[Policy, ...]] = {
    "xeon6": DEFAULT_POLICIES,
}


def get_runtime_tuning_policies(
    recipe_hardware: object,
) -> tuple[Policy, ...]:
    if not isinstance(recipe_hardware, str) or not recipe_hardware:
        raise ValueError(
            "Runtime tuning requires the resolved Recipes JSON to declare "
            "a non-empty `hardware` field."
        )

    policies = HARDWARE_TUNING_POLICIES.get(recipe_hardware)
    if policies is None:
        supported = ", ".join(sorted(HARDWARE_TUNING_POLICIES))
        raise ValueError(
            "Runtime tuning is not supported for recipe hardware "
            f"{recipe_hardware!r}. Currently supported: {supported}."
        )
    return policies


def finetune_runtime_config(
    config: dict[str, Any],
    *,
    hardware: HardwareInfo | None = None,
    workload: WorkloadHints | None = None,
    policies: tuple[Policy, ...] = DEFAULT_POLICIES,
) -> TuningResult:
    """Return overrides; never mutate the recipe-derived config in place."""
    hints = workload or WorkloadHints()
    hints.validate()

    result = TuningResult()
    for policy in policies:
        policy(config, hardware, hints, result)
    return result
