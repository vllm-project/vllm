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

from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from hardware_detection import HardwareInfo

from vllm.config.scheduler import SchedulerConfig


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


def _get_active_sequence_count(
    config: dict[str, Any],
    workload: WorkloadHints,
) -> int:
    """Return the sequence count used to size one scheduler iteration."""
    if workload.concurrency is not None:
        return workload.concurrency

    configured = config.get("max-num-seqs")
    if (
        isinstance(configured, int)
        and not isinstance(configured, bool)
        and configured > 0
    ):
        return configured

    return SchedulerConfig.DEFAULT_MAX_NUM_SEQS


def _estimate_prefills_per_step(
    workload: WorkloadHints,
    active_sequences: int,
) -> float:
    """Estimate steady-state prompt arrivals per decode scheduler step."""
    # Reserve enough prefill budget for at least one representative prompt.
    prefills_per_step = 1.0

    # In steady state, active_sequences / output_tokens approximates how many
    # requests finish and are replaced by new prefills per decode step.
    if workload.output_tokens is not None:
        prefills_per_step = max(
            prefills_per_step,
            active_sequences / workload.output_tokens,
        )

    # With target QPS + TPOT, approximate how many prompts arrive during one
    # decode scheduler-step interval.
    if workload.target_qps is not None and workload.tpot_sla_ms is not None:
        prefills_per_step = max(
            prefills_per_step,
            workload.target_qps * workload.tpot_sla_ms / 1000.0,
        )

    # A step cannot replace more requests than are active.
    return min(float(active_sequences), prefills_per_step)


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
            node.available_memory_bytes / node.total_memory_bytes - reserve_fraction,
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
    del hardware
    if workload.input_tokens is None:
        return

    active_sequences = _get_active_sequence_count(config, workload)

    # Decode requests consume roughly one token per active sequence in one
    # scheduler iteration. Prefills share the remaining scheduler token budget.
    decode_budget = active_sequences
    prefills_per_step = _estimate_prefills_per_step(workload, active_sequences)
    prefill_budget = ceil(workload.input_tokens * prefills_per_step)

    # Use vLLM's scheduler defaults/constraints as floors rather than local
    # recipe-specific magic constants.
    candidate = max(
        SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
        active_sequences,
        decode_budget + prefill_budget,
    )

    # Chunked prefill normally lets a prompt span scheduler iterations. If the
    # recipe explicitly disables it, vLLM requires the token budget to cover
    # max_model_len when max_model_len is known.
    if config.get("enable-chunked-prefill") is False:
        max_model_len = config.get("max-model-len")
        if (
            isinstance(max_model_len, int)
            and not isinstance(max_model_len, bool)
            and max_model_len > 0
        ):
            candidate = max(candidate, max_model_len)

    _record_override(
        result,
        "max-num-batched-tokens",
        candidate,
        (
            "workload-derived scheduler budget "
            f"(active_sequences={active_sequences}, "
            f"prefills_per_step={prefills_per_step:.2f}, "
            f"decode_budget={decode_budget}, "
            f"prefill_budget={prefill_budget})"
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
