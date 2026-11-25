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

from math import ceil
from typing import Any

from core.tuning import Policy, TuningResult, WorkloadHints, record_sweep_override
from hardware.registry import get_default_hardware_backend, get_hardware_backend


def _get_active_sequence_count(
    config: dict[str, Any],
    workload: WorkloadHints,
) -> int:
    """Return the per-replica sequence count for one scheduler iteration."""
    if workload.concurrency is not None:
        return ceil(workload.concurrency / _get_data_parallel_size(config))

    configured = config.get("max-num-seqs")
    if (
        isinstance(configured, int)
        and not isinstance(configured, bool)
        and configured > 0
    ):
        return configured

    from vllm.config.scheduler import SchedulerConfig

    return SchedulerConfig.DEFAULT_MAX_NUM_SEQS


def _get_data_parallel_size(config: dict[str, Any]) -> int:
    """Return a valid configured DP size, falling back to one replica."""
    configured = config.get("data-parallel-size", 1)
    if (
        isinstance(configured, int)
        and not isinstance(configured, bool)
        and configured > 0
    ):
        return configured
    return 1


def _estimate_prefills_per_step(
    workload: WorkloadHints,
    active_sequences: int,
    data_parallel_size: int,
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
        per_replica_qps = workload.target_qps / data_parallel_size
        prefills_per_step = max(
            prefills_per_step,
            per_replica_qps * workload.tpot_sla_ms / 1000.0,
        )

    # A step cannot replace more requests than are active.
    return min(float(active_sequences), prefills_per_step)


def _resolve_max_num_seqs(
    config: dict[str, Any],
    hardware: Any,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del hardware
    if workload.concurrency is None:
        return

    data_parallel_size = _get_data_parallel_size(config)
    per_replica_concurrency = ceil(workload.concurrency / data_parallel_size)

    record_sweep_override(
        result,
        "max-num-seqs",
        per_replica_concurrency,
        (
            "benchmark seed derived from the optional global target concurrency "
            f"and data_parallel_size={data_parallel_size}; config.yml keeps the "
            "recipe/vLLM scheduler default"
        ),
    )


def _resolve_max_num_batched_tokens(
    config: dict[str, Any],
    hardware: Any,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del hardware
    if workload.input_tokens is None:
        return

    data_parallel_size = _get_data_parallel_size(config)
    active_sequences = _get_active_sequence_count(config, workload)

    # Decode requests consume roughly one token per active sequence in one
    # scheduler iteration. Prefills share the remaining scheduler token budget.
    decode_budget = active_sequences
    prefills_per_step = _estimate_prefills_per_step(
        workload,
        active_sequences,
        data_parallel_size,
    )
    prefill_budget = ceil(workload.input_tokens * prefills_per_step)

    # Use vLLM's scheduler defaults/constraints as floors rather than local
    # recipe-specific magic constants.
    from vllm.config.scheduler import SchedulerConfig

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

    record_sweep_override(
        result,
        "max-num-batched-tokens",
        candidate,
        (
            "workload-derived benchmark seed "
            f"(per_replica_active_sequences={active_sequences}, "
            f"data_parallel_size={data_parallel_size}, "
            f"prefills_per_step={prefills_per_step:.2f}, "
            f"decode_budget={decode_budget}, "
            f"prefill_budget={prefill_budget}); config.yml keeps the "
            "recipe/vLLM scheduler default"
        ),
    )


SCHEDULER_TUNING_POLICIES: tuple[Policy, ...] = (
    _resolve_max_num_seqs,
    _resolve_max_num_batched_tokens,
)


def get_runtime_tuning_policies(
    recipe_hardware: object,
    *,
    tune_scheduler: bool = False,
) -> tuple[Policy, ...]:
    policies = get_hardware_backend(recipe_hardware).tuning_policies()
    if tune_scheduler:
        policies = (*policies, *SCHEDULER_TUNING_POLICIES)
    return policies


def finetune_runtime_config(
    config: dict[str, Any],
    *,
    hardware: Any = None,
    workload: WorkloadHints | None = None,
    policies: tuple[Policy, ...] | None = None,
) -> TuningResult:
    """Return deployment overrides and sweep seeds without mutating config."""
    hints = workload or WorkloadHints()
    hints.validate()

    if policies is None:
        # Preserve the historical CPU default for callers without a recipe ID.
        policies = get_default_hardware_backend().tuning_policies()

    result = TuningResult()
    for policy in policies:
        policy(config, hardware, hints, result)
    return result
