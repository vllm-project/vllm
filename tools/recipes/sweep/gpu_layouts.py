# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded GPU-aware parallel layouts for recipe sweeps.

The planner verifies visibility and attention-head partitioning. It cannot
prove weight/KV-cache fit; startup failures remain isolated by the sweep runner.
"""

from __future__ import annotations

from typing import Any

from core.model_constraints import load_head_constraints
from core.tuning import WorkloadHints
from hardware.gpu import GPULayoutPolicy


def build_gpu_parallel_layout_plan(
    config: dict[str, Any],
    workload: WorkloadHints,
    gpu_count: int,
    *,
    tune_scheduler: bool = False,
    policy: GPULayoutPolicy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Import lazily to keep normal recipe conversion and CPU sweeps independent.
    from .sweep_generation import (
        _scheduler_baseline_for_dp,
        validate_sweep_workload,
    )

    policy = policy or GPULayoutPolicy()
    validate_sweep_workload(workload)
    if gpu_count < 1:
        raise ValueError(
            f"{policy.device_label} parallel sweep requires at least one visible GPU"
        )
    for key in (
        "pipeline-parallel-size",
        "prefill-context-parallel-size",
        "decode-context-parallel-size",
    ):
        if config.get(key, 1) != 1:
            raise ValueError(f"{policy.device_label} sweep currently requires {key}=1")
    if config.get("enable-expert-parallel") is True:
        raise ValueError(
            f"{policy.device_label} parallel sweep does not yet support "
            "expert parallel layouts"
        )

    recipe_tp = config.get("tensor-parallel-size", 1)
    recipe_dp = config.get("data-parallel-size", 1)
    for name, value in (
        ("tensor-parallel-size", recipe_tp),
        ("data-parallel-size", recipe_dp),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid {name}: {value!r}")
    if recipe_tp * recipe_dp > gpu_count:
        raise ValueError(f"Recipe TP*DP exceeds visible {policy.device_label} count")

    constraints = None
    failure = None
    try:
        constraints = load_head_constraints(config)
    except Exception as exc:
        failure = str(exc)

    # Offer DP>1 only when the backend permits it and model metadata
    # positively identifies a non-MoE model, avoiding unsupported MoE layouts.
    from .sweep_generation import inspect_model_parallel_metadata

    try:
        is_moe = inspect_model_parallel_metadata(config).is_moe
    except Exception:
        is_moe = None

    skipped: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    tps = sorted(set(policy.supported_tp) | {recipe_tp}, reverse=True)
    for tp in tps:
        if tp > gpu_count:
            continue
        if constraints is None and tp != recipe_tp:
            skipped.append(
                {
                    "_benchmark_name": f"tp{tp}_dp1",
                    "policy": "head_config_unavailable",
                    "reason": f"Cannot verify model head compatibility: {failure}",
                }
            )
            continue
        if constraints is not None and not all(c.supports(tp) for c in constraints):
            skipped.append(
                {
                    "_benchmark_name": f"tp{tp}_dp1",
                    "policy": "head_divisibility",
                    "reason": "Model attention/KV head don't support this TP",
                }
            )
            continue

        # Sweep TP-only layouts, plus full-card DP on known non-MoE models.
        # DP support and memory fit are verified by benchmark startups.
        dps = {1}
        if (
            policy.allow_dense_dp
            and is_moe is False
            and gpu_count % tp == 0
            and gpu_count // tp > 1
        ):
            dps.add(gpu_count // tp)
        for dp in sorted(dps):
            name = f"tp{tp}_dp{dp}"
            candidate = {
                "_benchmark_name": name,
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
            }
            if tune_scheduler:
                seqs, batch = _scheduler_baseline_for_dp(config, workload, dp)
                candidate["max_num_seqs"] = seqs
                candidate["max_num_batched_tokens"] = batch
            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            f"No {policy.device_label} TP/DP candidates passed head/device checks"
        )
    return candidates, skipped
