# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared conservative GPU fallback algorithm with backend-supplied policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.memory_estimation import estimate_kv_bytes, estimate_weight_bytes
from core.model_constraints import load_head_constraints
from core.validation import RecipeSkipError


@dataclass(frozen=True, slots=True)
class GPUFallbackPolicy:
    device_label: str
    supported_tp: tuple[int, ...]
    supports_device: Callable[[str], bool]
    unsupported_device_message: str
    reserve_fraction: float
    weight_multiplier: float
    runtime_reserve_bytes: int
    default_memory_utilization: float = 0.90


def _positive(config: dict[str, Any], key: str) -> int:
    value = config.get(key, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid recipe {key}={value!r}")
    return value


def select_gpu_fallback(
    config: dict[str, Any], hardware: Any, *, policy: GPUFallbackPolicy
) -> dict[str, Any]:
    """Choose a feasible smaller single-node layout, or SKIP with evidence.

    Feasibility is intentionally conservative and remains ESTIMATED. The vLLM
    server startup and test request remain the final correctness checks.
    """
    tp = _positive(config, "tensor-parallel-size")
    dp = _positive(config, "data-parallel-size")
    pp = _positive(config, "pipeline-parallel-size")
    if (
        pp != 1
        or any(
            config.get(k, 1) != 1
            for k in ("prefill-context-parallel-size", "decode-context-parallel-size")
        )
        or config.get("enable-expert-parallel")
    ):
        raise RecipeSkipError(
            f"{policy.device_label}_FALLBACK_UNSUPPORTED_LAYOUT",
            "Only TP/DP without PP, context or expert parallelism can be auto-scaled",
        )
    visible = hardware.gpu_count
    if visible < 1:
        raise RecipeSkipError(
            f"INSUFFICIENT_{policy.device_label}_DEVICES",
            f"No visible {policy.device_label} devices",
        )
    if tp * dp <= visible:
        return {
            "changed": False,
            "recipe": {"tp": tp, "dp": dp, "pp": pp},
            "effective": {"tp": tp, "dp": dp, "pp": pp},
            "visible_gpus": visible,
            "fit_status": "recipe_unmodified",
        }
    if any(not policy.supports_device(d.name) for d in hardware.devices):
        raise RecipeSkipError(
            f"{policy.device_label}_FALLBACK_UNSUPPORTED_HARDWARE",
            policy.unsupported_device_message,
        )
    if config.get("quantization") in ("bitsandbytes", "gguf") or config.get(
        "load-format"
    ) in ("gguf", "bitsandbytes"):
        raise RecipeSkipError(
            f"{policy.device_label}_FALLBACK_UNKNOWN_MEMORY",
            "Cannot reliably estimate the requested weight loading format",
        )
    try:
        constraints = load_head_constraints(config)
        weight_bytes, provenance = estimate_weight_bytes(config)
    except Exception as exc:
        raise RecipeSkipError(
            f"{policy.device_label}_FALLBACK_UNKNOWN_MEMORY",
            f"Cannot verify a smaller TP: {exc}",
        ) from exc
    usable = min(
        min(
            d.free_memory_bytes - int(d.total_memory_bytes * policy.reserve_fraction),
            int(
                d.total_memory_bytes
                * float(
                    config.get(
                        "gpu-memory-utilization", policy.default_memory_utilization
                    )
                )
            ),
        )
        for d in hardware.devices
    )
    if usable <= 0:
        raise RecipeSkipError(
            f"{policy.device_label}_FALLBACK_INSUFFICIENT_MEMORY",
            f"No available {policy.device_label} memory after reserve",
        )
    evaluated = []
    # DP first: preserve original TP when feasible; then shrink TP by powers of two.
    candidates = [
        n for n in sorted(policy.supported_tp, reverse=True) if n <= tp and n <= visible
    ]
    for candidate_tp in candidates:
        if not constraints or not all(c.supports(candidate_tp) for c in constraints):
            evaluated.append({"tp": candidate_tp, "reason": "head_constraints"})
            continue
        try:
            kv_bytes = estimate_kv_bytes(config, candidate_tp)
        except Exception as exc:
            raise RecipeSkipError(
                f"{policy.device_label}_FALLBACK_UNKNOWN_MEMORY",
                f"Cannot estimate KV cache: {exc}",
            ) from exc
        # Backend-provided replication/loading margin and runtime reserve.
        required = (
            math.ceil(weight_bytes * policy.weight_multiplier / candidate_tp)
            + kv_bytes
            + policy.runtime_reserve_bytes
        )
        evaluated.append(
            {
                "tp": candidate_tp,
                "estimated_per_gpu_bytes": required,
                "usable_per_gpu_bytes": usable,
                "fits": required <= usable,
            }
        )
        if required > usable:
            continue
        candidate_dp = min(dp, visible // candidate_tp)
        if candidate_dp < 1:
            continue
        return {
            "changed": (candidate_tp, candidate_dp) != (tp, dp),
            "recipe": {"tp": tp, "dp": dp, "pp": pp},
            "effective": {"tp": candidate_tp, "dp": candidate_dp, "pp": pp},
            "visible_gpus": visible,
            "fit_status": "estimated_requires_startup_validation",
            "weight_bytes": weight_bytes,
            "weight_source": provenance,
            "estimated_per_gpu_bytes": required,
            "usable_per_gpu_bytes": usable,
            "candidates": evaluated,
        }
    raise RecipeSkipError(
        f"{policy.device_label}_FALLBACK_INSUFFICIENT_MEMORY",
        "No smaller supported TP/DP layout passed conservative VRAM checks",
        recipe_parallelism={"tp": tp, "dp": dp, "pp": pp},
        visible_gpus=visible,
        candidates=evaluated,
    )
