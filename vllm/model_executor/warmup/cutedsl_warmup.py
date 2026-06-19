# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrate CuTeDSL warmup providers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
import time
from itertools import chain
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

CuTeDSLCompileFn = Callable[[], None]

# Attributes that commonly hold backend/kernel integration objects. The
# orchestrator does not know which are CuTeDSL-backed; objects opt in by
# defining get_cutedsl_warmup_plan(runner).
_CUTEDSL_PROVIDER_CHILD_ATTRS = (
    "impl",
    "prefill_backend",
    "_prefill_backend",
    "quant_method",
    "moe_kernel",
    "fused_experts",
)


@dataclass(frozen=True)
class CuTeDSLCompileUnit:
    name: str
    key: Hashable
    compile: CuTeDSLCompileFn


@dataclass(frozen=True)
class CuTeDSLWarmupPlan:
    """Warmup work requested by one CuTeDSL integration."""

    provider: str
    compile_units: tuple[CuTeDSLCompileUnit, ...] = ()


# Provider discovery.


# Walk likely integration objects looking for warmup providers.
def _iter_cutedsl_provider_candidates(
    runner: "GPUModelRunner",
) -> Iterable[object]:
    """Yield objects that may provide CuTeDSL warmup plans."""
    static_context = runner.vllm_config.compilation_config.static_forward_context
    roots = chain(static_context.values(), runner.get_model().modules())

    seen: set[int] = set()
    for root in roots:
        candidates: list[object | None] = [root]
        index = 0
        while index < len(candidates):
            candidate = candidates[index]
            index += 1
            if candidate is None:
                continue

            candidate_id = id(candidate)
            if candidate_id in seen:
                continue

            seen.add(candidate_id)
            yield candidate

            for attr in _CUTEDSL_PROVIDER_CHILD_ATTRS:
                candidates.append(getattr(candidate, attr, None))


# Plan collection and execution.


# Ask provider objects for their warmup plans.
def _collect_cutedsl_warmup_plans(
    runner: "GPUModelRunner",
) -> list[CuTeDSLWarmupPlan]:
    plans: list[CuTeDSLWarmupPlan] = []
    for target in _iter_cutedsl_provider_candidates(runner):
        get_plan = getattr(target, "get_cutedsl_warmup_plan", None)
        if get_plan is None:
            continue

        plan = get_plan(runner)
        if plan is None:
            continue
        if not isinstance(plan, CuTeDSLWarmupPlan):
            raise TypeError(
                "get_cutedsl_warmup_plan must return CuTeDSLWarmupPlan "
                "or None"
            )
        plans.append(plan)
    return plans


# Drop duplicate compile units across providers.
def _collect_unique_compile_units(
    plans: Iterable[CuTeDSLWarmupPlan],
) -> list[CuTeDSLCompileUnit]:
    seen: set[Hashable] = set()
    compile_units: list[CuTeDSLCompileUnit] = []

    for plan in plans:
        for unit in plan.compile_units:
            if unit.key in seen:
                logger.debug(
                    "Skipping duplicate CuTeDSL compile unit provider=%s "
                    "name=%s key=%r.",
                    plan.provider,
                    unit.name,
                    unit.key,
                )
                continue

            seen.add(unit.key)
            compile_units.append(unit)

    return compile_units


# Execute compile units under inference mode.
def _compile_cutedsl_warmup_units(
    compile_units: Iterable[CuTeDSLCompileUnit],
) -> int:
    compiled = 0
    with torch.inference_mode():
        for unit in compile_units:
            unit.compile()
            compiled += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return compiled


# Run all CuTeDSL warmup before serving.
@instrument(span_name="CuTeDSL warmup")
def cutedsl_warmup(runner: "GPUModelRunner") -> None:
    """Run CuTeDSL compile providers before serving."""
    if not current_platform.is_cuda():
        logger.info("Skipping CuTeDSL warmup on non-CUDA platform.")
        return

    if runner.is_pooling_model:
        logger.info("Skipping CuTeDSL warmup for pooling model.")
        return

    plans = _collect_cutedsl_warmup_plans(runner)
    if not plans:
        logger.info(
            "Skipping CuTeDSL warmup because no providers requested warmup."
        )
        return

    provider_names = list(dict.fromkeys(plan.provider for plan in plans))
    compile_units = _collect_unique_compile_units(plans)
    logger.info(
        "Warming up CuTeDSL providers=%s with compile_units=%d.",
        provider_names,
        len(compile_units),
    )

    start_time = time.perf_counter()
    compiled_count = _compile_cutedsl_warmup_units(compile_units)
    logger.info(
        "CuTeDSL warmup compiled %d units in %.2f s.",
        compiled_count,
        time.perf_counter() - start_time,
    )
