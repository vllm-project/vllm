# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run registered CuTeDSL warmup compile units."""

# TODO(roberto): Remove this compatibility registry after registered CuTeDSL
# warmups are migrated to the shared JIT warmup infrastructure.
# https://github.com/vllm-project/vllm/pull/47451

from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

import torch
from tqdm import tqdm

from vllm.distributed import is_global_first_rank
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)

CuTeDSLCompileFn = Callable[[], None]


@dataclass(frozen=True)
class CuTeDSLCompileUnit:
    name: str
    key: Hashable
    compile: CuTeDSLCompileFn


_CUTEDSL_WARMUP_PROVIDERS: weakref.WeakSet[object] = weakref.WeakSet()


def register_cutedsl_warmup_provider(provider: object) -> None:
    """Register an object that can expose CuTeDSL warmup compile units."""
    _CUTEDSL_WARMUP_PROVIDERS.add(provider)


# Yield compile units from registered providers.
def _iter_cutedsl_warmup_compile_units() -> Iterable[CuTeDSLCompileUnit]:
    for provider in tuple(_CUTEDSL_WARMUP_PROVIDERS):
        get_units = getattr(provider, "get_cutedsl_warmup_compile_units", None)
        if not callable(get_units):
            continue

        compile_units = get_units()
        if compile_units is None:
            continue
        for unit in compile_units:
            if not isinstance(unit, CuTeDSLCompileUnit):
                raise TypeError(
                    "get_cutedsl_warmup_compile_units must return "
                    "CuTeDSLCompileUnit objects"
                )
            yield unit


# Drop duplicate compile units across providers.
def _collect_unique_compile_units(
    compile_units: Iterable[CuTeDSLCompileUnit],
) -> list[CuTeDSLCompileUnit]:
    seen: set[Hashable] = set()
    unique_compile_units: list[CuTeDSLCompileUnit] = []

    for unit in compile_units:
        if unit.key in seen:
            continue

        seen.add(unit.key)
        unique_compile_units.append(unit)

    return unique_compile_units


# Execute compile units under inference mode.
def _compile_cutedsl_warmup_units(
    compile_units: Iterable[CuTeDSLCompileUnit],
) -> int:
    compiled = 0
    if is_global_first_rank():
        compile_units = tqdm(compile_units, desc="Compiling CuTeDSL kernels")
    with torch.inference_mode():
        for unit in compile_units:
            unit.compile()
            compiled += 1
        torch.accelerator.synchronize()
    return compiled


# Run all CuTeDSL warmup before serving.
@instrument(span_name="CuTeDSL warmup")
def cutedsl_warmup() -> None:
    """Run CuTeDSL compile providers before serving."""
    if not current_platform.is_cuda():
        logger.info("Skipping CuTeDSL warmup on non-CUDA platform.")
        return

    compile_units = _collect_unique_compile_units(_iter_cutedsl_warmup_compile_units())
    if not compile_units:
        logger.info("Skipping CuTeDSL warmup because no compile units were requested.")
        return

    unit_names = list(dict.fromkeys(unit.name for unit in compile_units))
    logger.info(
        "Warming up CuTeDSL compile_units=%d names=%s.",
        len(compile_units),
        unit_names,
    )

    start_time = time.perf_counter()
    compiled_count = _compile_cutedsl_warmup_units(compile_units)
    logger.info(
        "CuTeDSL warmup compiled %d units in %.2f s.",
        compiled_count,
        time.perf_counter() - start_time,
    )
