# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm B12X JIT kernels used by a loaded model."""

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _collect_warmup_units(
    model: torch.nn.Module,
    token_counts: tuple[int, ...],
    output_dtype: torch.dtype,
) -> Iterable[B12xWarmupUnit]:
    units: dict[object, B12xWarmupUnit] = {}
    for layer in model.modules():
        provider = getattr(layer, "b12x_warmup_provider", None)
        get_unit = getattr(provider, "get_b12x_warmup_unit", None)
        if not callable(get_unit):
            continue
        unit = get_unit(layer, token_counts, output_dtype)
        assert isinstance(unit, B12xWarmupUnit)
        units.setdefault(unit.key, unit)
    return units.values()


def _compile_warmup_units(
    units: Iterable[B12xWarmupUnit],
) -> Counter[str]:
    warmed: Counter[str] = Counter()
    with torch.inference_mode():
        for unit in units:
            unit.compile()
            warmed[unit.name] += 1
        if warmed:
            torch.accelerator.synchronize()
    return warmed


def b12x_warmup(worker: "Worker", cudagraph_capture_sizes: list[int]) -> None:
    if not current_platform.is_cuda():
        return
    if not current_platform.is_device_capability_family(120):
        return

    output_dtype = getattr(
        getattr(worker, "model_config", None),
        "dtype",
        torch.bfloat16,
    )
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16
    token_counts = b12x_warmup_token_counts(
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    units = _collect_warmup_units(
        worker.get_model(),
        token_counts,
        output_dtype,
    )
    for name, count in _compile_warmup_units(units).items():
        logger.info_once(
            "Warmed up %d B12X %s linear GEMM signatures.",
            count,
            name,
        )
