# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precompile the expert GEMMs of the Prims-TS MoE.

Untuned Prims-TS MoE calls pick their expert-GEMM configurations from the
number of tokens and compile them on first use. The startup dummy runs only
reach some token counts, so serving would compile the others, stalling every
rank for seconds.
"""

import time
from collections.abc import Iterable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.prims_ts_nvfp4_moe import (
    PrimsTsNvFp4ExpertsModular,
)
from vllm.utils.flashinfer import has_flashinfer_prims_ts_moe_precompile

logger = init_logger(__name__)


def _compile_signature(value: Any) -> Any:
    """`value` with each tensor reduced to its type and layout."""
    if isinstance(value, torch.Tensor):
        return (value.dtype, tuple(value.shape), value.stride(), value.device)
    if isinstance(value, tuple):
        return tuple(_compile_signature(item) for item in value)
    return value


def _distinct_calls(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """`calls` without those that differ from an earlier one in tensor data
    only, which compiles nothing new."""
    distinct: dict[tuple, dict[str, Any]] = {}
    for kwargs in calls:
        signature = tuple(
            (name, _compile_signature(value)) for name, value in kwargs.items()
        )
        distinct.setdefault(signature, kwargs)
    return list(distinct.values())


@torch.inference_mode()
def prims_ts_moe_warmup(model: torch.nn.Module) -> None:
    """Precompile the untuned expert GEMMs of every Prims-TS MoE layer."""
    layers = [
        (module, experts)
        for module in model.modules()
        if isinstance(
            experts := getattr(module, "prims_ts_moe_warmup_provider", None),
            PrimsTsNvFp4ExpertsModular,
        )
    ]
    if not layers:
        return
    if not has_flashinfer_prims_ts_moe_precompile():
        logger.warning(
            "This FlashInfer cannot precompile the Prims-TS MoE; its expert "
            "GEMMs will compile during inference."
        )
        return

    start = time.perf_counter()
    calls = _distinct_calls(
        experts.untuned_precompile_kwargs(module) for module, experts in layers
    )
    num_configs = sum(
        PrimsTsNvFp4ExpertsModular.precompile_untuned(kwargs) for kwargs in calls
    )
    logger.info(
        "Precompiled %d Prims-TS MoE expert configurations in %.2fs.",
        num_configs,
        time.perf_counter() - start,
    )
