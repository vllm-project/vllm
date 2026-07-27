# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime adapter for checked-in RMSNorm block-quant kernels."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.kernels.helion_generated.dispatcher import (
    _load_launcher,
    _runtime_platform,
    _select_bucketed_module,
    _selected_cases,
    vllm_helion_generated_lib,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

OP_NAME = "rms_norm_per_block_quant"
NATIVE_OP_NAME = "rms_norm_per_block_quant"


def _eligible_module(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or result.shape != input.shape
        or result.dtype != current_platform.fp8_dtype()
        or result.device != input.device
        or not result.is_contiguous()
        or weight.shape != (input.shape[1],)
        or weight.dtype != input.dtype
        or weight.device != input.device
        or not weight.is_contiguous()
        or scale_ub is not None
        or residual is not None
        or not is_scale_transposed
        or scale.dtype != torch.float32
        or scale.device != input.device
        or group_size < 1
        or input.shape[1] % group_size != 0
        or scale.shape != (input.shape[0], input.shape[1] // group_size)
        or scale.stride() != (1, input.shape[0])
    ):
        return None
    return _select_bucketed_module(
        OP_NAME,
        _runtime_platform(),
        (input.shape[1], group_size),
        input.shape[0],
    )


def rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    module_path = _eligible_module(
        result,
        input,
        weight,
        scale,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )
    if module_path is None:
        torch.ops._C.rms_norm_per_block_quant(
            result,
            input,
            weight,
            scale,
            epsilon,
            scale_ub,
            residual,
            group_size,
            is_scale_transposed,
        )
        return
    _load_launcher(module_path)(
        result,
        input,
        weight,
        scale,
        epsilon,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )


def warmup(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    fp8_dtype = current_platform.fp8_dtype()
    for hidden_size, group_size, num_tokens in _selected_cases(OP_NAME, token_counts):
        input = torch.empty(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        result = torch.empty_like(input, dtype=fp8_dtype)
        weight = torch.empty(hidden_size, device=device, dtype=input.dtype)
        groups_per_row = hidden_size // group_size
        scale = torch.empty(
            (groups_per_row, num_tokens), device=device, dtype=torch.float32
        ).t()
        rms_norm_per_block_quant(
            result,
            input,
            weight,
            scale,
            1e-6,
            None,
            None,
            group_size,
            True,
        )


direct_register_custom_op(
    op_name="rms_norm_per_block_quant",
    op_func=rms_norm_per_block_quant,
    mutates_args=["result", "scale", "residual"],
    fake_impl=lambda *args, **kwargs: None,
    target_lib=vllm_helion_generated_lib,
)
