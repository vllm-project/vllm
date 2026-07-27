# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime adapter for checked-in SiLU-and-mul block-quant kernels."""

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

OP_NAME = "silu_and_mul_per_block_quant"
NATIVE_OP_NAME = "silu_and_mul_per_block_quant"


def _eligible_module(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.shape[1] % 2 != 0
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or out.shape != (input.shape[0], input.shape[1] // 2)
        or out.dtype != current_platform.fp8_dtype()
        or out.device != input.device
        or not out.is_contiguous()
        or scale_ub is not None
        or not is_scale_transposed
        or scales.dtype != torch.float32
        or scales.device != input.device
        or group_size < 1
        or out.shape[1] % group_size != 0
        or scales.shape != (input.shape[0], out.shape[1] // group_size)
        or scales.stride() != (1, input.shape[0])
    ):
        return None
    return _select_bucketed_module(
        OP_NAME,
        _runtime_platform(),
        (out.shape[1], group_size),
        input.shape[0],
    )


def silu_and_mul_per_block_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    module_path = _eligible_module(
        out,
        input,
        scales,
        group_size,
        scale_ub,
        is_scale_transposed,
    )
    if module_path is None:
        torch.ops._C.silu_and_mul_per_block_quant(
            out,
            input,
            scales,
            group_size,
            scale_ub,
            is_scale_transposed,
        )
        return
    _load_launcher(module_path)(
        out,
        input,
        scales,
        group_size,
        scale_ub,
        is_scale_transposed,
    )


def warmup(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    fp8_dtype = current_platform.fp8_dtype()
    for intermediate_size, group_size, num_tokens in _selected_cases(
        OP_NAME, token_counts
    ):
        input = torch.empty(
            (num_tokens, 2 * intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        out = torch.empty(
            (num_tokens, intermediate_size), device=device, dtype=fp8_dtype
        )
        groups_per_row = intermediate_size // group_size
        scales = torch.empty(
            (groups_per_row, num_tokens), device=device, dtype=torch.float32
        ).t()
        silu_and_mul_per_block_quant(out, input, scales, group_size, None, True)


direct_register_custom_op(
    op_name="silu_and_mul_per_block_quant",
    op_func=silu_and_mul_per_block_quant,
    mutates_args=["out", "scales"],
    fake_impl=lambda *args, **kwargs: None,
    target_lib=vllm_helion_generated_lib,
)
