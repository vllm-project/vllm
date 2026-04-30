# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IR op for dynamic per-token-group FP8 quantization."""

import torch
from torch import Tensor

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _execute_per_token_group_quant_fp8,
)

from ..op import register_op


@register_op
def dynamic_group_quant_fp8(
    x: Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    use_ue8m0: bool | None = None,
    out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-token-group FP8 quantization with optional pre-allocated activations.

    When ``out`` is ``None``, quantized activations are allocated. When ``out``
    is provided, results are written into ``out`` (in-place on that buffer).

    This is the public API; the shared implementation lives in
    ``fp8_utils._execute_per_token_group_quant_fp8``.
    """
    return _execute_per_token_group_quant_fp8(
        x,
        group_size,
        eps,
        dtype,
        column_major_scales,
        tma_aligned_scales,
        out,
        use_ue8m0,
    )


@dynamic_group_quant_fp8.register_input_generator
def _dynamic_group_quant_fp8_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    group_size: int = 128,
    column_major_scales: bool = False,
) -> tuple:
    assert hidden_size % group_size == 0, "hidden_size must be divisible by group_size"
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (
        x,
        group_size,
        1e-10,
        None,
        column_major_scales,
        False,
        None,
        None,
    )
