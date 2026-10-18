# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A shared BF16-to-FP32 projection with one fixed CuTe DSL reduction."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from .ll_bf16 import LLBf16Gemm


class FixedBf16Gemm(LLBf16Gemm):
    def dispatch(
        self, *, M: int, K: int, N: int, output_dtype: torch.dtype = torch.float32
    ) -> LLBf16Gemm.CompileKey:
        return self.CompileKey(
            backend="splitk", split_k=1, num_stages=4, output_dtype=output_dtype
        )


_KERNEL = FixedBf16Gemm()


def _bf16_fixed_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _KERNEL._validate_inputs(a, weight, torch.float32)
    if a.shape[0] == 0:
        return torch.empty((0, weight.shape[0]), device=a.device, dtype=torch.float32)
    return _KERNEL(a.contiguous(), weight.contiguous(), torch.float32)


def _bf16_fixed_mm_fake(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (a.shape[0], weight.shape[0]), device=a.device, dtype=torch.float32
    )


direct_register_custom_op(
    op_name="bf16_fixed_mm",
    op_func=_bf16_fixed_mm,
    fake_impl=_bf16_fixed_mm_fake,
)


def bf16_fixed_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute live BF16 operands using a fixed K256, single-split reduction."""
    return torch.ops.vllm.bf16_fixed_mm(a, weight)
