# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm output-gating kernel for HY V4 MLA."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_SIZE = 2048


@triton.jit
def _bf16_sigmoid_mul_kernel(
    attention_ptr,
    gate_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    attention = tl.load(attention_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)

    # torch.sigmoid preserves its BF16 input dtype. Reproduce that intermediate
    # rounding boundary before the multiply so this remains bitwise equivalent
    # to ``attention * torch.sigmoid(gate)``.
    sigmoid_bf16 = tl.sigmoid(gate).to(tl.bfloat16)
    output = attention.to(tl.float32) * sigmoid_bf16.to(tl.float32)
    tl.store(output_ptr + offsets, output, mask=mask)


def _hy4_rocm_bf16_sigmoid_mul_impl(
    attention: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    assert attention.dtype == gate.dtype == torch.bfloat16
    assert attention.shape == gate.shape
    assert attention.is_contiguous() and gate.is_contiguous()

    output = torch.empty_like(attention)
    num_elements = attention.numel()
    if num_elements == 0:
        return output
    _bf16_sigmoid_mul_kernel[(triton.cdiv(num_elements, _BLOCK_SIZE),)](
        attention,
        gate,
        output,
        num_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    return output


def _hy4_rocm_bf16_sigmoid_mul_fake(
    attention: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    del gate
    return torch.empty_like(attention)


direct_register_custom_op(
    op_name="hy4_rocm_bf16_sigmoid_mul",
    op_func=_hy4_rocm_bf16_sigmoid_mul_impl,
    fake_impl=_hy4_rocm_bf16_sigmoid_mul_fake,
)


def hy4_rocm_bf16_sigmoid_mul(
    attention: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.ops.vllm.hy4_rocm_bf16_sigmoid_mul(attention, gate)
