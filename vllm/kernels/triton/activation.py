# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_MAX_INTERMEDIATE_SIZE = 32768
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _gelu_and_mul_sparse_kernel(
    x_ptr,
    output_ptr,
    d: tl.constexpr,
    std_multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    row_start = row * 2 * d

    gate = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row_start + d + offsets, mask=mask, other=0.0).to(tl.float32)

    input_dtype = x_ptr.dtype.element_ty
    mean = tl.sum(gate, axis=0) / d
    centered = tl.where(mask, gate - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / d
    mean = mean.to(input_dtype).to(tl.float32)
    std = tl.sqrt(variance).to(input_dtype).to(tl.float32)
    scaled_std = (std * std_multiplier).to(input_dtype).to(tl.float32)
    cutoff = (mean + scaled_std).to(input_dtype).to(tl.float32)
    sparse_gate = (gate - cutoff).to(input_dtype).to(tl.float32)
    sparse_gate = tl.where(sparse_gate < 0.0, 0.0, sparse_gate)

    inner = 0.7978845608028654 * (
        sparse_gate + 0.044715 * sparse_gate * sparse_gate * sparse_gate
    )
    tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    activated = 0.5 * sparse_gate * (1.0 + tanh)
    activated = activated.to(input_dtype).to(tl.float32)
    tl.store(output_ptr + row * d + offsets, activated * up, mask=mask)


def _supports_gelu_and_mul_sparse(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> bool:
    del std_multiplier
    if (
        x.device.type != "cuda"
        or x.dtype not in _SUPPORTED_DTYPES
        or not x.is_contiguous()
        or x.ndim == 0
        or approximate != "tanh"
    ):
        return False
    last_dim = x.shape[-1]
    return (
        last_dim > 0 and last_dim % 2 == 0 and last_dim // 2 <= _MAX_INTERMEDIATE_SIZE
    )


@torch.library.custom_op(
    "vllm::gelu_and_mul_sparse_triton",
    mutates_args=(),
    device_types="cuda",
)
def _gelu_and_mul_sparse_triton_op(x: Tensor, std_multiplier: float) -> Tensor:
    d = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output

    n_rows = x.numel() // x.shape[-1]

    def grid(meta):
        return (n_rows,)

    _gelu_and_mul_sparse_kernel[grid](
        x,
        output,
        d=d,
        std_multiplier=std_multiplier,
        BLOCK_SIZE=triton.next_power_of_2(d),
        num_warps=8,
    )
    return output


@_gelu_and_mul_sparse_triton_op.register_fake
def _gelu_and_mul_sparse_triton_fake(x: Tensor, std_multiplier: float) -> Tensor:
    del std_multiplier
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
    )


@ir.ops.gelu_and_mul_sparse.register_impl(
    "triton",
    supported=HAS_TRITON and current_platform.is_cuda(),
    supports_args=_supports_gelu_and_mul_sparse,
)
def gelu_and_mul_sparse_triton(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> Tensor:
    assert approximate == "tanh"
    return _gelu_and_mul_sparse_triton_op(x, std_multiplier)
