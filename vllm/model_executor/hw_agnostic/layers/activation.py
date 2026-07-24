# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.model_executor.hw_agnostic.custom_op import CustomOp
from vllm.triton_utils import tl, triton


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


@CustomOp.register("silu_and_mul_with_clamp")
class SiluAndMulWithClamp(CustomOp):
    """SwiGLU with input clamping. ``d = x.shape[-1] // 2``;
    ``out = silu(clamp(x[:d], max=L)) * clamp(x[d:], min=-L, max=L)``."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


@triton.jit
def _swiglustep_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    tl.store(o_row_ptr + offsets, result, mask=mask)


def swiglustep_and_mul_triton(
    output: torch.Tensor, input: torch.Tensor, limit: float = 7.0
):
    b, n = input.shape
    assert input.ndim == 2
    assert n % 2 == 0
    d = n // 2

    def grid(meta):
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    _swiglustep_and_mul_kernel[grid](
        output,
        output.stride(0),
        input,
        input.stride(0),
        limit=limit,
        d=d,
        BLOCK_SIZE=1024,
    )
