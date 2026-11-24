# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiLU-and-mul (optionally clamped) fused with the MXFP8 quant of its output.

On gfx950 the MXFP8 linear after the shared expert's activation quantizes the
bf16 activation in a separate launch; this writes the e4m3 values and row-major
`[M, D / 32]` E8M0 scales directly, bit-identical to `SiluAndMul` /
`SiluAndMulWithClamp` followed by `mxfp8_e4m3_quantize`.
"""

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _silu_and_mul_mxfp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    D: tl.constexpr,
    LIMIT: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rows[:, None] < M
    rows64 = rows[:, None].to(tl.int64)
    gate = tl.load(x_ptr + rows64 * 2 * D + cols[None, :], mask=mask, other=0.0)
    up = tl.load(x_ptr + rows64 * 2 * D + D + cols[None, :], mask=mask, other=0.0)
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    # Round to the activation's bf16 first, as the unfused path materializes it.
    y = (gate / (1.0 + tl.exp(-gate)) * up).to(x_ptr.dtype.element_ty)
    y = tl.reshape(y.to(tl.float32), (BLOCK_M, BLOCK_N // 32, 32))
    # Same arithmetic as _mxfp8_quant_triton_kernel (mxfp8_e4m3_quantize).
    amax = tl.maximum(tl.max(tl.abs(y), axis=2), TINY)
    sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    q = tl.reshape(y * tl.exp2(127.0 - sb)[:, :, None], (BLOCK_M, BLOCK_N))
    tl.store(
        q_ptr + rows64 * D + cols[None, :], q.to(q_ptr.dtype.element_ty), mask=mask
    )
    groups = tl.program_id(1) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    tl.store(s_ptr + rows64 * (D // 32) + groups[None, :], sb.to(tl.uint8), mask=mask)


def _config(num_tokens: int, d: int) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps), tuned on MI355X for d = 2304."""
    if num_tokens <= 256:
        block_m, block_n, warps = 1, 256, 1
    elif num_tokens <= 4096:
        block_m, block_n, warps = 16, 128, 4
    else:
        block_m, block_n, warps = 32, 256, 8
    while d % block_n:
        block_n //= 2
    return block_m, block_n, warps


def silu_and_mul_mxfp8(
    x: torch.Tensor, swiglu_limit: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[M, 2 * D]` gate/up -> MXFP8 `[M, D]` e4m3 values and `[M, D / 32]`
    uint8 E8M0 scales of `silu(clamp(gate)) * clamp(up)`."""
    assert x.dim() == 2 and x.is_contiguous()
    num_tokens, d = x.shape[0], x.shape[1] // 2
    assert d % MXFP8_BLOCK_SIZE == 0
    values = torch.empty((num_tokens, d), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (num_tokens, d // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    if num_tokens == 0:
        return values, scales
    block_m, block_n, warps = _config(num_tokens, d)
    _silu_and_mul_mxfp8_kernel[(triton.cdiv(num_tokens, block_m), d // block_n)](
        x,
        values,
        scales,
        num_tokens,
        D=d,
        LIMIT=0.0 if swiglu_limit is None else float(swiglu_limit),
        HAS_LIMIT=swiglu_limit is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        FP8_MAX=float(torch.finfo(MXFP8_VALUE_DTYPE).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        num_warps=warps,
    )
    return values, scales
