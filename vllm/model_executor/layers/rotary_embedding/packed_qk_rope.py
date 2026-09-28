# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place interleaved RoPE on the Q/K slices of a packed QKV buffer."""

import torch

from vllm.triton_utils import tl, triton


# seqlen is not specialized: image sizes vary per request.
@triton.jit(do_not_specialize=["seqlen"])
def _packed_qk_rope_kernel(
    qk_ptr,  # (seqlen, 2 * nheads, headdim) strided view of the packed QKV
    freqs_ptr,  # (seqlen, rotary_dim) fp32 re/im view of the complex freqs
    seqlen,
    qk_row_stride,
    qk_head_stride,
    rotary_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(rotary_dim)
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < seqlen

    base = qk_ptr + pid_h * qk_head_stride + rm[:, None] * qk_row_stride
    mask_x = mask_m[:, None] & (rk < rotary_dim)[None, :]
    x = tl.load(base + rk[None, :], mask=mask_x, other=0.0).to(tl.float32)
    f = tl.load(freqs_ptr + rm[:, None] * rotary_dim + rk[None, :], mask=mask_x)
    cos, sin = tl.split(tl.reshape(f, (BLOCK_M, BLOCK_K // 2, 2)))

    x0, x1 = tl.split(tl.reshape(x, (BLOCK_M, BLOCK_K // 2, 2)))
    o0 = x0 * cos - x1 * sin
    o1 = x1 * cos + x0 * sin
    tl.store(base + rk[None, :], tl.interleave(o0, o1), mask=mask_x)


def packed_qk_rope_(xqkv: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    """Rotate Q and K in place inside a packed QKV buffer.

    Args:
        xqkv: contiguous (seqlen, 3, nheads, headdim); only the Q and K
            slices are rotated.
        freqs_cis: contiguous (seqlen, headdim // 2) complex64 rotary freqs,
            read through its interleaved re/im fp32 view.

    Bitwise identical to ``ApplyRotaryEmb(enable_fp32_compute=True)`` applied
    to Q and K separately, but in one kernel with ~5x less memory traffic.
    Requires triton: callers must check HAS_TRITON and use the unfused path
    otherwise. Contract violations raise.

    """
    assert xqkv.ndim == 4 and xqkv.size(1) == 3, xqkv.shape
    assert xqkv.is_contiguous(), xqkv.shape
    assert freqs_cis.ndim == 2 and freqs_cis.is_contiguous(), freqs_cis.shape
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype

    seq_length, _, num_heads, head_dim = xqkv.shape
    rotary_dim = freqs_cis.size(-1) * 2
    assert freqs_cis.size(0) == seq_length, (freqs_cis.shape, xqkv.shape)
    assert rotary_dim == head_dim, (freqs_cis.shape, xqkv.shape)  # full rotation
    qk = xqkv.as_strided(
        (seq_length, 2 * num_heads, head_dim),
        (3 * num_heads * head_dim, head_dim, 1),
    )

    BLOCK_M = 4
    # seqlen on grid.x: HIP caps gridDim.y at 65535.
    grid = (triton.cdiv(seq_length, BLOCK_M), 2 * num_heads)
    _packed_qk_rope_kernel[grid](
        qk,
        freqs_cis.view(torch.float32),
        seq_length,
        qk.stride(0),
        qk.stride(1),
        rotary_dim=rotary_dim,
        BLOCK_M=BLOCK_M,
        num_warps=2 if rotary_dim <= 64 else 4,
    )
