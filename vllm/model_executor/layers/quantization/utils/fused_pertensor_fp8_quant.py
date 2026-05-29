# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused single-launch per-tensor-dynamic FP8-e4m3 quant for ROCm/gfx950.

Replaces aiter's 3-kernel triplet with a single Triton launch in the decode
regime. Output is byte-identical to the aiter triplet (accuracy-neutral).
"""

import torch

from vllm.triton_utils import tl, triton

FP8_MAX = 448.0  # OCP E4M3 max on gfx950 (not FNUZ)
_FP8_MAX = tl.constexpr(448.0)
# recip-mul (not divide) reproduces aiter's folded scale bit-for-bit
_INV_FP8_MAX = tl.constexpr(1.0 / 448.0)

SINGLE_BLOCK_CAP = 40960  # single-block 1-launch cap
MULTIBLOCK_CAP = 8_388_608  # 2-launch cap; above this -> aiter fallback (prefill)

_SB_BLOCK = 16384  # single-block grid-stride chunk
_NPART = 256  # amax-partial count == grid width (MI355X has 256 CUs)
_MB_BLOCK = 4096  # multi-block grid-stride chunk


@triton.jit
def _fused_pt_quant_singleblock(
    x_ptr,  # *bf16 [N] (flattened)
    out_ptr,  # *fp8e4m3 [N]
    scale_ptr,  # *f32 [1]
    N,
    BLOCK: tl.constexpr,
):
    # pass 1: global amax
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    off = tl.arange(0, BLOCK)
    i = 0
    while i < N:
        idx = i + off
        mask = idx < N
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
        i += BLOCK
    scale = tl.max(amax, axis=0) * _INV_FP8_MAX
    tl.store(scale_ptr, scale)
    inv_scale = 1.0 / scale

    # pass 2: quantize
    i = 0
    while i < N:
        idx = i + off
        mask = idx < N
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        q = v * inv_scale
        q = tl.minimum(tl.maximum(q, -_FP8_MAX), _FP8_MAX)
        tl.store(out_ptr + idx, q.to(tl.float8e4nv), mask=mask)
        i += BLOCK


@triton.jit
def _amax_partial(x_ptr, partial_ptr, N, BLOCK: tl.constexpr):
    # each block writes its own partial slot; max is order-independent -> byte-exact
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    off = tl.arange(0, BLOCK)
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    i = pid * BLOCK
    stride = nprog * BLOCK
    while i < N:
        idx = i + off
        mask = idx < N
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
        i += stride
    tl.store(partial_ptr + pid, tl.max(amax, axis=0))


@triton.jit
def _quant_from_partials(
    x_ptr,
    out_ptr,
    scale_ptr,
    partial_ptr,
    N,
    NPART: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    # global amax from partials (order-independent max -> byte-exact)
    poff = tl.arange(0, NPART)
    amax = tl.max(tl.load(partial_ptr + poff), axis=0)
    scale = amax * _INV_FP8_MAX
    if pid == 0:
        tl.store(scale_ptr, scale)
    inv_scale = 1.0 / scale
    off = tl.arange(0, BLOCK)
    i = pid * BLOCK
    stride = nprog * BLOCK
    while i < N:
        idx = i + off
        mask = idx < N
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        q = v * inv_scale
        q = tl.minimum(tl.maximum(q, -_FP8_MAX), _FP8_MAX)
        tl.store(out_ptr + idx, q.to(tl.float8e4nv), mask=mask)
        i += stride


def fused_per_tensor_dynamic_fp8_quant(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic FP8-e4m3 quant; returns ``(x_fp8, scale[1])``.

    Byte-identical to ``aiter.dynamic_per_tensor_quant`` for every dispatch path.
    """
    x = x.contiguous()
    N = x.numel()
    if out is None:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty(1, dtype=torch.float32, device=x.device)

    if N <= SINGLE_BLOCK_CAP:
        _fused_pt_quant_singleblock[(1,)](x, out, scale, N, BLOCK=_SB_BLOCK)
        return out, scale

    if N <= MULTIBLOCK_CAP:
        partials = torch.empty(_NPART, dtype=torch.float32, device=x.device)
        _amax_partial[(_NPART,)](x, partials, N, BLOCK=_MB_BLOCK)
        _quant_from_partials[(_NPART,)](
            x, out, scale, partials, N, NPART=_NPART, BLOCK=_MB_BLOCK
        )
        return out, scale

    # prefill: aiter fallback
    import aiter

    aiter.dynamic_per_tensor_quant(out, x, scale)
    return out, scale
