# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark context-KV write for all draft layers in one Triton launch (ROCm).

The ROCm counterpart of the CUDA op behind `DSparkContextKVWriter`: from the
fused projection output `all_kv` `[num_ctx, L, 2, nkv, 128]` it norms K
(Gemma RMSNorm), applies partial NeoX RoPE and stores K and V into every
layer's paged cache `[num_blocks, nkv, block_size, 2 * 128]` (K at
`[..., :128]`, V at `[..., 128:]`); slots < 0 are not written. The
layers' caches and scales are addressed through tables of their addresses,
which may lie anywhere relative to each other.

Each step rounds as vLLM's ROCm ops do, so the cache bytes equal theirs: the
norm is `gemma_rmsnorm`'s (an fp32 sum over the 128-wide head, the output
rounded to bf16), RoPE is `rotary_embedding`'s (fp32 from the bf16 inputs
and cos / sin, one rounding to bf16) and the fp8 store is
`triton_reshape_and_cache_flash`'s (`x / scale` cast to the cache's
e4m3 type). One program takes a token, a layer and a KV head.
"""

import torch

from vllm.triton_utils import tl, triton

_HEAD_DIM = 128


@triton.jit
def _context_kv_kernel(
    kv_ptr,  # all_kv [N, L, 2, nkv, 128] bf16
    s_tok,
    s_layer,
    s_kv,
    s_head,
    knorm_ptr,  # [L, 128] bf16 K-norm weights
    cos_sin_ptr,  # [max_pos, ROT] bf16: cos | sin
    pos_ptr,  # [N] int64
    slot_ptr,  # [N] int64
    cache_ptrs,  # [L] int64 addresses of the layers' caches
    scale_ptrs,  # [2, L] int64 addresses of the k scales, then the v scales
    nkv,
    block_size,
    eps,
    L: tl.constexpr,
    ROT: tl.constexpr,
    FP8: tl.constexpr,
    ELEM: tl.constexpr,  # the cache's element type
):
    t = tl.program_id(0)
    layer = tl.program_id(1)
    h = tl.program_id(2)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    src = kv_ptr + t * s_tok + layer * s_layer + h * s_head
    w_ptr = knorm_ptr + layer * 128
    d = tl.arange(0, 128)
    # gemma_rmsnorm: the head's statistics, then x * rstd * (1 + w) to bf16
    k = tl.load(src + d).to(tl.float32)
    var = tl.sum(k * k, axis=0) / 128
    rstd = 1.0 / tl.sqrt(var + eps)
    kn = (k * rstd * (1.0 + tl.load(w_ptr + d).to(tl.float32))).to(tl.bfloat16)
    # rotary_embedding on the leading ROT dims (NeoX pairs i, i + ROT / 2)
    HALF: tl.constexpr = ROT // 2
    i = tl.arange(0, HALF)
    x = tl.load(src + i).to(tl.float32) * rstd
    x = (x * (1.0 + tl.load(w_ptr + i).to(tl.float32))).to(tl.bfloat16)
    y = tl.load(src + HALF + i).to(tl.float32) * rstd
    y = (y * (1.0 + tl.load(w_ptr + HALF + i).to(tl.float32))).to(tl.bfloat16)
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    pos = tl.load(pos_ptr + t)
    cos = tl.load(cos_sin_ptr + pos * ROT + i).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * ROT + HALF + i).to(tl.float32)
    xo = (x * cos - y * sin).to(tl.bfloat16)
    yo = (y * cos + x * sin).to(tl.bfloat16)
    v = tl.load(src + s_kv + d)
    blk = slot // block_size
    off = slot % block_size
    cache = tl.load(cache_ptrs + layer).to(tl.pointer_type(ELEM))
    dst = cache + ((blk * nkv + h) * block_size + off) * (2 * 128)
    if FP8:
        # triton_reshape_and_cache_flash: x / scale, cast by the fp8 store
        ks = tl.load(tl.load(scale_ptrs + layer).to(tl.pointer_type(tl.float32)))
        vs = tl.load(tl.load(scale_ptrs + L + layer).to(tl.pointer_type(tl.float32)))
        tl.store(dst + i, xo / ks)
        tl.store(dst + HALF + i, yo / ks)
        tl.store(dst + d, kn / ks, mask=d >= ROT)
        tl.store(dst + 128 + d, v / vs)
    else:
        tl.store(dst + i, xo)
        tl.store(dst + HALF + i, yo)
        tl.store(dst + d, kn, mask=d >= ROT)
        tl.store(dst + 128 + d, v)


def context_kv(
    all_kv: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    slot_mapping: torch.Tensor,
    cache_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    block_size: int,
    fp8: bool,
) -> None:
    """Norm, rope and store the context K/V of every layer into the caches at
    `cache_ptrs` (OCP e4m3 for `fp8`, else bf16)."""
    num_ctx, num_layers, _, nkv, head_dim = all_kv.shape
    assert head_dim == _HEAD_DIM and all_kv.stride(-1) == 1
    _context_kv_kernel[(num_ctx, num_layers, nkv)](
        all_kv,
        all_kv.stride(0),
        all_kv.stride(1),
        all_kv.stride(2),
        all_kv.stride(3),
        k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        cache_ptrs,
        scale_ptrs,
        nkv,
        block_size,
        eps,
        L=num_layers,
        ROT=rotary_dim,
        FP8=fp8,
        ELEM=tl.float8e4nv if fp8 else tl.bfloat16,
        num_warps=4,
    )
