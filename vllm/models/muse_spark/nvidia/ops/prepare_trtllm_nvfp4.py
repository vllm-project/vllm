# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-expert RMSNorm fused with the TRT-LLM MoE's NVFP4 input quantization.

Matches ``MuseSpark1xRMSNorm`` (one BF16 rounding) followed by
``scaled_fp4_quant(..., is_sf_swizzled_layout=False)`` with the static global
scale, and reads the pooler output either contiguous or in the rank-block
layout ``[tp, M, H / tp]`` left by a column all-gather.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BLOCK_K = 1024


@functools.cache
def _enable_pdl() -> bool:
    return current_platform.is_cuda() and current_platform.is_arch_support_pdl()


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quantize_nvfp4_global_scale(values, global_scale, BLOCK_K: tl.constexpr):
    """``scaled_fp4_quant``'s arithmetic (nvfp4_utils.cuh), linear scales."""
    groups = tl.reshape(values, [BLOCK_K // 16, 16])
    amax = tl.max(tl.abs(groups), axis=1)
    sf = (global_scale * (amax * _rcp_approx(6.0))).to(tl.float8e4nv)
    sf_f32 = sf.to(tl.float32)
    out_scale = tl.where(
        sf_f32 != 0.0, _rcp_approx(sf_f32 * _rcp_approx(global_scale)), 0.0
    )
    scaled = groups * out_scale[:, None]
    magnitude = tl.abs(scaled)
    # Round to nearest even on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6},
    # saturating, as cvt.rn.satfinite.e2m1x2.f32 does.
    code = (
        (magnitude > 0.25).to(tl.uint8)
        + (magnitude >= 0.75).to(tl.uint8)
        + (magnitude > 1.25).to(tl.uint8)
        + (magnitude >= 1.75).to(tl.uint8)
        + (magnitude > 2.5).to(tl.uint8)
        + (magnitude >= 3.5).to(tl.uint8)
        + (magnitude > 5.0).to(tl.uint8)
    )
    code |= (scaled < 0).to(tl.uint8) << 3
    lo, hi = tl.split(tl.reshape(code, [BLOCK_K // 2, 2]))
    return (lo & 0x0F) | (hi << 4), sf


@triton.jit
def _norm_quant_trtllm_nvfp4_kernel(
    x,
    weight,
    global_scale,
    packed_out,
    sf_out,
    num_tokens,
    eps,
    gain_center,
    HIDDEN: tl.constexpr,
    SHARD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    token = tl.program_id(0)
    block = tl.program_id(1)
    # Column d of the token lives in shard d // SHARD of the rank-block layout.
    d = tl.arange(0, HIDDEN)
    row = tl.load(x + ((d // SHARD) * num_tokens + token) * SHARD + d % SHARD)
    row = row.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(row * row, axis=0) * (1.0 / HIDDEN) + eps)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(x + ((k // SHARD) * num_tokens + token) * SHARD + k % SHARD)
    gamma = tl.load(weight + k).to(tl.float32) + gain_center
    # Round to BF16 as the unfused norm -> quantize path does.
    values = ((values.to(tl.float32) * rstd) * gamma).to(tl.bfloat16).to(tl.float32)
    packed, sf = _quantize_nvfp4_global_scale(values, tl.load(global_scale), BLOCK_K)
    tl.store(
        packed_out
        + token * (HIDDEN // 2)
        + block * (BLOCK_K // 2)
        + tl.arange(0, BLOCK_K // 2),
        packed,
    )
    tl.store(
        sf_out
        + token * (HIDDEN // 16)
        + block * (BLOCK_K // 16)
        + tl.arange(0, BLOCK_K // 16),
        sf.to(tl.uint8, bitcast=True),
    )


def norm_quant_trtllm_nvfp4(
    pooled: torch.Tensor,
    num_tokens: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    gain_center: float,
    global_scale: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + NVFP4 quant of ``[M, H]`` pooled states, in any rank-block layout.

    ``pooled`` is ``[M, H]`` or the ``[tp * M, H / tp]`` output of a dim-0
    all-gather of column shards. Returns packed ``[M, H/2]`` uint8 values and
    linear ``[M, H/16]`` uint8 (E4M3) block scales.
    """
    hidden = norm_weight.shape[0]
    shard = pooled.shape[1]
    assert pooled.dtype == torch.bfloat16 and pooled.is_contiguous()
    assert pooled.shape[0] * shard == num_tokens * hidden and hidden % _BLOCK_K == 0
    if out is None:
        packed = torch.empty(
            num_tokens, hidden // 2, dtype=torch.uint8, device=pooled.device
        )
        sf = torch.empty(
            num_tokens, hidden // 16, dtype=torch.uint8, device=pooled.device
        )
    else:
        packed, sf = out
        assert packed.shape == (num_tokens, hidden // 2)
        assert sf.shape == (num_tokens, hidden // 16)
        assert packed.dtype == sf.dtype == torch.uint8
        assert packed.is_contiguous() and sf.is_contiguous()
        assert packed.device == sf.device == pooled.device
    if num_tokens == 0:
        return packed, sf
    enable_pdl = _enable_pdl()
    _norm_quant_trtllm_nvfp4_kernel[(num_tokens, hidden // _BLOCK_K)](
        pooled,
        norm_weight,
        global_scale,
        packed,
        sf,
        num_tokens,
        norm_eps,
        gain_center,
        HIDDEN=hidden,
        SHARD=shard,
        BLOCK_K=_BLOCK_K,
        USE_PDL=enable_pdl,
        num_warps=8,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return packed, sf
