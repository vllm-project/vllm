# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL MoE for MiniMax-M3 MXFP8 weights on gfx950 (TP4 shapes: hidden 6144,
intermediate 768 per rank). ``mxfp8_moe`` picks the chain by batch size: ``decode``
(bf16 x fp8, M <= 256), ``mid`` (fp8 x fp8, 257..3071, atomic output), ``prefill``
(fp8 x fp8, 3072..65536, partials + top-k reduction). The weights are the tensors
``ModelOptMxFp8FusedMoE`` stores for AITER_MXFP8 (``shuffle_mxfp8_moe_weights``);
``MiniMaxM3FlyDSLMxfp8Experts`` (``fused_moe/experts``) runs this in place of
``AiterMxfp8Experts``. Shared sort / tile map / reduce helpers: ``moe_flydsl_common``.
"""

from __future__ import annotations

import torch

from . import decode, mid, prefill

MAX_DECODE_TOKENS = decode.MAX_DECODE_TOKENS
MIN_MID_TOKENS = mid.MIN_MID_TOKENS
MAX_MID_TOKENS = mid.MAX_MID_TOKENS
MIN_PREFILL_TOKENS = prefill.MIN_PREFILL_TOKENS
MAX_PREFILL_TOKENS = prefill.MAX_PREFILL_TOKENS
MAX_TOKENS = MAX_PREFILL_TOKENS
assert MAX_DECODE_TOKENS + 1 == MIN_MID_TOKENS
assert MAX_MID_TOKENS + 1 == MIN_PREFILL_TOKENS

# the fp8 chains fuse SwiGLU-OAI with these constants (decode takes them as arguments)
SWIGLU_ALPHA = prefill.GEMM1_SWIGLU_ALPHA
SWIGLU_LIMIT = prefill.GEMM1_SWIGLU_LIMIT

# One batch size per kernel configuration the profile batch does not compile
# itself (mid sort blocks of 32 / 64 / 128 rows, prefill blocks of 128 / 256
# rows with gemm2's small-batch n-split): the first prefill-range call, i.e.
# the model runner's profile run, also runs these on prefixes of its batch so
# that no request waits on a compile.
_WARM_UP_TOKENS = (512, 768, 1536, 3072, 16384)
_warmed = False


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """Layer shapes all three chains are written for."""
    return decode.supports_shapes(
        hidden_size, intermediate_size
    ) and prefill.supports_shapes(hidden_size, intermediate_size)


def supports_batch(x: torch.Tensor) -> bool:
    """Runtime gate for one call: ``[M, hidden]`` bf16, contiguous, up to
    ``MAX_TOKENS`` rows."""
    return (
        x.dim() == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _dispatch(
    x,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    hidden_size,
    intermediate_size,
    num_experts,
    swiglu_alpha,
    swiglu_limit,
    fused_shared_expert,
    out,
):
    n_tokens = x.shape[0]
    if n_tokens <= MAX_DECODE_TOKENS:
        return decode.a16w8_decode_moe(
            x,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            fused_shared_expert=fused_shared_expert,
            out=out,
        )
    assert swiglu_alpha == SWIGLU_ALPHA and swiglu_limit == SWIGLU_LIMIT, (
        swiglu_alpha,
        swiglu_limit,
    )
    chain = mid.a8w8_mid_moe if n_tokens <= MAX_MID_TOKENS else prefill.a8w8_prefill_moe
    return chain(
        x,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        out=out,
    )


def mxfp8_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    swiglu_alpha: float = SWIGLU_ALPHA,
    swiglu_limit: float = SWIGLU_LIMIT,
    fused_shared_expert: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer -> ``[M, hidden_size]`` bf16 (into ``out`` when given). ``w13`` /
    ``w2`` (fp8 e4m3) and their e8m0 scales are the layer tensors after
    ``shuffle_mxfp8_moe_weights``. ``fused_shared_expert``: the last expert is
    aiter's fused shared expert, routed by every token (decode may then use its
    wide sort layout); False for ModelOpt MXFP8, where vLLM keeps it separate.
    """
    global _warmed
    kw = dict(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        fused_shared_expert=fused_shared_expert,
    )
    n_tokens = x.shape[0]
    assert 1 <= n_tokens <= MAX_TOKENS, n_tokens
    if not _warmed and n_tokens >= MIN_PREFILL_TOKENS:
        _warmed = True
        for m in _WARM_UP_TOKENS:
            if m < n_tokens:
                _dispatch(
                    x[:m],
                    w13,
                    w13_scale,
                    w2,
                    w2_scale,
                    topk_weights[:m],
                    topk_ids[:m],
                    out=None,
                    **kw,
                )
    return _dispatch(
        x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, out=out, **kw
    )


__all__ = [
    "MAX_DECODE_TOKENS",
    "MAX_MID_TOKENS",
    "MAX_PREFILL_TOKENS",
    "MAX_TOKENS",
    "MIN_MID_TOKENS",
    "MIN_PREFILL_TOKENS",
    "SWIGLU_ALPHA",
    "SWIGLU_LIMIT",
    "decode",
    "mid",
    "mxfp8_moe",
    "prefill",
    "supports_batch",
    "supports_shapes",
]
