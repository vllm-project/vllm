# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL prefill MoE for MiniMax-M3 MXFP8 weights on gfx950 (fp8 x fp8),
``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``.

The MXFP8 layer (``ModelOptMxFp8FusedMoE``, backend AITER_MXFP8) runs aiter's
a8w8 chain: routing sort, fused per-token fp8 quant, stage-1 GEMM writing the
fp8 intermediate + e8m0 scales, stage-2 GEMM (bf16 ``[M, topk, H]`` partials
or atomics) and the top-k reduction. This package keeps aiter's quant kernel
and the ``moe_flydsl_common`` sort / tile map / bf16 reduce, and replaces the
two GEMMs with the fp8 ports of the a4w4 prefill kernels:

* ``gemm1``: gate/up fp8 GEMM (4-wave 2x2, ``v_mfma_scale_f32_16x16x128_f8f6f4``
  with fp8 operands, AGPR accumulators, 128-K steps) with swiglu-OAI and the
  per-32-column MXFP8 quant of the intermediate fused into the epilogue; reads
  the gate/up-interleaved W13 the AITER_MXFP8 backend stores;
* ``gemm2``: down fp8 GEMM writing bf16 ``[M, topk, H]`` partials
  (``out_mode="bf16"``, reduced by ``moe_flydsl_common.reduce_bf16``) or MXFP8
  partials (``"fp8"``, reduced by ``reduce_fp8``).

Weights: ``w13`` ``[E, 2I, H]`` fp8 e4m3 (``shuffle_weight(is_guinterleave=True,
gate_up=True)``), ``w13_scale`` (``shuffle_scale(..., True, True)``), ``w2``
``[E, H, I]`` (``shuffle_weight``), ``w2_scale`` (``shuffle_scale``): the tensors
``convert_to_fp8_moe_kernel_format`` leaves on the layer. Only their data
pointers are used.

``install_prefill_fast_path`` (``VLLM_ROCM_USE_M3_FLYDSL_PREFILL_MOE``) wraps
the layer's ``quant_method.apply`` and routes every batch above the decode cap
to FlyDSL: ``moe_a8w8_mid`` for ``MIN_MID_TOKENS <= M < MIN_PREFILL_TOKENS``,
this chain from ``MIN_PREFILL_TOKENS`` up.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.ops.moe_a8w8_mid import MIN_MID_TOKENS, a8w8_mid_moe
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.prefill import (
    MAX_PREFILL_TOKENS,
    MIN_PREFILL_TOKENS,
    _get_reduce_bf16,
    _get_sort,
    _get_tile_map,
    _run_compiled,
    block_m_for,
)

logger = init_logger(__name__)

GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
# CTAs sharing one m-block's 24 n-tiles in gemm2 (see ``gemm2.compile_moe_gemm2``).
# gemm2 alone, MI355X: 6 beats 4 by 4-9% up to 16384 tokens (512: 177 -> 167 us,
# 4096: 234 -> 230, 16384: 621 -> 591); 4 wins from 32768 (1069 vs 1077, 65536:
# 2027 vs 2062), where the rotated n-tile sweep already fills the machine.
GEMM2_N_SPLIT = 6
GEMM2_N_SPLIT_LARGE = 4
GEMM2_N_SPLIT_LARGE_FROM_TOKENS = 32768


def gemm2_n_split_for(n_tokens: int) -> int:
    return (
        GEMM2_N_SPLIT_LARGE
        if n_tokens >= GEMM2_N_SPLIT_LARGE_FROM_TOKENS
        else GEMM2_N_SPLIT
    )


def default_out_mode() -> str:
    """gemm2 output mode (see ``gemm2.compile_moe_gemm2``): "bf16" = token-major
    bf16 partials + ``moe_flydsl_common.reduce_bf16`` (deterministic); "fp8" = MXFP8
    partials + ``reduce_fp8`` (deterministic, faster, one more quantization).
    Follows aiter's fp8 route-out switch, ``AITER_FLYDSL_STAGE2_FP8=1``
    (read per call, like aiter); a caller may also pass ``out_mode`` explicitly."""
    return "fp8" if os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1" else "bf16"


_GEMM1_BLOCK_K = 128
_GEMM2_INTERMEDIATE = 768


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 unrolls the K loop by 4 steps of 128 after 4 peeled ones; gemm2's
    pipeline is written for K = 768."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % 256 == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


def supports_batch(x: torch.Tensor) -> bool:
    """Runtime gate for one call of the installed fast path: every batch above
    the decode cap, ``MIN_MID_TOKENS <= M <= MAX_PREFILL_TOKENS``."""
    return (
        x.dim() == 2
        and MIN_MID_TOKENS <= x.shape[0] <= MAX_PREFILL_TOKENS
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


@functools.cache
def _get_gemm1(
    hidden_size: int, intermediate_size: int, num_experts: int, block_m: int
):
    from .gemm1 import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=block_m
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    block_m: int,
    n_split: int,
    out_mode: str,
):
    from .gemm2 import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=n_split,
        sort_block_m=block_m,
        out_mode=out_mode,
    )


@functools.cache
def _get_reduce_fp8(hidden_size: int, topk: int):
    from .reduce_fp8 import compile_moe_reduce_fp8

    return compile_moe_reduce_fp8(H=hidden_size, topk=topk)


def a8w8_prefill_moe(
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
    out: torch.Tensor | None = None,
    out_mode: str | None = None,
) -> torch.Tensor:
    """One MoE layer for ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``:
    stage 1 (sort, aiter fp8 quant, tile map, gemm1), then gemm2 + the reduction
    of ``out_mode`` (default ``default_out_mode()``): "bf16" partials +
    ``moe_flydsl_common.reduce_bf16``, "fp8" partials + ``reduce_fp8``. Returns
    ``[M, hidden_size]`` bf16."""
    from .gemm2 import OUT_MODES, gemm2_grid

    out_mode = default_out_mode() if out_mode is None else out_mode
    assert out_mode in OUT_MODES, out_mode
    n_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    bufs, _a_q, _a_s, h_q, h_s, num_m_blocks = a8w8_prefill_stage1(
        x,
        w13,
        w13_scale,
        topk_weights,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    if out_mode == "fp8":
        gemm2_out = torch.empty(
            (n_tokens * topk, hidden_size), dtype=torch.uint8, device=device
        )
        partial_scale = torch.empty(
            (n_tokens * topk * (hidden_size // 32),), dtype=torch.uint8, device=device
        )
    else:
        gemm2_out = torch.empty(
            (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
        )
        partial_scale = torch.empty((16,), dtype=torch.uint8, device=device)
    num_m_blocks2 = (num_m_blocks * bm) // 128
    n_split = gemm2_n_split_for(n_tokens)
    grid2 = gemm2_grid(num_m_blocks2, n_split)
    # The bf16 partials reach 4.03 GB at 65536 tokens: they are passed by their
    # own element view (a byte view has more than 2^31 elements), and gemm2 /
    # reduce_bf16 address them with i32 byte offsets and record counts that wrap
    # past 2^31 -- the buffer instructions read both as u32, so 4 GB is the limit.
    _run_compiled(
        _get_gemm2(
            hidden_size, intermediate_size, num_experts, topk, bm, n_split, out_mode
        ),
        h_q.view(-1),
        _u8_flat(w2),
        gemm2_out.view(-1),
        h_s,
        _u8_flat(w2_scale),
        partial_scale,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        grid2,
        stream,
    )
    if out_mode == "fp8":
        _run_compiled(
            _get_reduce_fp8(hidden_size, topk),
            gemm2_out.view(-1),
            partial_scale,
            topk_weights.to(torch.float32).contiguous().view(-1),
            out.view(-1),
            n_tokens,
            stream,
        )
        return out
    _run_compiled(
        _get_reduce_bf16(hidden_size, topk),
        gemm2_out.view(-1),
        out.view(-1),
        n_tokens,
        stream,
    )
    return out


def a8w8_prefill_stage1(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
):
    """Sort + aiter's fp8 quant + tile map + gemm1. Returns ``(bufs, a_q, a_s,
    h_q, h_s, num_m_blocks)``: ``h_q`` fp8 ``[rows, I]`` and its e8m0 scales in
    sorted-row order (the layout gemm2 reads)."""
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import SortBuffers
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import tile_map_grid

    n_tokens, hidden = x.shape
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
    assert hidden == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    inter = intermediate_size

    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
    _get_sort(num_experts, topk, bm)(
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)
    )
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(
        x,
        bufs.sorted_ids,
        bufs.num_valid_ids,
        token_num=n_tokens,
        topk=topk,
        block_size=bm,
        quant_dtype=dtypes.fp8,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    _run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        grid1,
        stream,
    )
    h_q = torch.empty((rows, inter), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    _run_compiled(
        _get_gemm1(hidden_size, inter, num_experts, bm),
        _u8_flat(a_q),
        _u8_flat(w13),
        h_q.view(-1),
        _u8_flat(a_s),
        _u8_flat(w13_scale),
        h_s,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        tile_map,
        grid1,
        stream,
    )
    return bufs, a_q, a_s, h_q, h_s, num_m_blocks


def _unsupported_reason(layer) -> str | None:
    """Static checks on a RoutedExperts layer; None when the fast path applies."""
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode import (
        _unsupported_reason as decode_unsupported_reason,
    )

    reason = decode_unsupported_reason(layer)
    if reason is not None and not reason.startswith("shapes "):
        return reason
    if (
        float(layer.swiglu_alpha) != GEMM1_SWIGLU_ALPHA
        or float(layer.swiglu_limit) != GEMM1_SWIGLU_LIMIT
    ):
        return (
            f"swiglu alpha/limit {layer.swiglu_alpha}/{layer.swiglu_limit} != 1.702/7"
        )
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    if not supports_shapes(hidden, inter):
        return f"shapes hidden={hidden} intermediate={inter} not tiled by the kernels"
    return None


def _fast_path(layer, x, topk_weights, topk_ids, hidden: int, inter: int):
    chain = a8w8_mid_moe if x.shape[0] < MIN_PREFILL_TOKENS else a8w8_prefill_moe
    return chain(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        topk_weights,
        topk_ids,
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=layer.w13_weight.shape[0],
    )


# One batch size per kernel configuration the profile batch does not compile
# itself (mid sort blocks of 32 / 64 / 128 rows, prefill blocks of 128 / 256
# rows with gemm2's small-batch n-split): run once on prefixes of the first
# batch that reaches the prefill range, i.e. the model runner's profile run,
# so no request waits on a compile.
_WARM_UP_TOKENS = (512, 768, 1536, 3072, 16384)
_warmed = False


def _warm_up(layer, x, topk_weights, topk_ids, hidden: int, inter: int) -> None:
    for m in _WARM_UP_TOKENS:
        if m < x.shape[0]:
            logger.debug("M3 FlyDSL a8w8 MoE warm-up: %d tokens", m)
            _fast_path(layer, x[:m], topk_weights[:m], topk_ids[:m], hidden, inter)


def install_prefill_fast_path(experts, prefix: str = "") -> bool:
    """Route every ``MIN_MID_TOKENS <= M <= MAX_PREFILL_TOKENS`` call of a
    MiniMax-M3 MXFP8 MoE layer to the FlyDSL a8w8 chains: ``moe_a8w8_mid``
    below ``MIN_PREFILL_TOKENS``, this package from there up.

    ``experts`` is what ``FusedMoEFactory`` returned or the RoutedExperts layer
    itself. Wraps the layer's ``quant_method.apply`` (on top of the decode
    package's wrapper when that is installed, so that no batch size is left to
    aiter); a call outside the gate (unfused shared experts, an unexpected
    dtype or layout) goes to the wrapped implementation unchanged. Returns
    True when installed.
    """
    layer = getattr(experts, "routed_experts", experts)
    try:
        reason = _unsupported_reason(layer)
    except Exception as exc:  # a layer/config shape this gate does not know
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        logger.info_once(
            "M3 FlyDSL a8w8 mid/prefill MoE not used for %s: %s",
            prefix or "experts",
            reason,
        )
        return False
    qm = layer.quant_method
    if getattr(qm, "_m3_prefill_fast_path", False):
        return True

    orig_apply = qm.apply
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition

    def apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        **kwargs,
    ):
        global _warmed
        if shared_experts is not None or kwargs or not supports_batch(x):
            return orig_apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                **kwargs,
            )
        if not _warmed and x.shape[0] >= MIN_PREFILL_TOKENS:
            _warmed = True
            _warm_up(layer, x, topk_weights, topk_ids, hidden, inter)
        return _fast_path(layer, x, topk_weights, topk_ids, hidden, inter)

    qm.apply = apply
    qm._m3_prefill_fast_path = True
    logger.info_once(
        "M3 FlyDSL a8w8 mid/prefill MoE installed (MXFP8, %d <= M <= %d)",
        MIN_MID_TOKENS,
        MAX_PREFILL_TOKENS,
    )
    logger.debug("M3 FlyDSL a8w8 mid/prefill MoE installed for %s", prefix or "experts")
    return True
