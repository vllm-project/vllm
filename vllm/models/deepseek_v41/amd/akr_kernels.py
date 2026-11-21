# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 routes' only door to `amd_kernels`.

Calls only the shape-agnostic op entries of `amd_kernels.<category>` (and the
MXFP8 dense linears of `amd_kernels.gemm.mxfp8_common.gemm`), never a shape
package, so another shape needs no change here. Each entry reads its shape
from the tensors; where `<op>_blocker` returns a reason the call is not
covered: the reason is logged once and the wrapper returns None (or False),
having launched nothing, and the caller runs its stock path. Imports are lazy,
so vLLM imports without `amd_kernels`.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

logger = init_logger(__name__)

BF16 = torch.bfloat16


def _gemm():
    import amd_kernels.gemm as m

    return m


def _linear():
    from amd_kernels.gemm.mxfp8_common import gemm

    return gemm


def _attention():
    import amd_kernels.attention as m

    return m


def _indexer():
    import amd_kernels.indexer as m

    return m


def _moe():
    import amd_kernels.moe as m

    return m


def _norm():
    import amd_kernels.norm as m

    return m


def _activation():
    import amd_kernels.activation as m

    return m


def _sampling():
    import amd_kernels.sampling as m

    return m


def _covered(op: str, reason: str | None) -> bool:
    if reason is None:
        return True
    logger.info_once("amd_kernels %s: stock path (%s)", op, reason)
    return False


# ---------------------------------------------------------------- MXFP8 weights


class Mxfp8Weight:
    """A linear's MXFP8 weight in the entries' layout: AITER's (16, 16)
    preshuffle and shuffled E8M0 scales `[N, K / 32]`."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight = weight
        self.scale = scale
        self.n, self.k = weight.shape


def _row_scales(linear) -> torch.Tensor:
    n, k = linear.weight.shape
    scale = linear.weight_scale.data.view(torch.uint8)
    if scale.shape == (n // 32, k // 32):  # 32 x 32 checkpoint blocks
        scale = scale.repeat_interleave(32, dim=0)
    return scale[:n, : k // 32].contiguous()


def mxfp8_weight(linear) -> Mxfp8Weight | None:
    """The stock linear's preshuffled weight as is (no copy) and its scales in
    the entries' layout; None for a weight that is not MXFP8."""
    from aiter.ops.shuffle import shuffle_scale_a16w4

    w = getattr(linear, "weight", None)
    if (
        w is None
        or w.dtype != torch.float8_e4m3fn
        or w.dim() != 2
        or getattr(linear, "weight_scale", None) is None
    ):
        return None
    scale = _row_scales(linear)
    shuffled = getattr(linear, "weight_preshuffled", None)
    if shuffled is None:
        return Mxfp8Weight(*_linear().preshuffle_mxfp8_weight(w.data, scale))
    return Mxfp8Weight(shuffled, shuffle_scale_a16w4(scale, 1, False))


def mxfp8_dense_weight(linear) -> Mxfp8Weight | None:
    """`mxfp8_weight` for a shape the dense entry has tuned rows for."""
    w = getattr(linear, "weight", None)
    if w is None or tuple(w.shape) not in _linear().tuned_shapes(BF16):
        return None
    weight = mxfp8_weight(linear)
    if weight is not None:
        _linear().warmup_mxfp8_gemm_weight(weight.weight, weight.scale, 1024, BF16)
    return weight


def mxfp8_dense(
    x: torch.Tensor | QuantizedActivation, w: Mxfp8Weight
) -> torch.Tensor | None:
    """`x @ W.T` in bf16 where the dense entry has a winning row for `x`'s row
    count, else None."""
    gemm = _linear()
    if isinstance(x, QuantizedActivation):
        if gemm.mxfp8_linear_blocker(x.data.shape[0], w.n, w.k, BF16, True):
            return None
        return gemm.mxfp8_gemm(x.data, x.scale, w.weight, w.scale, BF16)
    if gemm.mxfp8_linear_blocker(x.shape[0], w.n, w.k, BF16, False):
        return None
    return gemm.mxfp8_linear(x, w.weight, w.scale, BF16)


# ---------------------------------------------------------------- MoE block


def router_weight(gate: torch.Tensor, top_k: int) -> torch.Tensor | None:
    moe = _moe()
    experts, hidden = gate.shape
    if (hidden, experts, top_k) not in moe.router_topk_shapes():
        return None
    return moe.router_shuffle_gate(gate, top_k)


def router_warmup(gate: torch.Tensor, top_k: int) -> None:
    experts, hidden = gate.shape
    _moe().warmup_router_topk(hidden, experts, top_k, gate.device)


def router_topk(x, gate, bias, scale, top_k, is_padding):
    return _moe().router_topk(
        x, gate, bias, top_k, scale, is_padding=is_padding, weight_shuffled=True
    )


def shared_expert_ok(gate_up: Mxfp8Weight, inter: int) -> bool:
    return (gate_up.n, gate_up.k) in _gemm().mxfp8_gate_up_swiglu_mxfp8_shapes() and (
        inter in _activation().clamp_silu_mul_mxfp8_shapes()
    )


def mxfp8_gate_up_swiglu(x, w: Mxfp8Weight):
    """Gate_up GEMM, clamped SwiGLU (limit 10) and MXFP8 out: (codes, scales)."""
    gemm = _gemm()
    if isinstance(x, QuantizedActivation):
        m, prequant = x.data.shape[0], True
    else:
        m, prequant = x.shape[0], False
    reason = gemm.mxfp8_gate_up_swiglu_mxfp8_blocker(m, w.n, w.k, prequant)
    if not _covered("gate_up_swiglu", reason):
        return None
    if prequant:
        return gemm.mxfp8_gate_up_swiglu_mxfp8_prequant(
            x.data, x.scale, w.weight, w.scale
        )
    return gemm.mxfp8_gate_up_swiglu_mxfp8(x, w.weight, w.scale)


def clamp_silu_mul_mxfp8(gate_up: torch.Tensor, limit: float):
    return _activation().clamp_silu_mul_mxfp8(gate_up, limit)


def warmup_clamp_silu_mul(inter: int, device: torch.device) -> None:
    _activation().warmup_clamp_silu_mul_mxfp8(inter, device)


def fused_ep_unsupported_reason(experts, max_tokens: int) -> str | None:
    return _moe().expert_parallel_blocker(
        experts.ep_size,
        experts.hidden_size,
        experts.intermediate_size,
        experts.top_k,
        experts.num_local_experts,
        max_tokens,
    )


def fused_ep_context(experts) -> Any:
    """The expert-parallel pools of this process (a collective)."""
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    return _moe().expert_parallel(
        ep.rank_in_group,
        ep.world_size,
        ep.cpu_group,
        torch.device("cuda", torch.accelerator.current_device_index()),
        hidden=experts.hidden_size,
        inter=experts.intermediate_size,
        top_k=experts.top_k,
        experts_per_rank=experts.num_local_experts,
    )


def fused_ep_weights(experts) -> Any:
    """Views of the layer's MegaMoE tensors, which AITER already shuffled into
    the kernel's layout (no copy)."""
    u8 = torch.uint8
    return _moe().fused_ep_expert_weights(
        experts.w13_weight.data.view(u8),
        experts.w13_weight_scale.data.view(u8),
        experts.w2_weight.data.view(u8),
        experts.w2_weight_scale.data.view(u8),
        world=experts.ep_size,
        top_k=experts.top_k,
        hidden=experts.hidden_size,
        inter=experts.intermediate_size,
        shuffled=True,
    )


def fused_ep_moe(ep, weights, x, x_quant, topk_weights, topk_ids, *, rows: int):
    if x_quant is not None:
        return ep(
            x_quant.data, x_quant.scale, weights, topk_weights, topk_ids, rows_hint=rows
        )
    return ep(x, None, weights, topk_weights, topk_ids, rows_hint=rows)


# ---------------------------------------------------------------- attention


def rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps):
    norm = _norm()
    if not _covered("rmsnorm_dual_mxfp8", norm.rmsnorm_dual_mxfp8_blocker(q, kv)):
        return None
    return norm.rmsnorm_dual_mxfp8(q, kv, q_weight, kv_weight, eps)


def rmsnorm_dual_warmup(q_dim: int, kv_dim: int, device: torch.device) -> None:
    _norm().warmup_rmsnorm_dual_mxfp8(q_dim, kv_dim, device)


def bf16_dual_gemm_weight(wa: torch.Tensor | None, wb: torch.Tensor):
    """kv_score (fp32 columns, optional) and weights_proj (bf16 columns) as
    one weight; None where the entry has no such shape."""
    gemm = _gemm()
    if (wb.shape[1], wb.shape[0]) not in gemm.bf16_gemm_f32_bf16_shapes():
        return None
    if wb.dtype != BF16 or (wa is not None and wa.dtype != BF16):
        return None
    return gemm.bf16_gemm_f32_bf16_shuffle_weight(wa, wb)


def bf16_dual_gemm_ok(x: torch.Tensor, w: torch.Tensor, na: int) -> bool:
    return _covered("bf16_gemm_f32_bf16", _gemm().bf16_gemm_f32_bf16_blocker(x, w, na))


def bf16_dual_gemm(x, w, out_a, out_b) -> None:
    _gemm().bf16_gemm_f32_bf16(x, w, out_a, out_b)


def compressor_wk(wk: torch.Tensor) -> torch.Tensor:
    return _attention().compressor_shuffle_wk(wk)


def compressor_insert(
    raw,
    positions,
    ring,
    ring_slots,
    query_start_loc,
    token_to_req,
    norm_weight,
    eps,
    main_cache,
    main_slots,
    cos_sin_main,
    kv_scale,
    wk,
    k_norm_weight,
    k_eps,
    idx_cache,
    idx_slots,
    cos_sin_idx,
    ratio,
) -> bool:
    attn = _attention()
    reason = attn.compressor_insert_blocker(
        raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx
    )
    if not _covered("compressor_insert", reason) or any(
        t.dtype != torch.int64 or not t.is_contiguous()
        for t in (positions, ring_slots, main_slots, idx_slots)
    ):
        return False
    attn.compressor_insert(
        raw,
        positions,
        ring,
        ring_slots,
        query_start_loc,
        token_to_req,
        norm_weight,
        eps,
        main_cache,
        main_slots,
        cos_sin_main,
        kv_scale,
        wk,
        k_norm_weight,
        k_eps,
        idx_cache,
        idx_slots,
        cos_sin_idx,
        ratio,
    )
    return True


def rope_fp8_kv_insert(
    q, kv, cache, slot_mapping, positions, cos_sin, kv_scale, q_scale_inv
) -> torch.Tensor | None:
    attn = _attention()
    reason = attn.rope_fp8_kv_insert_blocker(q, cache, cos_sin)
    if not _covered("rope_fp8_kv_insert", reason) or slot_mapping.dtype != torch.int64:
        return None
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    attn.rope_fp8_kv_insert(
        q, kv, q_fp8, cache, slot_mapping, positions, cos_sin, kv_scale, q_scale_inv
    )
    return q_fp8


def sparse_mla_decode(
    q,
    main_cache,
    main_indptr,
    main_indices,
    extra_cache,
    extra_indptr,
    extra_indices,
    attn_sink,
    softmax_scale,
    q_scale,
    kv_scale,
    positions,
    cos_sin,
    out_codes,
    out_scales,
) -> bool:
    attn = _attention()
    lists = (main_indptr, main_indices, extra_indptr, extra_indices)
    if (
        not _covered("sparse_mla_decode", attn.sparse_mla_decode_blocker(q, main_cache))
        or cos_sin.dtype != torch.float32
        or any(
            t is not None and (t.dtype != torch.int32 or not t.is_contiguous())
            for t in lists
        )
    ):
        return False
    attn.sparse_mla_decode(
        q,
        main_cache,
        main_indptr,
        main_indices,
        extra_cache,
        extra_indptr,
        extra_indices,
        attn_sink,
        softmax_scale,
        q_scale,
        kv_scale,
        positions,
        cos_sin,
        out_codes,
        out_scales,
    )
    return True


class GroupedWeight:
    def __init__(self, weight, scale, groups, n, k) -> None:
        self.weight, self.scale = weight, scale
        self.groups, self.n, self.k = groups, n, k


def mxfp8_grouped_weight(wo_a, groups: int, n: int) -> GroupedWeight | None:
    """wo_a's AITER-preshuffled weight `[G * N, K]` and its 32 x 32 block scales
    as `[G, N / 32, K / 32]` (views, no copy)."""
    w = getattr(wo_a, "weight_preshuffled", None)
    scale = getattr(wo_a, "weight_scale", None)
    if w is None or scale is None or w.dtype != torch.float8_e4m3fn:
        return None
    k = w.shape[1]
    if (groups, n, k) not in _gemm().mxfp8_grouped_gemm_shapes():
        return None
    if tuple(scale.shape) != (groups * n // 32, k // 32):
        return None
    _gemm().warmup_mxfp8_grouped_gemm(w.device, groups, n, k)
    scale = scale.view(torch.uint8).view(groups, n // 32, k // 32)
    return GroupedWeight(w, scale, groups, n, k)


def mxfp8_grouped_ok(m: int, w: GroupedWeight) -> bool:
    reason = _gemm().mxfp8_grouped_gemm_blocker(m, w.groups, w.n, w.k)
    return _covered("mxfp8_grouped_gemm", reason)


def mxfp8_grouped_gemm(x: QuantizedActivation, w: GroupedWeight):
    """(codes, scales) of the grouped GEMM with the MXFP8 epilogue."""
    return _gemm().mxfp8_grouped_gemm_mxfp8(x.data, x.scale, w.weight, w.scale)


# ---------------------------------------------------------------- indexer


class IndexerBuffers:
    """Static outputs shared by every indexer layer of one device (the layers
    run one after another on one stream)."""

    def __init__(self, device: torch.device, max_rows: int, topk: int) -> None:
        i32 = torch.int32
        self.slots = torch.zeros(max_rows * topk, dtype=i32, device=device)
        self.indptr = torch.zeros(max_rows + 1, dtype=i32, device=device)
        self.lens = torch.zeros(max_rows, dtype=i32, device=device)
        self.topk = topk
        self.max_rows = max_rows
        self.scores: dict[int, torch.Tensor] = {}

    def scores_for(self, max_ctx: int) -> torch.Tensor | None:
        cols = _indexer().mxfp4_indexer_scores_cols(max_ctx)
        if cols is None:
            return None
        buf = self.scores.get(cols)
        if buf is None:
            buf = torch.empty(
                self.max_rows, cols, dtype=torch.float32, device=self.slots.device
            )
            self.scores[cols] = buf
        return buf


def indexer_ok(heads: int, head_dim: int, page: int, max_ctx: int, next_n: int):
    return _indexer().mxfp4_indexer_blocker(heads, head_dim, page, max_ctx, next_n)


def indexer_ragged(buffers: IndexerBuffers, rows, is_valid, block_table, page):
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        return None
    return _indexer().mxfp4_indexer_ragged(
        slots=buffers.slots[: rows * buffers.topk],
        indptr=buffers.indptr[: rows + 1],
        lens=buffers.lens[:rows],
        is_valid=is_valid,
        slot_block_table=block_table,
        slot_block_size=page,
    )


def indexer_topk(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    out,
    *,
    next_n,
    max_ctx,
    ragged,
    candidates,
    consumer,
    buffers,
) -> bool:
    idx = _indexer()
    # q holds two FP4 values per byte
    heads, head_dim, page = q.shape[1], 2 * q.shape[2], kv_cache.shape[1]
    reason = idx.mxfp4_indexer_blocker(heads, head_dim, page, max_ctx, next_n)
    if not _covered("mxfp4_indexer", reason):
        return False
    if consumer:
        idx.mxfp4_indexer_candidate_topk(
            q,
            q_scale,
            weights,
            kv_cache,
            block_table,
            row_ends,
            candidates,
            out,
            next_n=next_n,
            max_ctx=max_ctx,
            ragged=ragged,
        )
    else:
        idx.mxfp4_indexer_topk(
            q,
            q_scale,
            weights,
            kv_cache,
            block_table,
            row_ends,
            out,
            next_n=next_n,
            max_ctx=max_ctx,
            candidates_out=candidates,
            ragged=ragged,
            scores=buffers.scores_for(max_ctx),
        )
    return True


# ---------------------------------------------------------------- mHC


def mhc_seam_gate(fn: torch.Tensor, hc: int, hidden: int) -> torch.Tensor:
    return _norm().mhc_pack_gate_weight(fn.contiguous(), hc, hidden)


def mhc_seam_ok(residual: torch.Tensor) -> bool:
    reason = _norm().mhc_post_pre_rmsnorm_blocker(residual)
    return _covered("mhc_post_pre_rmsnorm", reason)


def mhc_seam(residual, fn_packed, hc_scale, hc_base, norm_weight, **kwargs):
    return _norm().mhc_post_pre_rmsnorm(
        residual, fn_packed, hc_scale, hc_base, norm_weight, **kwargs
    )


def mhc_tail_ok(residual: torch.Tensor) -> bool:
    reason = _norm().mhc_post_collapse_rmsnorm_blocker(residual)
    return _covered("mhc_post_collapse_rmsnorm", reason)


def mhc_tail(
    residual,
    pre_mix,
    norm,
    *,
    sublayer_out=None,
    post_mix=None,
    comb_mix=None,
    keep_residual=False,
    aux_mean=False,
):
    kwargs: dict[str, Any] = {}
    if sublayer_out is not None:
        kwargs.update(
            sublayer_out=sublayer_out.contiguous(),
            post_mix=post_mix.contiguous(),
            comb_mix=comb_mix.contiguous(),
            keep_residual=keep_residual,
        )
    return _norm().mhc_post_collapse_rmsnorm(
        residual.contiguous(),
        pre_mix.contiguous(),
        norm.weight,
        eps=norm.variance_epsilon,
        aux_mean=aux_mean,
        **kwargs,
    )


def postmix_weight(linear, routed: bool) -> Mxfp8Weight | None:
    w = mxfp8_weight(linear)
    if w is None or (w.n, w.k, routed) not in _gemm().mxfp8_postmix_shapes():
        return None
    _gemm().warmup_mxfp8_postmix_gemm(w.weight, w.scale, routed)
    return w


def postmix_gate(fn: torch.Tensor, w: Mxfp8Weight, routed: bool) -> torch.Tensor:
    return _gemm().mxfp8_postmix_pack_gate_weight(fn.contiguous(), w.n, w.k, routed)


def postmix_ok(m: int, w: Mxfp8Weight, routed: bool) -> bool:
    reason = _gemm().mxfp8_postmix_blocker(m, w.n, w.k, routed)
    return _covered("mxfp8_postmix_gemm", reason)


def postmix_gates_gemm(
    x: QuantizedActivation,
    w: Mxfp8Weight,
    residual,
    post_mix,
    comb_mix,
    pre_mix,
    fn_packed,
    *,
    routed=None,
    aux_mean=False,
):
    """(new residual, partial layer input, gate partials, stream mean or None)."""
    return _gemm().mxfp8_postmix_gates_gemm(
        x.data,
        x.scale,
        w.weight,
        w.scale,
        residual,
        post_mix.contiguous(),
        comb_mix.contiguous(),
        pre_mix.contiguous(),
        fn_packed,
        None if routed is None else routed.contiguous(),
        aux_mean=aux_mean,
    )


def mhc_gate_finish(partials, x1, hc_scale, hc_base, norm, layer, *, quant: bool):
    """(post, comb, layer input, next pre[, codes, scales]) of the seam the
    gate partials belong to; `layer` supplies the mHC constants."""
    return _norm().mhc_gate_finish(
        partials,
        x1,
        hc_scale,
        hc_base,
        norm.weight,
        rms_eps=layer.rms_norm_eps,
        hc_pre_eps=layer.hc_eps,
        hc_sinkhorn_eps=layer.hc_eps,
        hc_post_mult=layer.hc_post_alpha,
        norm_eps=norm.variance_epsilon,
        sinkhorn_repeat=layer.hc_sinkhorn_iters,
        quant=quant,
    )


def mhc_warmup(hc: int, hidden: int, device: torch.device, tail: bool) -> None:
    norm = _norm()
    norm.warmup_mhc_post_pre_rmsnorm(hc, hidden, device)
    if tail:
        norm.warmup_mhc_post_collapse_rmsnorm(hc, hidden, device)


# ---------------------------------------------------------------- head, draft


def bf16_linear_ok(w: torch.Tensor) -> bool:
    return _covered("bf16_gemm", _gemm().bf16_gemm_blocker(w, None))


def bf16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _gemm().bf16_gemm(x, w)


def bf16_linear_warmup(w: torch.Tensor) -> None:
    _gemm().warmup_bf16_gemm(w.device, w.shape[0], w.shape[1], False)


def markov_chain_ok(w1: torch.Tensor, idx_map: torch.Tensor) -> bool:
    return _covered("markov_chain", _sampling().markov_chain_blocker(w1, idx_map))


def markov_chain(*args) -> None:
    _sampling().markov_chain(*args)


def markov_chain_warmup(w1: torch.Tensor, steps: int) -> None:
    vocab, rank = w1.shape
    _sampling().warmup_markov_chain(
        w1.device, vocab, rank, steps, idx_dtype=torch.int64, ids_dtype=torch.int32
    )
