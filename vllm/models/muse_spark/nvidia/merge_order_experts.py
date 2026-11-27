# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 Muse expert LoRA with PEFT-style rounding inside each weight tile."""

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_output import route_major_moe_output
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.triton_utils import tl, triton


@triton.jit
def _muse_weight_tile_kernel(
    X,
    W,
    A,
    B,
    Y,
    SORTED,
    GROUPS,
    PADDED,
    EXPERT_MAP,
    ROUTES,
    GLOBAL_E: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    INPUT_GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BM >= tl.load(PADDED):
        return
    group = tl.load(GROUPS + block)
    expert = group % GLOBAL_E
    slot = group // GLOBAL_E - 1
    if EXPERT_MAP is not None:
        # Routes are global, but both weights and LoRA factors are rank-local.
        expert = tl.load(EXPERT_MAP + expert)
        if expert < 0:
            return
    rows = tl.load(SORTED + block * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    # Expert offsets can exceed 32 bits for large checkpoint tensors.
    weight_offset = expert.to(tl.int64) * N * K
    factor_offset = slot.to(tl.int64) * E + expert
    acc = tl.full((BM, BN), 0, tl.float32)
    for kk in range(tl.cdiv(K, BK)):
        ks = kk * BK + tl.arange(0, BK)
        wmask = (cols[:, None] < N) & (ks[None, :] < K)
        w = tl.load(W + weight_offset + cols[:, None] * K + ks[None, :], wmask, 0)
        if slot >= 0:
            delta = tl.full((BN, BK), 0, tl.float32)
            for rank in tl.static_range(R):
                a = tl.load(A + (factor_offset * R + rank) * K + ks, ks < K, 0)
                b = tl.load(B + (factor_offset * N + cols) * R + rank, cols < N, 0)
                delta = tl.fma(b[:, None], a[None, :], delta)
            w = (w.to(tl.float32) + delta.to(tl.bfloat16).to(tl.float32)).to(
                tl.bfloat16
            )
        x = tl.load(
            X + (rows // INPUT_GROUP)[:, None] * K + ks[None, :],
            (rows[:, None] < ROUTES) & (ks[None, :] < K),
            0,
        )
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < ROUTES) & (cols[None, :] < N),
    )


def _muse_tile_projection(
    x, w, factors_a, factors_b, out, schedule, routes, input_group, expert_map=None
):
    (a,) = factors_a
    (b,) = factors_b
    e, n, k = w.shape
    rank = a.shape[2]
    assert a.shape == (a.size(0), e, rank, k) and b.shape == (a.size(0), e, n, rank)
    assert all(t.is_contiguous() for t in (x, w, a, b, out))
    sorted_ids, groups, padded = schedule
    _muse_weight_tile_kernel[(triton.cdiv(sorted_ids.numel(), 16), triton.cdiv(n, 32))](
        x,
        w,
        a,
        b,
        out,
        sorted_ids,
        groups,
        padded,
        expert_map,
        routes,
        e if expert_map is None else expert_map.numel(),
        e,
        n,
        k,
        rank,
        input_group,
        16,
        32,
        32,
    )


class MuseSparkMergeOrderExperts(TritonExperts):
    """Use FP32 adapter factors and BF16-rounded expert weights on Hopper."""

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        ctx = self._lora_context
        if ctx is None:
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        assert self._uses_post_gemm2_hook(hidden_states.size(0))
        assert ctx.tp_size in (1, 2) and not ctx.fully_sharded
        assert self.quant_dtype is None and not apply_router_weight_on_input
        assert self.w1_bias is None and self.w2_bias is None
        assert hidden_states.dtype == w1.dtype == w2.dtype == torch.bfloat16
        m, top_k = topk_ids.shape
        slots = ctx.punica_wrapper.token_lora_indices[:m]
        assert slots.shape == (m,)
        num_experts = global_num_experts if expert_map is not None else w1.size(0)
        groups = topk_ids + (slots + 1).to(topk_ids.dtype)[:, None] * num_experts
        schedule = moe_align_block_size(groups, 16, (ctx.max_loras + 1) * num_experts)
        gu = _resize_cache(workspace2, (m, top_k, w1.size(1)))
        _muse_tile_projection(
            hidden_states,
            w1,
            ctx.w13_lora_a_stacked,
            ctx.w13_lora_b_stacked,
            gu,
            schedule,
            groups.numel(),
            top_k,
            expert_map,
        )
        act = _resize_cache(workspace13, (m * top_k, w2.size(2)))
        self.activation(activation, act, gu.view(m * top_k, -1))
        down = _resize_cache(workspace2, (m, top_k, w2.size(1)))
        _muse_tile_projection(
            act,
            w2,
            ctx.w2_lora_a_stacked,
            ctx.w2_lora_b_stacked,
            down,
            schedule,
            groups.numel(),
            1,
            expert_map,
        )
        self._run_post_gemm2_hook(
            route_major_moe_output(down, topk_weights, topk_ids, expert_map), output
        )
