# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared orchestration for the native RDNA3 (gfx1100) fused-MoE methods.

The W4A16 (INT4) and MXFP4 RDNA3 paths run the same two-GEMM forward
(gate_up -> SwiGLU -> down, with the down-proj output_topk reduction fusing
moe_sum); only the per-GEMM HIP op differs. That common flow lives here; each
method supplies ``_gemm_w13`` / ``_gemm_w2``.
"""

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def repack_experts(packed: torch.Tensor, scale: torch.Tensor, deinterleave: bool):
    """[E, N, K/2] uint8 + [E, N, K/32] uint8 -> [E, K/8, N] int32 + [E, K/32, N].

    Per-expert into pre-allocated outputs (peak ~2x ONE expert tensor, not the
    whole stack) so a multi-GB MoE repack fits alongside the loaded weights.
    Four consecutive packed bytes little-endian = a uint32 of 8 E2M1 codes
    along K — exactly the [K/8, N] word layout the kernel reads. ``deinterleave``
    splits gate/up rows of w13 ([g0,u0,...] -> [g..., u...]) for GPT-OSS.
    """
    E, N, kh = packed.shape
    dev = packed.device
    b_q = torch.empty(E, kh // 4, N, dtype=torch.int32, device=dev)
    b_scale = torch.empty(E, scale.shape[2], N, dtype=torch.uint8, device=dev)
    for e in range(E):
        pe = packed[e]
        se = scale[e]
        if deinterleave:
            pe = torch.cat([pe[::2], pe[1::2]], dim=0)
            se = torch.cat([se[::2], se[1::2]], dim=0)
        b_q[e] = pe.contiguous().view(torch.int32).t()
        b_scale[e] = se.t()
    return b_q, b_scale


class RDNA3FusedMoEMixin:
    """Mixin: shared apply() + two-GEMM orchestration. Subclasses implement
    ``_gemm_w13`` and ``_gemm_w2`` (the op-specific HIP launch)."""

    def _gemm_w13(self, layer, a, c, tw, sti, eid, ntp, top_k, block_size_m, mul_tw):
        raise NotImplementedError

    def _gemm_w2(self, layer, a, c, tw, sti, eid, ntp, block_size_m, mul_tw, output_topk):
        raise NotImplementedError

    def _rdna3_run(
        self,
        layer,
        hidden_states,
        topk_weights,
        topk_ids,
        activation,
        apply_router_weight_on_input,
        global_num_experts,
        expert_map,
    ):
        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        total_tokens = num_tokens * top_k
        N_gate_up = layer.w13_weight_packed.shape[2]
        hidden_size = layer.w2_weight_packed.shape[2]
        dtype = hidden_states.dtype
        device = hidden_states.device
        intermediate = N_gate_up // 2 if activation.is_gated else N_gate_up

        if global_num_experts <= 0:
            global_num_experts = layer.w13_weight_packed.shape[0]
        block_size_m = 1 if num_tokens <= 4 else 4

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts, expert_map
        )

        # Use pre-allocated decode buffers when present and big enough.
        w1_buf = getattr(layer, "rdna3_w1_buf", None)
        if w1_buf is not None and total_tokens <= w1_buf.shape[0]:
            w1_out = w1_buf[:total_tokens]
            w1_out.zero_()
            act_out = layer.rdna3_act_buf[:total_tokens]
        else:
            w1_out = torch.zeros(total_tokens, N_gate_up, dtype=dtype, device=device)
            act_out = torch.empty(
                total_tokens, intermediate, dtype=dtype, device=device
            )

        topk_w_float = topk_weights.view(-1).float()
        empty_tw = getattr(layer, "rdna3_empty_tw", None)
        if empty_tw is None:
            empty_tw = torch.empty(0, device=device)

        self._gemm_w13(
            layer,
            hidden_states,
            w1_out,
            topk_w_float if apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
        )
        apply_moe_activation(activation, act_out, w1_out)

        out = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        self._gemm_w2(
            layer,
            act_out,
            out,
            topk_w_float if not apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            block_size_m,
            not apply_router_weight_on_input,
            top_k,
        )
        return out

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        return self._rdna3_run(
            layer,
            x,
            topk_weights,
            topk_ids,
            activation,
            layer.apply_router_weight_on_input,
            layer.global_num_experts,
            layer.expert_map,
        )
