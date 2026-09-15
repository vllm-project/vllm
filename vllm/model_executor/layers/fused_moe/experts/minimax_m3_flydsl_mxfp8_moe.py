# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP8 MoE on the FlyDSL kernels of
``vllm/models/minimax_m3/amd/ops/moe_mxfp8`` (gfx950). Same weights and quant
config as ``AiterMxfp8Experts``, whose ``apply`` is the fallback for what the
kernels do not cover; the MXFP8 MoE oracle lists this class ahead of it for the
``aiter`` backend. ``VLLM_ROCM_USE_M3_FLYDSL_MOE=0`` keeps aiter's kernels.
"""

from __future__ import annotations

import math

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
    AiterMxfp8Experts,
)

logger = init_logger(__name__)


def _kernels():
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8

    return moe_mxfp8


class MiniMaxM3FlyDSLMxfp8Experts(AiterMxfp8Experts):
    """MXFP8 MoE through the MiniMax-M3 FlyDSL chains (gfx950)."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        logger.info_once(
            "MiniMax-M3 FlyDSL MXFP8 MoE kernels selected (hidden %d, intermediate "
            "%d per rank, up to %d tokens per call; VLLM_ROCM_USE_M3_FLYDSL_MOE=0 "
            "keeps aiter's kernels)",
            moe_config.hidden_dim,
            moe_config.intermediate_size_per_partition,
            _kernels().MAX_TOKENS,
        )

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not is_supported:
            return False, reason
        if not envs.VLLM_ROCM_USE_M3_FLYDSL_MOE:
            return False, "VLLM_ROCM_USE_M3_FLYDSL_MOE=0"
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False, "kernels are written for gfx950"
        if moe_config.moe_parallel_config.use_ep:
            return False, "kernels do not support expert parallelism"
        if moe_config.has_bias:
            return False, "kernels do not support expert bias"
        try:
            kernels = _kernels()
        except Exception as exc:  # flydsl / kernel import problem: keep aiter
            return False, f"FlyDSL kernels not importable: {exc!r}"
        hidden = moe_config.hidden_dim
        inter = moe_config.intermediate_size_per_partition
        if not kernels.supports_shapes(hidden, inter):
            return False, (
                f"hidden={hidden} intermediate_size_per_partition={inter} is not "
                "tiled by the kernels (MiniMax-M3 at TP4: 6144 / 768)"
            )
        if (
            moe_config.hidden_dim_unpadded != hidden
            or moe_config.intermediate_size_per_partition_unpadded != inter
        ):
            return False, "kernels do not support padded dimensions"
        limit = moe_config.swiglu_limit
        if limit is None or not math.isclose(float(limit), kernels.SWIGLU_LIMIT):
            return False, (
                f"kernels hardcode swiglu_limit={kernels.SWIGLU_LIMIT}; got {limit}"
            )
        return True, None

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta,
        apply_router_weight_on_input: bool,
    ):
        kernels = _kernels()
        if apply_router_weight_on_input or not kernels.supports_batch(hidden_states):
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
        moe_config = self.moe_config
        alpha, limit = moe_config.swiglu_alpha, moe_config.swiglu_limit
        assert alpha is not None and limit is not None  # is_supported_config
        n_tokens, hidden = hidden_states.shape
        out = (
            output
            if output.dtype == torch.bfloat16
            and output.is_contiguous()
            and tuple(output.shape) == (n_tokens, hidden)
            else None
        )
        result = kernels.mxfp8_moe(
            hidden_states,
            w1,
            self.w1_scale_val,
            w2,
            self.w2_scale_val,
            topk_weights,
            topk_ids,
            hidden_size=hidden,
            intermediate_size=moe_config.intermediate_size_per_partition,
            num_experts=w1.shape[0],
            swiglu_alpha=float(alpha),
            swiglu_limit=float(limit),
            # aiter's fused shared expert is appended after the routed experts
            # (RoutedExperts sizes the weights by global + fused experts)
            fused_shared_expert=w1.shape[0] > global_num_experts,
            out=out,
        )
        if result is not output:
            output.copy_(result)
