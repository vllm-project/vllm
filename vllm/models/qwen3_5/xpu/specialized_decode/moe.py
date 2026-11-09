# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind and execute routed + gated shared experts for eligible decode rows."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from . import MAX_DECODE_ROWS, MAX_ROUTED_ROWS
from .weights import get_weights, prepare_weights

if TYPE_CHECKING:
    from vllm_xpu_kernels.moe_shared_fused_interface import XpuFusedMoeWithSharedExpert

    from ..model import Qwen3_5SparseMoeBlock


logger = init_logger(__name__)


class FusedMoeDecode:
    """A native MoE executor bound to one block's loaded parameters."""

    def __init__(self, block: "Qwen3_5SparseMoeBlock"):
        self.block = block
        self.plan: XpuFusedMoeWithSharedExpert | None = None
        self._weights: tuple[torch.Tensor, ...] | None = None
        self.dtype: torch.dtype | None = None
        self.hidden_dim = self.top_k = 0
        weights = get_weights(block)
        if isinstance(weights, str):
            logger.info_once("Qwen3.5 XPU fused MoE disabled: %s", weights)
            return
        self._weights = weights
        config = cast(MoERunner, block.experts).moe_config
        self.dtype = config.in_dtype
        self.hidden_dim = config.hidden_dim
        self.top_k = config.experts_per_token
        try:
            from vllm_xpu_kernels.moe_shared_fused_interface import (
                XpuFusedMoeWithSharedExpert as FusedMoe,
            )
            from vllm_xpu_kernels.moe_shared_fused_interface import (
                supports,
            )
        except ImportError:
            logger.info_once("Qwen3.5 XPU fused MoE disabled: provider unavailable")
            self._weights = None
            return
        if not supports(
            self.dtype,
            self._weights[0].dtype,
            self.top_k,
            self.hidden_dim,
            self._weights[3].shape[0],
            device=self._weights[0].device,
        ):
            logger.info_once(
                "Qwen3.5 XPU fused MoE disabled: native provider does not support "
                "the weight device, dtype or dimensions"
            )
            return
        weights = prepare_weights(self.block)
        if isinstance(weights, str):
            logger.info_once("Qwen3.5 XPU fused MoE disabled: %s", weights)
            return
        self._weights = weights
        w13, w2, shared_w13, shared_w2, s13, s2, shared_s13, shared_s2, gate = (
            self._weights
        )
        self.plan = FusedMoe(
            w13=w13,
            w13_scales=s13,
            w2=w2,
            w2_scales=s2,
            shared_w13=shared_w13,
            shared_w13_scale=shared_s13,
            shared_w2=shared_w2,
            shared_w2_scale=shared_s2,
            shared_gate=gate,
            top_k=self.top_k,
        )

    def get_plan(self, x: torch.Tensor) -> "XpuFusedMoeWithSharedExpert | None":
        if (
            self.plan is None
            or x.device != self.plan.device
            or x.ndim != 2
            or not 1 <= x.shape[0] <= MAX_DECODE_ROWS
            or x.shape[0] * self.top_k > MAX_ROUTED_ROWS
            or x.shape[1] != self.hidden_dim
            or x.dtype != self.dtype
            or not x.is_contiguous()
            or x.data_ptr() % 64
        ):
            return None
        return self.plan

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        plan = self.get_plan(hidden_states)
        if plan is None:
            return None
        block = self.block
        runner = cast(MoERunner, block.experts)
        router_logits, _ = block.gate(hidden_states)
        topk_weights, topk_ids = runner.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=runner.routed_experts.quant_method.topk_indices_dtype,
            input_ids=None,
        )
        output = plan.forward(
            torch.empty_like(hidden_states),
            hidden_states,
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.contiguous(),
        )
        if block.tp_size > 1 and not runner.moe_config.skip_final_all_reduce:
            output = tensor_model_parallel_all_reduce(output)
        return output
