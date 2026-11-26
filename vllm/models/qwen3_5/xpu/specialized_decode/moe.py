# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind and execute routed + gated shared experts for eligible decode rows."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from . import MAX_DECODE_ROWS, MAX_ROUTED_ROWS
from .planning import DecodeRuntime
from .weights import get_weights, prepare_weights

if TYPE_CHECKING:
    from vllm_xpu_kernels.moe_shared_fused_interface import XpuFusedMoeWithSharedExpert

    from ..model import Qwen3_5SparseMoeBlock


class FusedMoeDecode:
    """A native MoE executor bound to one block's loaded parameters."""

    def __init__(self, block: "Qwen3_5SparseMoeBlock", runtime: DecodeRuntime):
        self.block = block
        self.runtime = runtime
        self.reason: str | None = None
        self.native_launches = 0
        self._scratch: tuple[torch.Tensor, ...] = ()
        self._bindings: tuple = ()
        self.plan: XpuFusedMoeWithSharedExpert | None = None
        self._weights: tuple[torch.Tensor, ...] | None = None
        self.dtype: torch.dtype | None = None
        self.hidden_dim = self.top_k = 0
        weights = get_weights(block)
        if isinstance(weights, str):
            self.reason = weights
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
            self.reason = "optional shared-expert provider unavailable"
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
            self.reason = "native provider does not support device, dtype or dimensions"
            return
        weights = prepare_weights(self.block)
        if isinstance(weights, str):
            self.reason = weights
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

        self._scratch = tuple(
            self.plan.workspace(rows) for rows in range(1, MAX_DECODE_ROWS + 1)
        )
        self._bindings = tuple(self._binding(t) for t in self._weights)

    @staticmethod
    def _binding(tensor: torch.Tensor) -> tuple:
        return (
            id(tensor),
            tensor.data_ptr(),
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
            tensor.device,
        )

    def bindings_match(self) -> bool:
        """Verify current canonical parameters still match captured addresses."""
        weights = get_weights(self.block)
        if isinstance(weights, str) or self.plan is None:
            return False
        provider_weights = (
            self.plan.w13,
            self.plan.w2,
            self.plan.shared_w13,
            self.plan.shared_w2,
            self.plan.w13_scales,
            self.plan.w2_scales,
            self.plan.shared_w13_scale,
            self.plan.shared_w2_scale,
            self.plan.shared_gate,
        )
        return tuple(self._binding(t) for t in weights) == self._bindings and all(
            a is b for a, b in zip(weights, provider_weights)
        )

    @property
    def scratch_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._scratch)

    def get_plan(self, x: torch.Tensor) -> "XpuFusedMoeWithSharedExpert | None":
        step = self.runtime.step
        if (
            step is None
            or step.reason is not None
            or self.plan is None
            or x.device != self.plan.device
            or x.ndim != 2
            or step.padded_rows != x.shape[0]
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
        self.native_launches += 1
        output = plan.forward(
            torch.empty_like(hidden_states),
            hidden_states,
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.contiguous(),
        )
        if block.tp_size > 1 and not runner.moe_config.skip_final_all_reduce:
            output = tensor_model_parallel_all_reduce(output)
        return output
