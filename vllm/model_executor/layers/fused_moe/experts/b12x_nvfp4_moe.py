# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


def _import_b12x_tp_moe():
    try:
        return importlib.import_module("b12x.integration.tp_moe")
    except ImportError:
        return None


_B12X_MOE_WORKSPACE_POOLS: dict[int, Any] = {}


def _get_b12x_workspace_pool(device: torch.device) -> Any:
    """Return a lazily created per-device workspace pool for the B12X MoE kernel."""
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    pool = _B12X_MOE_WORKSPACE_POOLS.get(device_idx)
    if pool is None:
        tp_moe = _import_b12x_tp_moe()
        if tp_moe is None:
            raise ImportError("b12x is not installed or importable")
        pool = tp_moe.allocate_tp_moe_workspace_pool()
        _B12X_MOE_WORKSPACE_POOLS[device_idx] = pool
    return pool


class B12xExperts(mk.FusedMoEExpertsModular):
    """NVFP4 fused MoE via the optional external b12x SM12x backend."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        tp_moe = _import_b12x_tp_moe()
        if tp_moe is None:
            raise ImportError("b12x is not installed or importable")
        self._b12x_moe_fp4 = tp_moe.b12x_moe_fp4

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_b12x_input_scales_fused", False):
            return
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)
        layer._b12x_input_scales_fused = True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @property
    def writes_final_output_directly(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and _import_b12x_tp_moe() is not None
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUOAI]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        del moe_parallel_config
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, topk, global_num_experts, local_num_experts, expert_tokens_meta, activation
        return ((0,), (0,), (M, K))

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
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del global_num_experts, a2_scale, workspace13, workspace2, expert_tokens_meta
        assert expert_map is None, "B12xExperts does not support expert_map"
        assert activation.is_gated, "B12xExperts only supports gated activations"
        assert not apply_router_weight_on_input, (
            "B12xExperts requires router weights to be applied inside the kernel"
        )
        assert a1q_scale is None, "B12xExperts expects unquantized inputs"
        assert self.a1_gscale is not None
        assert self.a2_gscale is not None
        assert self.g1_alphas is not None
        assert self.g2_alphas is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        self._b12x_moe_fp4(
            hidden_states,
            self.a1_gscale,
            w1,
            self.w1_scale,
            self.g1_alphas,
            self.a2_gscale,
            w2,
            self.w2_scale,
            self.g2_alphas,
            topk_weights,
            topk_ids,
            workspace=_get_b12x_workspace_pool(hidden_states.device),
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
        )
