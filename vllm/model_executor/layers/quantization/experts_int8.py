# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch
from torch.nn import Module

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.moe_weight_loader import (
    MoeOnlineWeightLoader,
)


class ExpertsInt8Config(QuantizationConfig):
    """Config class for Int8 experts quantization."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return ExpertsInt8MoEMethod(self, layer.moe_config)
        return None


class ExpertsInt8MoEMethod(FusedMoEMethodBase):
    """
    MoE method for online Int8 quantization.

    Quantizes fp16/bf16 weights to int8 after loading with per-channel scales.
    Uses the Triton fused MoE kernel with int8_w8a16 mode via the modular
    kernel infrastructure.
    """

    def __init__(
        self,
        quant_config: ExpertsInt8Config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self.kernel: mk.FusedMoEModularKernel | None = None
        self.weight_loader = MoeOnlineWeightLoader(self)

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.kernel is not None:
            return self.kernel.prepare_finalize.topk_indices_dtype()
        return None

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.weight_loader.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        self.weight_loader.process_weights_after_loading(layer)

    def create_scale_tensors(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        # WEIGHT_SCALES (per-channel for int8)
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        return w13_weight_scale, w2_weight_scale

    def get_quantized_dtype(self) -> torch.dtype:
        return torch.int8

    def quantize_expert(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single expert's weight to int8 with per-channel scales."""
        # weight shape: [out_features, in_features]
        vmax = torch.iinfo(torch.int8).max
        channel_max = torch.max(torch.abs(weight), dim=1)[0]
        scales = channel_max / vmax
        # Avoid division by zero
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        # Quantize
        int8_weight = (
            torch.round(weight / scales.unsqueeze(1)).clamp(-vmax, vmax).to(torch.int8)
        )
        return int8_weight, scales

    def setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        # Update layer weights with quantized versions
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Only setup modular kernel for non-all2all or naive all2all cases
        if self.moe_quant_config and (
            (not self.moe.moe_parallel_config.use_all2all_kernels)
            or self.moe.moe_parallel_config.use_naive_all2all_kernels
        ):
            prepare_finalize = MoEPrepareAndFinalizeNoEP(
                defer_input_quant=False,
            )

            experts = TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

            self.kernel = mk.FusedMoEModularKernel(
                prepare_finalize,
                experts,
                shared_experts=None,
                moe_parallel_config=self.moe.moe_parallel_config,
            )

    def get_fused_moe_quant_config(self, layer: Module) -> FusedMoEQuantConfig | None:
        return int8_w8a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel is not None, (
            "Kernel not initialized. Ensure weights are loaded before calling apply()."
        )
        return self.kernel(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=True,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
