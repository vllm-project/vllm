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
    OnlineWeightLoaderMixin,
    _copy_missing_attrs,
)
from vllm.model_executor.utils import set_weight_attrs


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


class ExpertsInt8MoEMethod(FusedMoEMethodBase, OnlineWeightLoaderMixin):
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
        # Use shared mixin logic for weight creation
        self._mixin_create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            extra_weight_attrs,
        )

    def _create_scale_tensors(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Create Int8 per-channel scale tensors."""
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

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Handle dummy weights (--load_format dummy)
        if layer.w13_weight.device == torch.device("meta"):
            w13_weight = torch.nn.Parameter(
                torch.empty_like(layer.w13_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w13_weight, {"weight_loader": layer.w13_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w13_weight, w13_weight)
            layer.register_parameter("w13_weight", w13_weight)
            torch.nn.init.normal_(layer.w13_weight)

        if layer.w2_weight.device == torch.device("meta"):
            w2_weight = torch.nn.Parameter(
                torch.empty_like(layer.w2_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w2_weight, {"weight_loader": layer.w2_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w2_weight, w2_weight)
            layer.register_parameter("w2_weight", w2_weight)
            torch.nn.init.normal_(layer.w2_weight)

        # Quantize fp16/bf16 weights to int8 with per-channel scales
        w13, w13_scale = self._quantize_to_int8(layer.w13_weight)
        w2, w2_scale = self._quantize_to_int8(layer.w2_weight)

        # Replace parameters with quantized versions
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        # Setup modular kernel
        self._setup_kernel(layer)

    def _quantize_to_int8(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a weight tensor to int8 with per-channel scales.

        Args:
            weight: Tensor of shape [num_experts, out_features, in_features]

        Returns:
            Tuple of (quantized_weight, scales) where scales has shape
            [num_experts, out_features]
        """
        num_experts = weight.shape[0]
        int8_weight = torch.empty_like(weight, dtype=torch.int8)
        scales = torch.empty(
            (num_experts, weight.shape[1]), dtype=torch.float32, device=weight.device
        )

        vmax = torch.iinfo(torch.int8).max

        for expert in range(num_experts):
            expert_weight = weight[expert]  # [out_features, in_features]
            # Per-channel (per-row) quantization
            channel_max = torch.max(torch.abs(expert_weight), dim=1)[0]
            expert_scales = channel_max / vmax
            # Avoid division by zero
            expert_scales = torch.where(
                expert_scales == 0, torch.ones_like(expert_scales), expert_scales
            )
            scales[expert] = expert_scales

            # Quantize
            int8_weight[expert] = torch.round(
                expert_weight / expert_scales.unsqueeze(1)
            ).clamp(-vmax, vmax).to(torch.int8)

        return int8_weight, scales

    def _setup_kernel(self, layer: Module) -> None:
        """Setup the modular kernel after quantization."""
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

    def get_fused_moe_quant_config(
        self, layer: Module
    ) -> FusedMoEQuantConfig | None:
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
