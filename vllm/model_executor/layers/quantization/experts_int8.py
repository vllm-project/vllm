# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
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


class ExpertsInt8MoEMethod(FusedMoEMethodBase):
    """
    MoE method for online Int8 quantization.

    Quantizes fp16/bf16 weights to int8 during weight loading with
    per-channel scales. Uses the Triton fused MoE kernel with int8_w8a16 mode
    via the modular kernel infrastructure.
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
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        int8_dtype = torch.int8

        # Store layer dimensions for quantizing_weight_loader
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]

        # Wrap weight loader to do eager quantization and track loading progress
        wrapped_weight_loader = self._create_quantizing_weight_loader(
            layer, weight_loader
        )
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel) - int8 weights
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=int8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel) - int8 weights
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=int8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-channel scales for int8 quantization
        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

    def _create_quantizing_weight_loader(self, layer, weight_loader):
        """
        Creates a weight loader that performs eager int8 quantization
        during loading and sets up the modular kernel when complete.
        """
        # Track loading progress: 3 shards per expert (w1, w2, w3)
        total_shards = layer.num_experts * 3

        def quantize_and_call_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: int,
            expert_id: int,
        ):
            # Initialize tracking on first load
            if not hasattr(layer, "_loaded_shards"):
                layer._loaded_shards = 0

            tp_rank = get_tensor_model_parallel_rank()
            shard_size = layer.intermediate_size_per_partition
            shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
            device = get_tp_group().device
            loaded_weight = loaded_weight.to(device)

            # Eager quantization: quantize each shard immediately during load
            if shard_id == "w1":  # gate_proj -> first half of w13
                scales = quantize_in_place_and_get_scales(loaded_weight[shard, :])
                layer.w13_scale.data[expert_id, 0:shard_size].copy_(scales[:, 0])
            elif shard_id == "w3":  # up_proj -> second half of w13
                scales = quantize_in_place_and_get_scales(loaded_weight[shard, :])
                layer.w13_scale.data[expert_id, shard_size : 2 * shard_size].copy_(
                    scales[:, 0]
                )
            elif shard_id == "w2":  # down_proj
                scales = quantize_in_place_and_get_scales(loaded_weight[:, shard])
                layer.w2_scale.data[expert_id, :].copy_(scales[:, 0])
            else:
                raise ValueError(f"Shard id must be in [w1, w2, w3] but got {shard_id}")

            # Call original weight loader with quantized weight
            weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

            # Track progress and setup kernel when all weights are loaded
            layer._loaded_shards += 1
            if layer._loaded_shards == total_shards:
                self._setup_kernel(layer)
                del layer._loaded_shards
                layer._already_called_process_weights_after_loading = True

        return quantize_and_call_weight_loader

    def _setup_kernel(self, layer: torch.nn.Module) -> None:
        """Setup the modular kernel after all weights are loaded and quantized."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Only setup modular kernel for non-all2all or naive all2all cases
        # For complex all2all, FusedMoEModularMethod will handle kernel creation
        if self.moe_quant_config and (
            (not self.moe.moe_parallel_config.use_all2all_kernels)
            or self.moe.moe_parallel_config.use_naive_all2all_kernels
        ):
            # Create prepare/finalize (no input quantization for W8A16)
            prepare_finalize = MoEPrepareAndFinalizeNoEP(
                defer_input_quant=False,
            )

            # Create TritonExperts for int8_w8a16
            experts = TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

            # Create modular kernel
            self.kernel = mk.FusedMoEModularKernel(
                prepare_finalize,
                experts,
                shared_experts=None,
                moe_parallel_config=self.moe.moe_parallel_config,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Called after weight loading. For int8, quantization happens eagerly
        during loading, so this is mostly a no-op unless kernel wasn't set up.
        """
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Kernel setup should have happened in _create_quantizing_weight_loader
        # but handle edge cases where it didn't
        if self.kernel is None:
            self._setup_kernel(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a16_moe_quant_config(
            w1_scale=layer.w13_scale, w2_scale=layer.w2_scale, w1_zp=None, w2_zp=None
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


def quantize_in_place_and_get_scales(weight: torch.Tensor) -> torch.Tensor:
    """Quantize weight tensor to int8 in-place and return per-channel scales."""
    vmax = torch.iinfo(torch.int8).max
    scales = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] / vmax

    weight.div_(scales)
    weight.round_()
    weight.clamp_(-vmax, vmax)

    return scales
