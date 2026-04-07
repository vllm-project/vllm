# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online MXFP8 (microscaling FP8, block-32) quantization config and methods."""

from typing import Any

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8OnlineLinearMethod,
    Fp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    Mxfp8LinearOp,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Mxfp8Config(Fp8Config):
    """Config class for online MXFP8 MoE quantization."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        if activation_scheme != "dynamic":
            raise ValueError("mxfp8 only supports dynamic activation scheme.")
        super().__init__(
            is_checkpoint_fp8_serialized=False,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=None,
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin kernel supports MXFP8 on SM80+
        return 80

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return Mxfp8OnlineLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp8OnlineMoEMethod(self, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class Mxfp8OnlineLinearMethod(Fp8OnlineLinearMethod):
    """Online MXFP8 linear method.
    Loads bf16/fp16 checkpoints and quantizes weights to MXFP8 (microscaling
    FP8 with block-32 scales) during weight loading.

    Args:
        quant_config: The MXFP8 quantization config.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: "Mxfp8Config"):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        self.mxfp8_linear = Mxfp8LinearOp()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{MXFP8_BLOCK_SIZE}."
            )

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight_fp8, weight_scale = mxfp8_e4m3_quantize(layer.weight.contiguous())

        layer.input_scale = None
        replace_parameter(layer, "weight", weight_fp8.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        self.mxfp8_linear.process_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            bias=bias,
            workspace=getattr(layer, "workspace", None),
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
        )


class Mxfp8OnlineMoEMethod(Fp8OnlineMoEMethod):
    """MoE method for online MXFP8 (block) quantization."""

    uses_meta_device: bool = True

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        assert not quant_config.is_checkpoint_fp8_serialized
        assert quant_config.activation_scheme == "dynamic"

        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        self.block_quant = True
        self.weight_scale_name = "weight_scale"

        self.fp8_backend, self.experts_cls = select_mxfp8_moe_backend(config=self.moe)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            hidden_size % MXFP8_BLOCK_SIZE != 0
            or intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "Online MXFP8 MoE requires hidden/intermediate sizes divisible "
                f"by {MXFP8_BLOCK_SIZE}."
            )

        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]

    def _quantize_mxfp8_moe_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch quantization: bf16/fp16 weights -> MXFP8 (fp8 + uint8 scales)."""
        E = weight.size(0)
        first_q, first_s = mxfp8_e4m3_quantize(weight[0], is_sf_swizzled_layout=False)
        # Pre-allocate the output tensors rather than stacking.
        # This is important for consistent memory layout.
        w_quant = torch.empty(
            (E, *first_q.shape), dtype=first_q.dtype, device=weight.device
        )
        w_scales = torch.empty(
            (E, *first_s.shape), dtype=first_s.dtype, device=weight.device
        )
        w_quant[0] = first_q
        w_scales[0] = first_s
        for i in range(1, E):
            w_quant[i], w_scales[i] = mxfp8_e4m3_quantize(
                weight[i], is_sf_swizzled_layout=False
            )

        return w_quant, w_scales

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        w13, w13_scale = self._quantize_mxfp8_moe_weight(layer.w13_weight)
        w2, w2_scale = self._quantize_mxfp8_moe_weight(layer.w2_weight)

        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            layer.w13_input_scale,
            layer.w2_input_scale,
        )

        layer._already_called_process_weights_after_loading = True
