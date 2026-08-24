# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization support for GLM W4AFP8 checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    is_layer_skipped,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )


def _convert_signed_int4_to_uint4b8(weight: torch.Tensor) -> torch.Tensor:
    """Convert packed two's-complement INT4 to Humming's biased uint4."""
    if weight.dtype != torch.int8:
        raise ValueError("W4AFP8 packed weights must use int8 storage")
    if weight.size(-1) % 4:
        raise ValueError("W4AFP8 packed weight rows must contain full int32 words")
    return weight.view(torch.uint8).bitwise_xor(0x88).contiguous().view(torch.int32)


class _W4AFP8HummingWeightSchema:
    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        del shape_n_stacks, shape_k_stacks, num_experts
        from vllm.utils.humming import HummingWeightSchema, dtypes

        output = {
            "weight": _convert_signed_int4_to_uint4b8(tensors["weight"]),
            "weight_scale": tensors["weight_scale_inv"].to(param_dtype),
        }
        schema = HummingWeightSchema(
            b_dtype=dtypes.DataType.from_str("uint4"),
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=self.group_size,
            has_zero_point=False,
        )
        return schema, output


class W4AFP8Config(QuantizationConfig):
    """Configuration for GLM W4AFP8 checkpoint serialization."""

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        if group_size != 128:
            raise ValueError("W4AFP8 currently supports group_size=128 only")
        self.ignored_layers = ignored_layers or []
        self.group_size = group_size
        self.linear_quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=self.ignored_layers,
            weight_block_size=[128, 128],
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W4AFP8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        if quant_method != "w4afp8":
            raise ValueError(f"Unsupported W4AFP8 quant_method: {quant_method}")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config,
                ["modules_to_not_convert"],
                None,
            )
        return cls(
            ignored_layers=ignored_layers,
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
        self.linear_quant_config.ignored_layers = self.ignored_layers

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self.linear_quant_config)
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return W4AFP8MoEMethod(self, layer.moe_config)
        return None


class W4AFP8MoEMethod(FusedMoEMethodBase):
    """Convert packed signed INT4 experts to Humming's uint4 layout."""

    def __init__(
        self,
        quant_config: W4AFP8Config,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            hidden_size % self.group_size
            or intermediate_size_per_partition % self.group_size
        ):
            raise ValueError("W4AFP8 expert dimensions must be divisible by group_size")

        layer.params_dtype = params_dtype
        layer.weight_block_size = None
        weight_specs = {
            "w13_weight": (
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
            ),
            "w2_weight": (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
            ),
        }
        for name, shape in weight_specs.items():
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        scale_attrs = {
            **extra_weight_attrs,
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        }
        scale_specs = {
            "w13_weight_scale_inv": (
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // self.group_size,
            ),
            "w2_weight_scale_inv": (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
            ),
        }
        for name, scale_shape in scale_specs.items():
            param = torch.nn.Parameter(
                torch.empty(scale_shape, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, scale_attrs)

        input_scale_specs = {
            "w13_input_scale": (num_experts, self.moe.w13_num_shards),
            "w2_input_scale": (num_experts,),
        }
        for name, input_scale_shape in input_scale_specs.items():
            param = torch.nn.Parameter(
                torch.ones(input_scale_shape, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(
                param,
                {
                    **extra_weight_attrs,
                    "is_split_input_scale": name == "w13_input_scale",
                    "is_w4afp8_input_scale": True,
                },
            )

    @staticmethod
    def _prepare_input_scales(
        layer: RoutedExperts,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            layer.w13_input_scale.max().to(torch.float32).reshape(1),
            layer.w2_input_scale.max().to(torch.float32).reshape(1),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if self.moe.moe_parallel_config.ep_size != 1:
            raise NotImplementedError(
                "W4AFP8 currently supports expert parallel size 1 only"
            )
        if self.moe.moe_parallel_config.use_batched_activation_format:
            raise NotImplementedError(
                "W4AFP8 does not support the batched-expert activation format"
            )
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "W4AFP8 does not support applying router weights on input"
            )
        if getattr(self, "processed", False):
            return

        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            convert_to_humming_moe_kernel_format,
            make_humming_moe_kernel,
            make_humming_moe_quant_config,
            select_humming_moe_experts,
            weight_schema_to_quant_key,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8StaticTensorSym,
        )
        from vllm.utils.humming import HummingInputSchema, dtypes

        a1_scale, a2_scale = self._prepare_input_scales(layer)
        self.humming_configs = convert_to_humming_moe_kernel_format(
            layer,
            weight_schema=_W4AFP8HummingWeightSchema(self.group_size),
            input_schema=HummingInputSchema(a_dtype=dtypes.float8e4m3),
        )
        weight_key = weight_schema_to_quant_key(layer.weight_schemas["w13"])
        experts_cls = select_humming_moe_experts(
            config=self.moe,
            weight_key=weight_key,
            activation_key=kFp8StaticTensorSym,
        )
        if experts_cls is None:
            raise NotImplementedError(
                "The current deployment configuration is not supported by Humming"
            )

        self.moe_quant_config = make_humming_moe_quant_config(
            quant_dtype=str(layer.input_schemas["w13"].a_dtype),
            weight_dtype=str(layer.weight_schemas["w13"].b_dtype),
            weight_group_shape=GroupShape(1, self.group_size),
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            humming_configs=self.humming_configs,
        )
        self.moe_kernel = make_humming_moe_kernel(
            self.moe_quant_config,
            self.moe,
            experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.processed = True

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        del layer
        return self.moe_quant_config

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
