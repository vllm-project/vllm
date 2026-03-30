# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.humming_moe_utils import (
    humming_moe_align,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


try:
    from humming.dtypes import DataType
    from humming.layer import HummingMethod
    from humming.schema import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        HummingWeightSchema,
    )
    from humming.utils.weight import quantize_weight
except ImportError:
    HummingMethod = None


def assert_humming_available():
    assert HummingMethod is not None, (
        "humming is not available, please run "
        "'pip install git+https://github.com/inclusionAI/humming' to install it."
    )


def prepare_padded_shape(shape, x):
    padded_shape = math.ceil(shape / x) * x
    return padded_shape, padded_shape - shape


def prepare_param(tensor, name, extra_attrs):
    extra_attrs = extra_attrs.copy()
    scale_type = extra_attrs.pop("scale_type", None)
    param_cls_name_map = {
        "block": BlockQuantScaleParameter,
        "tensor": PerTensorScaleParameter,
        "group": GroupQuantScaleParameter,
        "channel": ChannelQuantScaleParameter,
        "input_scale": PerTensorScaleParameter,
    }

    param_cls: type[torch.nn.Parameter]
    if "packed_dim" in extra_attrs:
        param_cls = PackedvLLMParameter
    elif scale_type in param_cls_name_map:
        param_cls = param_cls_name_map[scale_type]
    elif "output_dim" in extra_attrs and "input_dim" in extra_attrs:
        param_cls = ModelWeightParameter
    elif "input_dim" in extra_attrs:
        param_cls = RowvLLMParameter
    elif "output_dim" in extra_attrs:
        param_cls = ChannelQuantScaleParameter
    else:
        param_cls = torch.nn.Parameter

    if param_cls == torch.nn.Parameter:
        param = torch.nn.Parameter(tensor, requires_grad=False)
        set_weight_attrs(param, extra_attrs)
    else:
        kwargs_keys = [
            "input_dim",
            "output_dim",
            "packed_dim",
            "packed_factor",
            "weight_loader",
        ]
        cls_kwargs = {}
        for key in extra_attrs.copy():
            if key in kwargs_keys:
                cls_kwargs[key] = extra_attrs.pop(key)

        param = param_cls(data=tensor, **cls_kwargs)
        set_weight_attrs(param, extra_attrs)
    param.param_name = name
    param.ignore_warning = True
    if scale_type in ["tensor", "input_scale"]:
        param.needs_scalar_to_array = True

    return param


def prepare_moe_param(tensor, name, extra_attrs):
    param = torch.nn.Parameter(tensor, requires_grad=False)
    if "scale_type" in extra_attrs:
        extra_attrs["quant_method"] = extra_attrs["scale_type"]

    if "input_dim" in extra_attrs and "output_dim" in extra_attrs:
        input_dim = extra_attrs["input_dim"]
        output_dim = extra_attrs["output_dim"]
        extra_attrs["is_transposed"] = input_dim < output_dim

    set_weight_attrs(param, extra_attrs)
    param.param_name = name
    return param


def may_pad_loaded_weight(param, loaded_weight):
    pad_shape = getattr(param, "pad_shape", None)
    if pad_shape is None:
        return loaded_weight
    value = 1 if loaded_weight.dtype == torch.float8_e8m0fnu else 0
    padding = []
    for x in pad_shape[::-1][: loaded_weight.ndim]:
        padding += [0, x]
    loaded_weight = torch.nn.functional.pad(
        input=loaded_weight,
        pad=padding,
        value=value,
    )
    return loaded_weight


def compressed_tensors_get_config(config: dict[str, Any], key: str):
    assert key in ["weights", "input_activations"]
    target_group_config = None
    for group_config in config["config_groups"].values():
        if "Linear" in group_config["targets"]:
            if "weights" not in group_config:
                return None
            if key not in group_config:
                return None
            target_group_config = group_config[key].copy()
            break

    if target_group_config is None:
        return None
    target_group_config["quant_method"] = config["quant_method"]
    if config["quant_method"] == "compressed-tensors":
        target_group_config["format"] = config["format"]
    elif config["quant_method"] == "modelopt":
        target_group_config["quant_algo"] = config["quant_algo"]
    return target_group_config


class HummingConfig(QuantizationConfig):
    packed_modules_mapping = {}

    def __init__(self, full_config: dict[str, Any] | None = None):
        assert_humming_available()
        self.full_config: dict[str, Any] = full_config or {}

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "humming"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HummingConfig":
        return cls(full_config=config)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        return "humming" if user_quant == "humming" else None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.hf_to_vllm_mapper = hf_to_vllm_mapper

    def is_layer_skipped(self, config: dict[str, Any], prefix: str):
        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        ignored_layers = self.get_from_keys_or(config, keys, []) or []
        if hasattr(self, "hf_to_vllm_mapper"):
            ignored_layers = self.hf_to_vllm_mapper.apply_list(ignored_layers)

        if any(module_name in prefix for module_name in ignored_layers):
            return True
        if "lm_head" in prefix:
            return True

        for regex in config.get("dynamic", {}):
            if regex[:1] != "-":
                continue
            if re.match(regex[2:], prefix):
                return True

        return False

    def get_layer_weight_schema(self, config: dict[str, Any], prefix: str):
        if self.is_layer_skipped(config, prefix):
            return None

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "weights")
            if group_config is None:
                return None
            config = group_config

        layer_config = config
        layer_dynamic = config.get("dynamic", {})
        if not isinstance(layer_dynamic, dict):
            layer_dynamic = {}
        for regex, override_config in layer_dynamic.items():
            if regex[:1] != "+":
                continue
            if re.match(regex[2:], prefix):
                layer_config = config.copy()
                layer_config.update(override_config)
                break

        if "quant_method" in layer_config:
            return BaseWeightSchema.from_config(layer_config)
        return None

    def get_layer_input_schema(self, config: dict[str, Any], prefix: str):
        if self.is_layer_skipped(config, prefix):
            return None
        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "input_activations")
            if group_config is None:
                return None
            config = group_config

        if config.get("quant_method", None) in BaseInputSchema.INPUT_SCHEMA_MAP:
            return BaseInputSchema.from_config(config)
        return None

    def get_quant_config_for_layer(
        self, prefix: str, layer_type: str
    ) -> "HummingLayerQuantizationConfig | None":
        weight_schema: BaseWeightSchema | None = None
        force_weight_schema: HummingWeightSchema | None = None

        if self.full_config:
            weight_schema = self.get_layer_weight_schema(self.full_config, prefix)

        is_online_quant = False
        online_quant_config = envs.VLLM_HUMMING_ONLINE_QUANT_CONFIG or {}
        if not self.full_config or online_quant_config.get("force_requant", False):
            online_quant_config["quant_method"] = "humming"
            schema = self.get_layer_weight_schema(online_quant_config, prefix)
            if not self.full_config:
                weight_schema = schema
                is_online_quant = True
            else:
                force_weight_schema = schema

        if weight_schema is not None:
            if weight_schema.quant_method == "mxfp4" and layer_type != "moe":
                return None
            input_schema = None
            force_input_schema = None

            if self.full_config:
                input_schema = self.get_layer_input_schema(self.full_config, prefix)

            if envs.VLLM_HUMMING_INPUT_QUANT_CONFIG:
                quant_config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG.copy()
                quant_config["quant_method"] = "humming"
                force_input_schema = self.get_layer_input_schema(quant_config, prefix)
                if input_schema is None:
                    input_schema = force_input_schema

            if force_weight_schema is not None and force_input_schema is None:
                force_input_schema = HummingInputSchema()

            return HummingLayerQuantizationConfig(
                weight_schema=weight_schema,
                input_schema=input_schema,
                force_weight_schema=force_weight_schema,
                force_input_schema=force_input_schema,
                is_online_quant=is_online_quant,
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        layer_type = "other"
        if isinstance(layer, FusedMoE):
            layer_type = "moe"
        elif isinstance(layer, LinearBase):
            layer_type = "linear"
        quant_config = self.get_quant_config_for_layer(prefix, layer_type)
        if quant_config is None:
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            elif isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            return HummingLinearMethod(quant_config)
        elif isinstance(layer, FusedMoE):
            return HummingMoEMethod(quant_config, layer.moe_config)
        return None


class HummingLayerQuantizationConfig(HummingConfig):
    def __init__(
        self,
        weight_schema: "BaseWeightSchema",
        input_schema: "BaseInputSchema | None" = None,
        force_weight_schema: "HummingWeightSchema | None" = None,
        force_input_schema: "HummingInputSchema | None" = None,
        is_online_quant: bool = False,
    ):
        self.weight_schema = weight_schema
        if input_schema is None:
            input_schema = HummingInputSchema()
        self.input_schema = input_schema
        self.force_weight_schema = force_weight_schema
        self.force_input_schema = force_input_schema
        self.is_online_quant = is_online_quant

    @classmethod
    def from_config(cls, config):
        weight_schema = BaseWeightSchema.from_config(config)
        return cls(weight_schema)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        raise NotImplementedError


class HummingLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: HummingLayerQuantizationConfig):
        self.quant_config = quant_config
        self.weight_schema = quant_config.weight_schema
        self.input_schema = quant_config.input_schema
        self.force_weight_schema = quant_config.force_weight_schema
        self.force_input_schema = quant_config.force_input_schema
        self.is_online_quant = self.quant_config.is_online_quant

    def prepare_weight_loader(self, layer: torch.nn.Module, weight_loader: Callable):
        def new_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: str | int | None = None,
        ):
            name = param.param_name
            float_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            is_unquantized = name == "weight" and loaded_weight.dtype in float_dtypes
            if is_unquantized and self.is_online_quant:
                assert isinstance(self.weight_schema, HummingWeightSchema)
                f16_dtype = DataType.from_torch_dtype(layer.param_dtype)
                has_global_scale = "TENSOR" in str(self.weight_schema.weight_scale_type)
                tensor_list = quantize_weight(
                    weight=loaded_weight,
                    dtype=self.weight_schema.b_dtype,
                    scale_dtype=self.weight_schema.bs_dtype or f16_dtype,
                    group_size=self.weight_schema.weight_scale_group_size,
                    has_zero_point=self.weight_schema.has_zero_point,
                    has_global_scale=has_global_scale,
                    is_fp_zero_point=self.weight_schema.is_fp_zero_point,
                    pack=True,
                )

                key_list = ["weight", "weight_scale", "zero_point", "global_scale"]
                for key, tensor in zip(key_list, tensor_list):
                    if tensor is None or tensor.nelement() == 0:
                        continue
                    param = getattr(layer, key)
                    param.weight_loader(param, tensor, shard_id)

                return None
            elif is_unquantized and not self.is_online_quant:
                # fallback to unquantized linear
                if layer.is_fallback:
                    return None
                layer.is_fallback = True
                for name, _ in list(layer.named_parameters()):
                    if name != "bias":
                        delattr(layer, name)
                delattr(layer, "locks")
                self.__class__ = UnquantizedLinearMethod  # type: ignore
                tensor = torch.empty(
                    (layer.output_size_per_partition, layer.input_size_per_partition),
                    dtype=layer.param_dtype,
                    device=param.device,
                )
                layer.weight = ModelWeightParameter(
                    data=tensor,
                    input_dim=1,
                    output_dim=0,
                    weight_loader=layer.extra_weight_attrs["weight_loader"],
                )
                if shard_id is not None:
                    return layer.weight.weight_loader(param, loaded_weight, shard_id)
                return layer.weight.weight_loader(param, loaded_weight)

            loaded_weight = self.weight_schema.process_loaded_weight(
                tensor=loaded_weight,
                name=name,
            )
            loaded_weight = may_pad_loaded_weight(param, loaded_weight)
            if shard_id is not None:
                return weight_loader(param, loaded_weight, shard_id)
            return weight_loader(param, loaded_weight)

        return new_weight_loader

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
        layer.is_fallback = False
        layer.param_dtype = params_dtype
        layer.input_size = input_size
        layer.output_size = output_size
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.output_partition_sizes = output_partition_sizes
        layer.extra_weight_attrs = extra_weight_attrs.copy()

        weight_loader = extra_weight_attrs.get("weight_loader", default_weight_loader)
        new_weight_loader = self.prepare_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = new_weight_loader

        for key in ["weight_block_size", "block_structure"]:
            block_size = getattr(self.weight_schema, key, None)
            if block_size is not None:
                layer.weight_block_size = block_size

        weight_tensor_attrs = self.weight_schema.get_padded_tensors_attrs(
            shape_n=layer.output_size_per_partition,
            shape_k=layer.input_size_per_partition,
            param_dtype=params_dtype,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            stack_size=len(layer.output_partition_sizes),
        )

        input_tensor_attrs = self.input_schema.get_tensors_attrs(
            shape_k=layer.input_size_per_partition,
            param_dtype=params_dtype,
            stack_size=len(layer.output_partition_sizes),
        )

        tensors_attrs = weight_tensor_attrs | input_tensor_attrs

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            extra_attrs = attrs.get("extra_attrs", {}).copy()
            extra_attrs.update(extra_weight_attrs)
            param = prepare_param(tensor, name, extra_attrs)
            setattr(layer, name, param)

        locks = torch.zeros(1024, dtype=torch.int32)
        layer.register_buffer("locks", locks)

        if self.force_input_schema is not None:
            self.input_schema = self.force_input_schema

        if not hasattr(layer, "weight"):
            param = prepare_param(torch.tensor(0), "weight", extra_weight_attrs)
            layer.weight = param

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.is_fallback:
            return None
        if not isinstance(self.weight_schema, HummingWeightSchema):
            self.weight_schema, tensors = self.weight_schema.convert_humming(
                tensors=layer.state_dict(),
                shape_n_stacks=layer.output_partition_sizes,
                shape_k_stacks=[layer.input_size_per_partition],
                param_dtype=layer.param_dtype,
            )

            self.input_schema, _ = self.input_schema.convert_humming(
                tensors=layer.state_dict(),
                shape_n_stacks=layer.output_partition_sizes,
                shape_k_stacks=[layer.input_size_per_partition],
                param_dtype=layer.param_dtype,
            )

            for name, _ in list(layer.named_parameters()):
                delattr(layer, name)

            for name, tensor in tensors.items():
                param = torch.nn.Parameter(tensor, requires_grad=False)
                setattr(layer, name, param)

            del tensors

        assert isinstance(self.weight_schema, HummingWeightSchema)
        force_requant = self.force_weight_schema is not None
        if force_requant and self.weight_schema != self.force_weight_schema:
            tensors = self.weight_schema.requant_tensors(
                tensors=layer.state_dict(),
                target_weight_schema=self.force_weight_schema,
                param_dtype=layer.param_dtype,
            )

            self.weight_schema = self.force_weight_schema

            for name, _ in list(layer.named_parameters()):
                if name != "bias":
                    delattr(layer, name)

            for name, tensor in tensors.items():
                param = torch.nn.Parameter(tensor, requires_grad=False)
                setattr(layer, name, param)

            del tensors

        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=layer.output_size_per_partition,
            shape_k=layer.input_size_per_partition,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            has_bias=layer.has_bias,
            torch_dtype=layer.param_dtype,
        )

        HummingMethod.transform_humming_layer(layer, already_padded=True)
        HummingMethod.prepare_default_kernel_configs(
            layer,
            use_batch_invariance=envs.VLLM_BATCH_INVARIANT,
            use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return HummingMethod.forward_layer(layer, x)


class HummingMoEMethod(FusedMoEMethodBase):
    def __init__(
        self, quant_config: HummingLayerQuantizationConfig, moe: "FusedMoEConfig"
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.moe = moe
        self.weight_schema = quant_config.weight_schema
        self.input_schema = quant_config.input_schema
        self.force_weight_schema = quant_config.force_weight_schema
        self.force_input_schema = quant_config.force_input_schema

    def prepare_weight_loader(self, layer, weight_loader):
        def new_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int | None = None,
            return_success: bool = False,
        ):
            name = param.param_name
            float_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            is_unquantized = name == "weight" and loaded_weight.dtype in float_dtypes
            if is_unquantized:
                assert isinstance(self.weight_schema, HummingWeightSchema)
                f16_dtype = DataType.from_torch_dtype(layer.param_dtype)
                has_global_scale = "TENSOR" in str(self.weight_schema.weight_scale_type)
                tensor_list = quantize_weight(
                    weight=loaded_weight,
                    dtype=self.weight_schema.b_dtype,
                    scale_dtype=self.weight_schema.bs_dtype or f16_dtype,
                    group_size=self.weight_schema.weight_scale_group_size,
                    has_zero_point=self.weight_schema.has_zero_point,
                    has_global_scale=has_global_scale,
                    is_fp_zero_point=self.weight_schema.is_fp_zero_point,
                    pack=True,
                )

                key_list = ["weight", "weight_scale", "zero_point", "global_scale"]
                success = True
                for key, tensor in zip(key_list, tensor_list):
                    if tensor is None or tensor.nelement() == 0:
                        continue
                    sublayer_name = "w2" if shard_id == "w2" else "w13"

                    param = getattr(layer, sublayer_name + "_" + key)
                    part_subccess = param.weight_loader(
                        param=param,
                        loaded_weight=tensor.cpu(),
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=return_success,
                    )
                    success = success and part_subccess

                return success if return_success else None

            loaded_weight = self.weight_schema.process_loaded_weight(
                tensor=loaded_weight,
                name=name,
            )
            loaded_weight = may_pad_loaded_weight(param, loaded_weight)
            return weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=return_success,
            )

        return new_weight_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.param_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader", default_weight_loader)
        weight_loader = self.prepare_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = weight_loader

        layer.sublayer_configs = {
            "w13": {
                "shape_n": intermediate_size_per_partition * 2,
                "shape_k": hidden_size,
                "tensors_attrs": self.weight_schema.get_padded_tensors_attrs(
                    shape_n=intermediate_size_per_partition * 2,
                    shape_k=hidden_size,
                    num_experts=num_experts,
                    param_dtype=params_dtype,
                    has_bias=self.moe.has_bias,
                    pad_n_to_multiple=256,
                    pad_k_to_multiple=128,
                ),
            },
            "w2": {
                "shape_n": hidden_size,
                "shape_k": intermediate_size_per_partition,
                "tensors_attrs": self.weight_schema.get_padded_tensors_attrs(
                    shape_n=hidden_size,
                    shape_k=intermediate_size_per_partition,
                    num_experts=num_experts,
                    param_dtype=params_dtype,
                    has_bias=self.moe.has_bias,
                    pad_n_to_multiple=256,
                    pad_k_to_multiple=128,
                ),
            },
        }

        for sublayer_name, configs in layer.sublayer_configs.items():
            for name, attrs in configs["tensors_attrs"].items():
                tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
                param = torch.nn.Parameter(tensor, requires_grad=False)
                extra_attrs = attrs.get("extra_attrs", {}).copy()
                extra_attrs.update(extra_weight_attrs)
                param = prepare_moe_param(tensor, name, extra_attrs)
                setattr(layer, f"{sublayer_name}_{name}", param)

        if self.force_input_schema is not None:
            self.input_schema = self.force_input_schema

        locks = torch.zeros(1024, dtype=torch.int32)
        layer.register_buffer("locks", locks)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return

    def apply_activation(
        self, layer: torch.nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        inputs_flat = inputs.view(-1, inputs.size(-1))
        if layer.activation.is_gated:
            outputs_flat = torch.empty(
                (inputs_flat.size(0), inputs_flat.size(1) // 2),
                dtype=inputs_flat.dtype,
                device=inputs.device,
            )
        else:
            outputs_flat = torch.empty_like(inputs_flat)

        apply_moe_activation(layer.activation, outputs_flat, inputs_flat)
        return outputs_flat.view(*inputs.shape[:-1], -1)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for sublayer_name, configs in layer.sublayer_configs.items():
            input_schema = self.input_schema
            weight_schema = self.weight_schema
            if not isinstance(weight_schema, HummingWeightSchema):
                tensors: dict[str, torch.Tensor] = dict(
                    (key.removeprefix(sublayer_name + "_"), value)
                    for key, value in layer.state_dict().items()
                    if key.startswith(sublayer_name + "_")
                )

                shape_k_stacks = [configs["shape_k"]]
                shape_n_stacks = [configs["shape_n"]]
                if sublayer_name == "w13":
                    shape_n_stacks = [configs["shape_n"] // 2] * 2

                weight_schema, tensors = weight_schema.convert_humming(
                    tensors=tensors,
                    shape_n_stacks=shape_n_stacks,
                    shape_k_stacks=shape_k_stacks,
                    param_dtype=layer.param_dtype,
                    num_experts=layer.num_experts,
                )

                input_schema, _ = input_schema.convert_humming(
                    tensors=tensors,
                    shape_n_stacks=shape_n_stacks,
                    shape_k_stacks=shape_k_stacks,
                    param_dtype=layer.param_dtype,
                    num_experts=layer.num_experts,
                )

                for name, _ in list(layer.named_parameters()):
                    if not name.startswith(sublayer_name + "_"):
                        continue
                    delattr(layer, name)

                for name, tensor in tensors.items():
                    name = f"{sublayer_name}_{name}"
                    param = torch.nn.Parameter(tensor, requires_grad=False)
                    setattr(layer, name, param)

            assert isinstance(weight_schema, HummingWeightSchema)
            force_requant = self.force_weight_schema is not None
            if force_requant and weight_schema != self.force_weight_schema:
                tensors = dict(
                    (key.removeprefix(sublayer_name + "_"), value)
                    for key, value in layer.state_dict().items()
                    if key.startswith(sublayer_name + "_")
                )

                tensors = weight_schema.requant_tensors(
                    tensors=tensors,
                    target_weight_schema=self.force_weight_schema,
                    param_dtype=layer.param_dtype,
                )

                weight_schema = self.force_weight_schema

                for name, _ in list(layer.named_parameters()):
                    if not name.startswith(sublayer_name + "_"):
                        continue
                    if name == sublayer_name + "_bias":
                        continue
                    delattr(layer, name)

                for name, tensor in tensors.items():
                    name = f"{sublayer_name}_{name}"
                    param = torch.nn.Parameter(tensor, requires_grad=False)
                    setattr(layer, name, param)

                del tensors

            HummingMethod.prepare_layer_meta(
                layer=layer,
                shape_n=configs["shape_n"],
                shape_k=configs["shape_k"],
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema,
                weight_schema=weight_schema,
                has_bias=self.moe.has_bias,
                num_experts=layer.num_experts,
                top_k=self.moe.experts_per_token,
                is_moe_down=sublayer_name == "w2",
                torch_dtype=layer.param_dtype,
                sublayer_name=sublayer_name,
            )

            HummingMethod.transform_humming_layer(
                layer=layer,
                sublayer_name=sublayer_name,
                already_padded=True,
            )
            HummingMethod.prepare_default_kernel_configs(
                layer,
                use_batch_invariance=envs.VLLM_BATCH_INVARIANT,
                use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
                sublayer_name=sublayer_name,
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sorted_token_ids, expert_ids, num_tokens_padded = humming_moe_align(
            layer.humming_block_size_configs["w13"],
            topk_ids=topk_ids,
            num_experts=layer.num_experts,
            expert_map=layer.expert_map,
        )

        output1 = HummingMethod.forward_layer(
            layer=layer,
            inputs=x,
            outputs=None,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            sublayer_name="w13",
        )

        input2 = self.apply_activation(layer, output1)

        output2 = HummingMethod.forward_layer(
            layer=layer,
            inputs=input2,
            outputs=None,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            sublayer_name="w2",
        )

        return torch.sum(output2, dim=1, out=x)
