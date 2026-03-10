# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from humming.layer import HummingMethod
from humming.schema import (
    BaseInputSchema,
    BaseWeightSchema,
    HummingInputSchema,
    HummingWeightSchema,
)
from vllm import envs
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
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
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


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

    if "packed_dim" in extra_attrs:
        param_cls = PackedvLLMParameter
    elif scale_type in param_cls_name_map:
        param_cls = param_cls_name_map[scale_type]
    elif "output_dim" not in extra_attrs or "input_dim" not in extra_attrs:
        param_cls = torch.nn.Parameter
    else:
        param_cls = ModelWeightParameter

    if param_cls == torch.nn.Parameter:
        param = param_cls(tensor, requires_grad=False)
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
    for x in pad_shape[::-1][:loaded_weight.ndim]:
        padding += [0, x]
    loaded_weight = torch.nn.functional.pad(
        input=loaded_weight,
        pad=padding,
        value=value,
    )
    return loaded_weight


def compressed_tensors_get_config(config: dict[str, Any], key: str) -> dict[str, Any]:
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
    target_group_config.pop("dynamic", None)
    target_group_config["quant_method"] = config["quant_method"]
    if config["quant_method"] == "compressed-tensors":
        target_group_config["format"] = config["format"]
    elif config["quant_method"] == "modelopt":
        target_group_config["quant_algo"] = config["quant_algo"]
    return target_group_config


class HummingLayerQuantizationConfig(QuantizationConfig):
    def __init__(
        self,
        weight_schema: BaseWeightSchema,
        input_schema: BaseInputSchema | None = None,
    ):
        self.weight_schema = weight_schema
        if input_schema is None:
            input_schema = HummingInputSchema()
        self.input_schema = input_schema

    @classmethod
    def from_config(cls, config):
        weight_schema = BaseWeightSchema.from_config(config)
        return cls(weight_schema)

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def get_min_capability(cls):
        return 75

    @classmethod
    def get_name(cls):
        return "humming"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        raise NotImplementedError


class HummingConfig(QuantizationConfig):
    def __init__(self, full_config: dict[str, Any] | None = None):
        self.full_config: dict[str, Any] = full_config or {}
        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        self.ignored_layers = self.get_from_keys_or(full_config, keys, []) or []

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
        self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    def is_layer_skipped(self, config: dict[str, Any], prefix: str):
        ignored_layers = self.ignored_layers
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

    def get_layer_weight_schema(
        self, config: dict[str, Any], prefix: str
    ) -> BaseWeightSchema | None:
        if self.is_layer_skipped(config, prefix):
            return None

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "weights")
            if group_config is None:
                return None
            config = group_config

        layer_config = config
        for regex, override_config in config.get("dynamic", {}).items():
            if regex[:1] != "+":
                continue
            if re.match(regex[2:], prefix):
                layer_config = config.copy()
                layer_config.update(override_config)
                break

        if "quant_method" in layer_config:
            return BaseWeightSchema.from_config(layer_config)
        return None

    def get_layer_input_schema(
        self, config: dict[str, Any], prefix: str
    ) -> BaseInputSchema | None:
        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            group_config = compressed_tensors_get_config(config, "input_activations")
            if group_config is None:
                return None
            config = group_config

        if "quant_method" in config:
            return BaseInputSchema.from_config(config)
        return None

    def get_quant_config_for_layer(
        self, prefix: str, layer_type: str
    ) -> HummingLayerQuantizationConfig | None:
        weight_schema = self.get_layer_weight_schema(self.full_config, prefix)

        if not self.full_config:
            weight_config = envs.VLLM_HUMMING_ONLINE_QUANT_CONFIG or {}
            weight_schema = self.get_layer_weight_schema(weight_config, prefix)

        if weight_schema is not None:
            if weight_schema.quant_method == "mxfp4" and layer_type != "moe":
                return None
            input_schema = None
            if self.full_config:
                input_schema = self.get_layer_input_schema(self.full_config, prefix)

            return HummingLayerQuantizationConfig(weight_schema, input_schema)
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


class HummingLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: HummingLayerQuantizationConfig):
        self.quant_config = quant_config
        self.weight_schema: BaseWeightSchema = quant_config.weight_schema
        self.input_schema: BaseInputSchema = quant_config.input_schema

    def prepare_weight_loader(self, weight_loader):
        def new_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: str | int | None = None,
        ):
            name = param.param_name
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
        layer.extra_weight_attrs = extra_weight_attrs

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

        print(self.input_schema, tensors_attrs)

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            extra_attrs = attrs.get("extra_attrs", {}).copy()
            extra_attrs.update(extra_weight_attrs)
            weight_loader = extra_attrs.get("weight_loader", default_weight_loader)
            extra_attrs["weight_loader"] = self.prepare_weight_loader(weight_loader)
            param = prepare_param(tensor, name, extra_attrs)
            setattr(layer, name, param)

        locks = torch.zeros(1024, dtype=torch.int32)
        layer.register_buffer("locks", locks)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
            use_stream_k=not vllm_is_batch_invariant(),
            use_f16_accum=False,
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
        self.weight_schema: BaseWeightSchema = quant_config.weight_schema
        self.input_schema: BaseInputSchema = quant_config.input_schema

    def prepare_weight_loader(self, weight_loader):
        def new_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: int | None = None,
            expert_id: int | None = None,
            return_success: bool = False,
        ):
            name = param.param_name
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
                weight_loader = extra_attrs.get("weight_loader", default_weight_loader)
                weight_loader = self.prepare_weight_loader(weight_loader)
                extra_attrs["weight_loader"] = weight_loader
                param = prepare_moe_param(tensor, name, extra_attrs)
                setattr(layer, f"{sublayer_name}_{name}", param)

        locks = torch.zeros(1024, dtype=torch.int32)
        layer.register_buffer("locks", locks)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return

    def prepare_activation_kwargs(self, layer: torch.nn.Module) -> dict[str, str]:
        if layer.activation == MoEActivation.SILU:
            return {"activation_type": "silu_glu"}
        elif layer.activation == MoEActivation.SWIGLUOAI:
            activation_func_impl = """
            const float g = fminf(a.x, 7);
            const float u = fmaxf(fminf(a.y, 7), -7);
            return (u + 1.0f) * __fdividef(g, 1.0f + __expf(-g * 1.702));
            """
            return {
                "activation_type": "custom_glu",
                "custom_activation_func_impl": activation_func_impl,
            }

        return {}

    def may_apply_activation(
        self, layer: torch.nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        if self.prepare_activation_kwargs(layer):
            return inputs

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
            input_schema = weight_schema = None
            if not isinstance(self.weight_schema, HummingWeightSchema):
                tensors: dict[str, torch.Tensor] = dict(
                    (key.removeprefix(sublayer_name + "_"), value)
                    for key, value in layer.state_dict().items()
                    if key.startswith(sublayer_name + "_")
                )

                shape_k_stacks = [configs["shape_k"]]
                shape_n_stacks = [configs["shape_n"]]
                if sublayer_name == "w13":
                    shape_n_stacks = [configs["shape_n"] // 2] * 2

                weight_schema, tensors = self.weight_schema.convert_humming(
                    tensors=tensors,
                    shape_n_stacks=shape_n_stacks,
                    shape_k_stacks=shape_k_stacks,
                    param_dtype=layer.param_dtype,
                    num_experts=layer.num_experts,
                )

                input_schema, _ = self.input_schema.convert_humming(
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

            HummingMethod.prepare_layer_meta(
                layer=layer,
                shape_n=configs["shape_n"],
                shape_k=configs["shape_k"],
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or self.input_schema,
                weight_schema=weight_schema or self.weight_schema,
                has_bias=self.moe.has_bias,
                num_experts=layer.num_experts,
                top_k=self.moe.experts_per_token,
                is_moe_down=sublayer_name == "w2",
                torch_dtype=layer.param_dtype,
                sublayer_name=sublayer_name,
            )

            should_prepare_for_glu = sublayer_name == "w13" and layer.activation in [
                MoEActivation.SILU
            ]
            HummingMethod.transform_humming_layer(
                layer=layer,
                sublayer_name=sublayer_name,
                should_prepare_for_glu=should_prepare_for_glu,
                already_padded=True,
            )
            activation_kwargs = {}
            if sublayer_name == "w13":
                activation_kwargs = self.prepare_activation_kwargs(layer)
            HummingMethod.prepare_default_kernel_configs(
                layer,
                use_stream_k=not vllm_is_batch_invariant(),
                use_f16_accum=False,
                sublayer_name=sublayer_name,
                **activation_kwargs,
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

        input2 = self.may_apply_activation(layer, output1)

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
