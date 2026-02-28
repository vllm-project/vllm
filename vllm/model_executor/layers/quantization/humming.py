# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import math
from typing import Any

import regex as re
import torch
from humming import dtypes
from humming.layer import HummingLayerMeta, HummingMethod

from vllm import envs
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
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
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
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
from vllm.model_executor.layers.quantization.utils.humming_weight_utils import (
    WEIGHT_CONVERTER_MAP,
)
from vllm.model_executor.utils import set_weight_attrs


def get_full_shape_nk(layer: torch.nn.Module, shape_n: int, shape_k: int):
    if isinstance(layer, ColumnParallelLinear):
        return shape_n * layer.tp_size, shape_k
    elif isinstance(layer, RowParallelLinear):
        return shape_n, shape_k * layer.tp_size
    elif isinstance(layer, ReplicatedLinear):
        return shape_n, shape_k
    else:
        raise ValueError("unsupported layer type: " + layer.__class__.__name__)


def narrow_tensors(
    tensors: dict[str, torch.Tensor],
    shape_n: int,
    shape_k: int,
    tp_size: int,
    tp_rank: int,
    is_row_parallel: bool,
) -> dict[str, torch.Tensor]:
    if tp_size == 1:
        return tensors

    full_size = 0
    for key, tensor in tensors.copy().items():
        narrow_dim = None
        if key in ["weight", "weight_scale", "zero_point"]:
            if not is_row_parallel:
                narrow_dim = -2
                full_size = shape_n
            elif is_row_parallel:
                narrow_dim = -1
                full_size = shape_k
        elif key == "bias" and not is_row_parallel:
            narrow_dim = -1
            full_size = shape_n

        if narrow_dim is None:
            continue

        size = tensor.size(narrow_dim)
        if size == 1:
            continue

        assert full_size % tp_size == 0
        split_size = full_size // tp_size
        assert size * split_size % full_size == 0
        new_size = size * split_size // full_size

        tensor = tensor.narrow(narrow_dim, new_size * tp_rank, new_size)
        tensors[key] = tensor.contiguous()

    return tensors


def get_ignored_layers(config):
    if "ignored_layers" in config:
        return config["ignored_layers"]
    if "modules_to_not_convert" in config:
        return config["modules_to_not_convert"]
    return []


def is_layer_skipped(prefix: str, ignore_layers: list[str]):
    return any(module_name in prefix for module_name in ignore_layers)


def parse_single_config(config):
    if "quant_method" not in config:
        return

    quant_method = config["quant_method"]

    if quant_method == "gptq":
        desc_act = config.get("desc_act", False)
        assert not desc_act, "desc_act is not supported by humming"
        result_dict = {
            "has_dynamic_zp": not config.get("sym", True),
            "block_shape": (1, config["group_size"]),
            "b_dtype": dtypes.DataType.from_str("uint" + str(config["bits"])),
        }
    elif quant_method == "awq":
        result_dict = {
            "has_dynamic_zp": config.get("zero_point", False),
            "block_shape": (1, config["group_size"]),
            "b_dtype": dtypes.DataType.from_str("uint" + str(config["bits"])),
        }
    elif quant_method == "fp8":
        b_dtype = dtypes.DataType.from_str("float8" + config.get("fmt", "e4m3"))
        result_dict = {
            "block_shape": config.get("weight_block_size", (0, 0)),
            "b_dtype": b_dtype,
        }
    elif quant_method == "modelopt":
        assert "quant_algo" in config
        quant_algo = config["quant_algo"]
        if quant_algo.lower() == "nvfp4":
            result_dict = {
                "block_shape": (1, 16),
                "has_global_scale": True,
                "b_dtype": dtypes.float4e2m1,
                "bs_dtype": dtypes.float8e4m3,
            }
        elif quant_algo.lower() == "mxfp8":
            result_dict = {
                "block_shape": (1, 32),
                "b_dtype": dtypes.float8e4m3,
                "bs_dtype": dtypes.float8e8m0,
            }
        else:
            raise ValueError(f"Invalid modelopt algo: {quant_algo}")

    elif quant_method == "mxfp4":
        result_dict = {
            "block_shape": (1, 32),
            "b_dtype": dtypes.float4e2m1,
            "bs_dtype": dtypes.float8e8m0,
        }
    elif quant_method == "bitnet":
        result_dict = {
            "block_shape": (0, 0),
            "b_dtype": dtypes.uint2,
        }
    else:
        raise ValueError(f"Invalid quant_method: {quant_method}")

    result_dict["weight_scale_group_size_n"] = result_dict["block_shape"][0]
    result_dict["weight_scale_group_size_k"] = result_dict["block_shape"][1]
    result_dict["ckpt_quant_method"] = quant_method

    return result_dict


def parse_single_input_config(config):
    config = config.copy()
    if "input_dtype" not in config:
        return {}

    return {
        "a_dtype": config["input_dtype"],
        "input_scale_group_size_k": config.get("group_size", 0),
    }


def prepare_padded_shape(shape, x):
    padded_shape = math.ceil(shape / x) * x
    return padded_shape, padded_shape - shape


class HummingLayerQuantizationConfig(QuantizationConfig):
    def __init__(
        self,
        a_dtype: str | None,
        b_dtype: str,
        c_dtype: str | None,
        bs_dtype: str | None,
        ckpt_quant_method: str | None,
        input_scale_group_size_n: int,
        input_scale_group_size_k: int,
        weight_scale_group_size_n: int,
        weight_scale_group_size_k: int,
        has_dynamic_zp: bool,
        has_global_scale: bool,
    ) -> None:
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.bs_dtype = bs_dtype
        self.ckpt_quant_method = ckpt_quant_method
        self.input_scale_group_size_n = input_scale_group_size_n
        self.input_scale_group_size_k = input_scale_group_size_k
        self.weight_scale_group_size_n = weight_scale_group_size_n
        self.weight_scale_group_size_k = weight_scale_group_size_k
        self.has_dynamic_zp = has_dynamic_zp
        self.has_global_scale = has_global_scale

    @classmethod
    def from_config(cls, config):
        return cls(
            a_dtype=config.get("a_dtype", None),
            b_dtype=config["b_dtype"],
            c_dtype=config.get("c_dtype", None),
            bs_dtype=config.get("bs_dtype", None),
            ckpt_quant_method=config.get("ckpt_quant_method", None),
            input_scale_group_size_n=config.get("input_scale_group_size_n", 0),
            input_scale_group_size_k=config.get("input_scale_group_size_k", 0),
            weight_scale_group_size_n=config.get("weight_scale_group_size_n", 0),
            weight_scale_group_size_k=config.get("weight_scale_group_size_k", 0),
            has_dynamic_zp=config.get("has_dynamic_zp", False),
            has_global_scale=config.get("has_global_scale", False),
        )

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
    ) -> "QuantizeMethodBase | None":
        raise NotImplementedError


class HummingConfig(QuantizationConfig):
    def __init__(self, full_config: dict[str, Any] | None = None) -> None:
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
        if user_quant == "humming":
            return cls.get_name()
        return None

    @classmethod
    def is_layer_skipped(cls, config: dict[str, Any], prefix: str):
        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        ignored_layers = cls.get_from_keys_or(config, keys, [])
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

    @classmethod
    def get_layer_weight_quant_config(
        cls, config: dict[str, Any], prefix: str, layer_type: str
    ) -> dict[str, Any] | None:
        if cls.is_layer_skipped(config, prefix):
            return None

        layer_config = parse_single_config(config)
        for regex, override_config in config.get("dynamic", {}).items():
            if regex[:1] != "+":
                continue
            if re.match(regex[2:], prefix):
                layer_config = config.copy()
                layer_config.update(override_config)
                layer_config = parse_single_config(layer_config)
                break

        if layer_config is not None:
            ckpt_quant_method = layer_config.get("ckpt_quant_method", "")
            if ckpt_quant_method == "mxfp4" and layer_type != "moe":
                return None

        return layer_config

    @classmethod
    def get_layer_input_quant_config(cls, prefix: str) -> dict[str, Any]:
        config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG
        if config is None:
            return {}

        layer_config = parse_single_input_config(config)
        for regex, override_config in config.get("dynamic", {}).items():
            if regex[:1] == "-":
                return {}
            elif regex[:1] != "+":
                continue
            if re.match(regex[2:], prefix):
                layer_config = config.copy()
                layer_config.update(override_config)
                layer_config = parse_single_input_config(layer_config)
                break

        return layer_config

    def get_quant_config_for_layer(
        self, prefix: str, layer_type: str
    ) -> HummingLayerQuantizationConfig | None:
        weight_config = self.get_layer_weight_quant_config(
            self.full_config, prefix, layer_type
        )

        if not self.full_config:
            weight_config = envs.VLLM_HUMMING_ONLINE_QUANT_CONFIG or {}
            weight_config = self.get_layer_weight_quant_config(
                weight_config, prefix, layer_type
            )
            if weight_config is not None:
                weight_config["ckpt_quant_method"] = None

        if weight_config is not None:
            config = weight_config.copy()
            config.update(self.get_layer_input_quant_config(prefix))
            return HummingLayerQuantizationConfig.from_config(config)
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
        ckpt_quant_method = self.quant_config.ckpt_quant_method
        weight_converter_cls = WEIGHT_CONVERTER_MAP[ckpt_quant_method]
        self.weight_converter = weight_converter_cls(quant_config)

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
        f16_dtype = dtypes.DataType.from_torch_dtype(params_dtype)
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.output_size = output_size
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        group_size = self.quant_config.weight_scale_group_size_k
        assert input_size_per_partition % group_size == 0

        self.quant_config.a_dtype = self.quant_config.a_dtype or f16_dtype
        self.quant_config.c_dtype = self.quant_config.c_dtype or f16_dtype
        self.quant_config.bs_dtype = self.quant_config.bs_dtype or f16_dtype
        is_f16_accum_supported = (
            self.quant_config.c_dtype == dtypes.float16
            and self.quant_config.a_dtype in [dtypes.float16, dtypes.float8e4m3]
        )
        self.use_f16_accum = is_f16_accum_supported and envs.VLLM_HUMMING_USE_F16_ACCUM

        shape_n, pad_shape_n = prepare_padded_shape(output_size_per_partition, 256)
        shape_k, pad_shape_k = prepare_padded_shape(input_size_per_partition, 128)
        meta = HummingLayerMeta(
            a_dtype=self.quant_config.a_dtype,
            b_dtype=self.quant_config.b_dtype,
            c_dtype=self.quant_config.c_dtype,
            bs_dtype=self.quant_config.bs_dtype,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            has_bias=layer.has_bias,
            weight_scale_group_size=self.quant_config.weight_scale_group_size_k,
            has_dynamic_zp=self.quant_config.has_dynamic_zp,
            has_global_scale=self.quant_config.has_global_scale,
        )
        self.meta = meta
        HummingMethod.create_weights(layer, meta)
        self.process_params(layer)

    def process_params(self, layer: torch.nn.Module):
        weight_loader = self.get_weight_loader(layer)
        names = ["weight", "weight_scale", "zero_point", "global_scale", "bias"]
        unused_names = list(self.weight_converter.unused_names)

        for name in names:
            param = getattr(layer, name, None)
            if param is None:
                continue

            set_weight_attrs(param, {"weight_loader": weight_loader})
            param.param_name = name
            param.ckpt_name = self.weight_converter.get_ckpt_name(name)
            if name == param.ckpt_name:
                continue

            buffer = torch.nn.Buffer(param.data)

            # rename param name (layer_name -> ckpt_name) for weight loading
            setattr(layer, param.ckpt_name, param)
            # convert from a Parameter to a Buffer to avoid conflicts with ckpt_name
            delattr(layer, param.param_name)
            setattr(layer, param.param_name, buffer)

        # set unused names to empty tensors and empty weight_loaders
        # to avoid loading errors
        for name in names + unused_names:
            ckpt_name = name
            if name not in unused_names:
                ckpt_name = self.weight_converter.get_ckpt_name(name)
            if hasattr(layer, ckpt_name):
                continue

            tensor = torch.tensor([])
            param = torch.nn.Parameter(tensor, requires_grad=False)

            def empty_weight_loader(*args, **kwargs):
                return

            set_weight_attrs(param, {"weight_loader": empty_weight_loader})
            setattr(layer, ckpt_name, param)

    def get_weight_loader(self, layer: torch.nn.Module):
        shard_id_map = {"q": 0, "k": 1, "v": 2}

        def weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: str | int | None = None,
        ):
            param_name = param.param_name
            if isinstance(shard_id, str):
                shard_id = shard_id_map[shard_id]
            offset_n = sum(layer.logical_widths[: (shard_id or 0)])
            shape_n = self.meta.shape_n - self.meta.pad_shape_n
            if isinstance(shard_id, int):
                shape_n = layer.logical_widths[shard_id]
            shape_k = self.meta.shape_k - self.meta.pad_shape_k
            shape_n, shape_k = get_full_shape_nk(layer, shape_n, shape_k)
            data = self.weight_converter.convert(
                loaded_weight,
                param_name,
                shape_n=shape_n,
                shape_k=shape_k,
            )

            data = narrow_tensors(
                tensors=data,
                shape_n=shape_n,
                shape_k=shape_k,
                tp_size=layer.tp_size,
                tp_rank=layer.tp_rank,
                is_row_parallel=isinstance(layer, RowParallelLinear),
            )

            HummingMethod.load_weight(
                layer=layer,
                offset_n=offset_n,
                packed=True,
                **data,
            )

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        HummingMethod.finish_load(layer)
        use_stream_k = not vllm_is_batch_invariant()
        HummingMethod.prepare_default_kernel_configs(
            layer,
            use_stream_k=use_stream_k,
            use_f16_accum=self.use_f16_accum,
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
        ckpt_quant_method = self.quant_config.ckpt_quant_method
        weight_converter_cls = WEIGHT_CONVERTER_MAP[ckpt_quant_method]
        self.weight_converter = weight_converter_cls(quant_config)
        self.is_loaded_weight_narrowed = ckpt_quant_method == "mxfp4"

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
        layer.top_k = self.moe.experts_per_token

        group_size = self.quant_config.weight_scale_group_size_k
        assert intermediate_size_per_partition % group_size == 0

        f16_dtype = dtypes.DataType.from_torch_dtype(params_dtype)
        self.quant_config.a_dtype = self.quant_config.a_dtype or f16_dtype
        self.quant_config.c_dtype = self.quant_config.c_dtype or f16_dtype
        self.quant_config.bs_dtype = self.quant_config.bs_dtype or f16_dtype

        is_f16_accum_supported = (
            self.quant_config.c_dtype == dtypes.float16
            and self.quant_config.a_dtype in [dtypes.float16, dtypes.float8e4m3]
        )
        self.use_f16_accum = is_f16_accum_supported and envs.VLLM_HUMMING_USE_F16_ACCUM

        base_meta = HummingLayerMeta(
            a_dtype=self.quant_config.a_dtype,
            b_dtype=self.quant_config.b_dtype,
            c_dtype=self.quant_config.c_dtype,
            bs_dtype=self.quant_config.bs_dtype,
            shape_n=0,
            shape_k=0,
            has_bias=self.moe.has_bias,
            weight_scale_group_size=self.quant_config.weight_scale_group_size_k,
            has_dynamic_zp=self.quant_config.has_dynamic_zp,
            has_global_scale=self.quant_config.has_global_scale,
        )

        shape_n, shape_k = intermediate_size_per_partition * 2, hidden_size
        shape_n, pad_shape_n = prepare_padded_shape(shape_n, 256)
        shape_k, pad_shape_k = prepare_padded_shape(shape_k, 128)
        meta1 = dataclasses.replace(
            base_meta,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts,
            sublayer_name="w13",
        )
        self.meta1 = meta1

        shape_n, shape_k = hidden_size, intermediate_size_per_partition
        shape_n, pad_shape_n = prepare_padded_shape(shape_n, 256)
        shape_k, pad_shape_k = prepare_padded_shape(shape_k, 128)
        meta2 = dataclasses.replace(
            base_meta,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts,
            sublayer_name="w2",
        )
        self.meta2 = meta2

        HummingMethod.create_weights(layer, meta1)
        HummingMethod.create_weights(layer, meta2)
        self.process_params(layer)

    def process_params(self, layer: torch.nn.Module):
        weight_loader = self.get_weight_loader(layer)
        names = ["weight", "weight_scale", "zero_point", "global_scale", "bias"]
        unused_names = list(self.weight_converter.unused_names)

        for name in names:
            for sublayer in ["w13", "w2"]:
                param = getattr(layer, sublayer + "_" + name, None)
                if param is None:
                    continue

                set_weight_attrs(param, {"weight_loader": weight_loader})
                ckpt_name = self.weight_converter.get_ckpt_name(name)
                param.ckpt_name = ckpt_name
                param.param_name = name
                param.sublayer = sublayer
                if name == ckpt_name:
                    continue
                buffer = torch.nn.Buffer(param.data)

                setattr(layer, sublayer + "_" + ckpt_name, param)
                delattr(layer, sublayer + "_" + name)
                setattr(layer, sublayer + "_" + name, buffer)

        for name in names + unused_names:
            ckpt_name = name
            if name not in unused_names:
                ckpt_name = self.weight_converter.get_ckpt_name(name)

            for sublayer in ["w13", "w2"]:
                if hasattr(layer, sublayer + "_" + name):
                    continue

                tensor = torch.tensor([])
                param = torch.nn.Parameter(tensor, requires_grad=False)

                def empty_weight_loader(*args, **kwargs):
                    return

                set_weight_attrs(param, {"weight_loader": empty_weight_loader})
                setattr(layer, sublayer + "_" + ckpt_name, param)

    def get_weight_loader(self, layer: torch.nn.Module):
        shard_id_map = {"w1": 0, "w3": 1}

        def weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: int | None = None,
            expert_id: int | None = None,
            return_success: bool = False,
        ):
            param_name = param.param_name

            offset_n = None
            if param.sublayer == "w13":
                shape_n = self.meta1.shape_n - self.meta1.pad_shape_n
                shape_k = self.meta1.shape_k - self.meta1.pad_shape_k
                if isinstance(shard_id, str):
                    shape_n = shape_n // 2
                    shard_id = shard_id_map[shard_id]
                if isinstance(shard_id, int):
                    offset_n = shape_n * shard_id
                if not self.is_loaded_weight_narrowed:
                    shape_n = shape_n * layer.tp_size
            else:
                shape_n = self.meta2.shape_n - self.meta2.pad_shape_n
                shape_k = self.meta2.shape_k - self.meta2.pad_shape_k
                if not self.is_loaded_weight_narrowed:
                    shape_k = shape_k * layer.tp_size

            num_experts = layer.num_experts if expert_id is None else None
            data = self.weight_converter.convert(
                loaded_weight,
                param_name,
                shape_n=shape_n,
                shape_k=shape_k,
                num_experts=num_experts,
            )

            if not self.is_loaded_weight_narrowed:
                data = narrow_tensors(
                    tensors=data,
                    shape_n=shape_n,
                    shape_k=shape_k,
                    tp_size=layer.tp_size,
                    tp_rank=layer.tp_rank,
                    is_row_parallel=param.sublayer == "w2",
                )

            HummingMethod.load_weight(
                layer=layer,
                offset_n=offset_n,
                expert_id=expert_id,
                packed=True,
                sublayer_name=param.sublayer,
                **data,
            )

            return True if return_success else None

        return weight_loader

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
        HummingMethod.finish_load(
            layer,
            sublayer_name="w13",
            should_preprocess_for_glu=layer.activation == MoEActivation.SILU,
        )
        HummingMethod.finish_load(layer, sublayer_name="w2")

        use_stream_k = not vllm_is_batch_invariant()
        HummingMethod.prepare_default_kernel_configs(
            layer,
            sublayer_name="w13",
            use_stream_k=use_stream_k,
            use_f16_accum=self.use_f16_accum,
            is_moe=True,
            top_k=layer.top_k,
            is_moe_down=False,
            **self.prepare_activation_kwargs(layer),
        )
        HummingMethod.prepare_default_kernel_configs(
            layer,
            sublayer_name="w2",
            use_stream_k=use_stream_k,
            use_f16_accum=self.use_f16_accum,
            is_moe=True,
            top_k=layer.top_k,
            is_moe_down=True,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sorted_token_ids, expert_ids, num_tokens_post_padded = humming_moe_align(
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
            num_tokens_post_padded=num_tokens_post_padded,
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
            num_tokens_post_padded=num_tokens_post_padded,
            sublayer_name="w2",
        )

        return output2.sum(1)
