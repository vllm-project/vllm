# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import math
from typing import Any

import torch
from humming import dtypes
from humming.layer import HummingLayerMeta, HummingMethod

from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
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
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    WEIGHT_CONVERTER_MAP,
)
from vllm.model_executor.utils import set_weight_attrs


def get_ignored_layers(config):
    if "ignored_layers" in config:
        return config["ignored_layers"]
    if "modules_to_not_convert" in config:
        return config["modules_to_not_convert"]
    return []


def is_layer_skipped(prefix: str, ignore_layers: list[str]):
    return any(module_name in prefix for module_name in ignore_layers)


class HummingConfig(QuantizationConfig):
    def __init__(
        self,
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        bs_dtype: str,
        ckpt_quant_method: str | None,
        input_scale_group_size_n: int,
        input_scale_group_size_k: int,
        weight_scale_group_size_n: int,
        weight_scale_group_size_k: int,
        has_dynamic_zp: bool,
        has_global_scale: bool,
        ignored_layers: list[str] | None,
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.bs_dtype = bs_dtype
        self.ckpt_quant_method = ckpt_quant_method
        self.full_config = full_config
        self.input_scale_group_size_n = input_scale_group_size_n
        self.input_scale_group_size_k = input_scale_group_size_k
        self.weight_scale_group_size_n = weight_scale_group_size_n
        self.weight_scale_group_size_k = weight_scale_group_size_k
        self.has_dynamic_zp = has_dynamic_zp
        self.has_global_scale = has_global_scale
        self.ignored_layers = ignored_layers or []

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
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HummingConfig":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        ignored_layers = get_ignored_layers(config)

        has_dynamic_zp = False
        has_global_scale = False
        if quant_method == "gptq":
            has_dynamic_zp = not cls.get_from_keys(config, ["sym"])
            desc_act = config.get("desc_act", False)
            assert not desc_act, "desc_act is not supported by humming"
            block_shape = (1, config["group_size"])
            b_dtype = dtypes.DataType.from_str("uint" + str(config["bits"]))
            bs_dype = None
        elif quant_method == "awq":
            has_dynamic_zp = cls.get_from_keys(config, ["zero_point"])
            block_shape = (1, config["group_size"])
            b_dtype = dtypes.uint4
            bs_dype = None
        elif quant_method == "fp8":
            block_shape = cls.get_from_keys_or(config, ["weight_block_size"], (0, 0))
            b_dtype = dtypes.float8e4m3
            bs_dype = None
        elif quant_method == "modelopt":
            block_shape = (1, 16)
            b_dtype = dtypes.float4e2m1
            bs_dype = dtypes.float8e4m3
            has_global_scale = True
        elif quant_method == "mxfp4":
            block_shape = (1, 32)
            b_dtype = dtypes.float4e2m1
            bs_dype = dtypes.float8e8m0
        else:
            raise AssertionError(f"Invalid quant_method: {quant_method}")

        return cls(
            a_dtype=None,
            b_dtype=b_dtype,
            c_dtype=None,
            bs_dtype=bs_dype,
            ckpt_quant_method=quant_method,
            input_scale_group_size_n=0,
            input_scale_group_size_k=0,
            weight_scale_group_size_n=block_shape[0],
            weight_scale_group_size_k=block_shape[1],
            has_dynamic_zp=has_dynamic_zp,
            has_global_scale=has_global_scale,
            ignored_layers=ignored_layers,
            full_config=config.copy(),
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        if user_quant == "humming":
            return cls.get_name()
        return None
    
    def is_layer_skipped(self, layer: torch.nn.Module, prefix: str):
        ignored_layers = self.ignored_layers
        if any(module_name in prefix for module_name in ignored_layers):
            return True
        quant_method = self.full_config.get("quant_method", None)

        return isinstance(layer, LinearBase) and quant_method == "mxfp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if self.is_layer_skipped(layer, prefix):
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            return HummingLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return HummingMoEMethod(self, layer.moe_config)
        return None


class HummingLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: HummingConfig):
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
        layer.weight_block_size = None

        meta = HummingLayerMeta(
            a_dtype=self.quant_config.a_dtype or f16_dtype,
            b_dtype=self.quant_config.b_dtype,
            c_dtype=self.quant_config.c_dtype or f16_dtype,
            bs_dtype=self.quant_config.bs_dtype or f16_dtype,
            shape_n=output_size,
            shape_k=input_size,
            has_bias=layer.has_bias,
            weight_scale_group_size=self.quant_config.weight_scale_group_size_k,
            has_dynamic_zp=self.quant_config.has_dynamic_zp,
            has_global_scale=self.quant_config.has_global_scale,
        )
        HummingMethod.create_weights(layer, meta)
        self.process_params(layer)

    def process_params(self, layer: torch.nn.Module):
        weight_loader = self.get_weight_loader(layer)
        # bias is inited and loaded outside HummingLinearMethod
        names = ["weight", "weight_scale", "zero_point", "global_scale"]
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
            shard_id: int | None = None,
        ):
            param_name = param.param_name
            data = self.weight_converter.convert(loaded_weight, param_name)
            shard_id = shard_id_map.get(shard_id, shard_id or 0)
            offset_n = sum(layer.logical_widths[:shard_id])
            HummingMethod.load_weight(
                layer=layer,
                offset_n=offset_n,
                packed=True,
                **data,
            )

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        HummingMethod.finish_load(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_stream_k = not vllm_is_batch_invariant()

        # TODO: config tunning
        block_size_m = math.ceil(x.size(0) / 16) * 16
        block_size_m = min(block_size_m, 64)
        block_shape = (block_size_m, 128, 64)
        warp_shape = (block_size_m, 64, 32)

        import os

        if not os.path.exists("/tmp/aa.pt"):
            torch.save(
                (
                    layer.weight.data,
                    layer.weight_scale.data,
                    layer.zero_point.data,
                ),
                "/tmp/aa.pt",
            )

        return HummingMethod.forward_layer(
            layer=layer,
            block_shape=block_shape,
            warp_shape=warp_shape,
            inputs=x,
            use_stream_k=use_stream_k,
        )


class HummingMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: HummingConfig, moe: "FusedMoEConfig") -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.moe = moe
        ckpt_quant_method = self.quant_config.ckpt_quant_method
        weight_converter_cls = WEIGHT_CONVERTER_MAP[ckpt_quant_method]
        self.weight_converter = weight_converter_cls(quant_config)

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

        dtype = dtypes.DataType.from_torch_dtype(params_dtype)
        base_meta = HummingLayerMeta(
            a_dtype=self.quant_config.a_dtype or dtype,
            b_dtype=self.quant_config.b_dtype,
            c_dtype=self.quant_config.c_dtype or dtype,
            bs_dtype=self.quant_config.bs_dtype or dtype,
            shape_n=0,
            shape_k=0,
            has_bias=self.moe.has_bias,
            weight_scale_group_size=self.quant_config.weight_scale_group_size_k,
            has_dynamic_zp=self.quant_config.has_dynamic_zp,
            has_global_scale=self.quant_config.has_global_scale,
        )

        meta1 = dataclasses.replace(
            base_meta,
            shape_n=intermediate_size_per_partition * 2,
            shape_k=hidden_size,
            num_experts=num_experts,
            sublayer_name="w13",
        )
        self.meta1 = meta1

        meta2 = dataclasses.replace(
            base_meta,
            shape_n=hidden_size,
            shape_k=intermediate_size_per_partition,
            num_experts=num_experts,
            sublayer_name="w2",
        )
        self.meta2 = meta2

        HummingMethod.create_weights(layer, meta1)
        HummingMethod.create_weights(layer, meta2)
        self.process_params(layer)

    def process_params(self, layer: torch.nn.Module):
        weight_loader = self.get_weight_loader(layer)
        # unlink HummingLinearMethod,
        # the bias os FusedMoE is manager by HummingMoEMethod
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
        ):
            param_name = param.param_name
            data = self.weight_converter.convert(loaded_weight, param_name)

            offset_n = None
            if param.sublayer == "w13":
                shard_id = shard_id_map.get(shard_id, shard_id or 0)
                offset_n = self.meta1.shape_n // 2 * shard_id

            HummingMethod.load_weight(
                layer=layer,
                offset_n=offset_n,
                expert_id=expert_id,
                packed=True,
                sublayer_name=param.sublayer,
                **data,
            )

        return weight_loader

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        HummingMethod.finish_load(layer, "w13")
        HummingMethod.finish_load(layer, "w2")

    def prepare_buffer(
        self,
        layer: FusedMoE,
        output_shape_m: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        output_shape1 = (output_shape_m, layer.humming_metas["w13"].shape_n)
        output_shape2 = (output_shape_m, layer.humming_metas["w2"].shape_n)
        buffer_size1 = output_shape1[0] * output_shape1[1]
        buffer_size2 = output_shape2[0] * output_shape2[1]

        buffer_size = max(buffer_size1, buffer_size2) + buffer_size1 // 2
        buffer = torch.empty(buffer_size, dtype=dtype, device=device)
        output1 = buffer[buffer_size1 // 2 :][:buffer_size1].view(*output_shape1)
        input2 = buffer[:buffer_size1 // 2].view(output_shape_m, -1)
        output2 = buffer[buffer_size1 // 2 :][:buffer_size2].view(*output_shape2)

        return output1, input2, output2

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO: config tunning
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=32,
            num_experts=layer.num_experts,
            expert_map=layer.expert_map,
            ignore_invalid_experts=True,
        )

        output_shape_m = x.size(0) * topk_ids.size(-1)
        output1, input2, output2 = self.prepare_buffer(
            layer,
            output_shape_m,
            x.dtype,
            x.device,
        )

        block_shape = (32, 64, 64)
        warp_shape = (16, 64, 32)

        output1 = HummingMethod.forward_layer(
            layer=layer,
            block_shape=block_shape,
            warp_shape=warp_shape,
            inputs=x,
            outputs=output1,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            sublayer_name="w13",
            top_k=topk_ids.size(-1),
            is_moe_down=False,
        )

        from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation

        apply_moe_activation(layer.activation, input2, output1)
        # torch.ops._C.silu_and_mul(input2, output1)

        output2 = HummingMethod.forward_layer(
            layer=layer,
            block_shape=block_shape,
            warp_shape=warp_shape,
            inputs=input2,
            outputs=None,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            sublayer_name="w2",
            is_moe_down=True,
            top_k=topk_ids.size(-1),
        )

        return output2.view(x.size(0), topk_ids.size(-1), -1).sum(1)
