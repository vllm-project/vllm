import fnmatch
import re
from typing import Any, Dict, List, Optional, cast

import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
    QuarkMoEMethod)
from vllm.model_executor.layers.quantization.quark.schemes import (
    QuarkScheme, QuarkW8A8Fp8, QuarkW8A8Int8)
from vllm.model_executor.layers.quantization.quark.utils import (
    deep_compare, should_ignore_layer)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FUSED_LAYER_NAME_MAPPING)
from vllm.platforms import current_platform

__all__ = ["QuarkLinearMethod"]


class QuarkConfig(QuantizationConfig):

    def __init__(self,
                 quant_config: Dict[str, Any],
                 kv_cache_group: Optional[List[str]] = None,
                 kv_cache_config: Optional[Dict[str, Any]] = None,
                 pack_method: str = "reorder"):
        if kv_cache_group is None:
            kv_cache_group = []
        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        # Check if the layer is skipped for quantization.
        exclude_layers = cast(List[str], self.quant_config.get("exclude"))
        if should_ignore_layer(prefix, ignore=exclude_layers):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return QuarkLinearMethod(self)
        if isinstance(layer, Attention):
            return QuarkKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return QuarkMoEMethod.get_moe_method(self,
                                                 module=layer,
                                                 layer_name=prefix)
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":
        export_config = config.get("export")
        if export_config is None:
            raise ValueError("The export key should be included in "
                             "the configurations of Quark quantized model")
        kv_cache_group = cast(List[str], export_config.get("kv_cache_group"))
        pack_method = cast(str, export_config.get("pack_method"))

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(Dict[str, Any],
                                      config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not kv_cache_set.issubset(layer_quant_set):
                raise ValueError("The Quark quantized model has the "
                                 "kv_cache_group parameter setting, "
                                 "but no kv_cache quantization settings "
                                 "were found in the quantization "
                                 "configuration.")

            q_configs = [
                cast(Dict[str, Any], layer_quant_config.get(name))
                for name in kv_cache_group
            ]
            if not all(
                    deep_compare(q_config, q_configs[0])
                    for q_config in q_configs):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different.")
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError(
                    "The kv_cache quantization configuration is empty.")

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            for q_config in q_configs:
                q_config["output_tensors"] = None

        return cls(quant_config=config,
                   kv_cache_group=kv_cache_group,
                   kv_cache_config=kv_cache_config,
                   pack_method=pack_method)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _check_scheme_supported(self,
                                min_capability: int,
                                error: bool = True) -> bool:
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.")
            return supported
        else:
            return False

    def _is_fp8_w8a8(self, weight_quant: Optional[Dict[str, Any]],
                     input_quant: Optional[Dict[str, Any]]) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported
        is_fp8_dtype = (weight_quant.get("dtype") == "fp8_e4m3"
                        and input_quant.get("dtype") == "fp8_e4m3")
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = (weight_quant.get("qscheme")
                                           in ["per_tensor", "per_channel"])

        if not (is_fp8_dtype and is_static_weight
                and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.get("is_dynamic"):
            return True

        # Confirm activation scheme is supported.
        is_per_tensor_activation = (input_quant.get("qscheme") == "per_tensor")
        return is_per_tensor_activation

    def _is_static_tensor_w8a8(self, weight_quant: Optional[Dict[str, Any]],
                               input_quant: Optional[Dict[str, Any]]) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return False

        is_int8_dtype = (weight_quant.get("dtype") == "int8"
                         and input_quant.get("dtype") == "int8")

        is_tensor = (weight_quant.get("qscheme")
                     in ["per_tensor", "per_channel"]
                     and input_quant.get("qscheme") == "per_tensor")

        is_static = (not weight_quant.get("is_dynamic")
                     and not input_quant.get("is_dynamic"))

        is_weight_symmetric = (weight_quant.get("symmetric") is True)

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_int8_dtype and is_tensor and is_weight_symmetric and is_static

    def _find_matched_config(self, layer_name: str,
                             module: torch.nn.Module) -> Dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in FUSED_LAYER_NAME_MAPPING:
            shard_proj_names = FUSED_LAYER_NAME_MAPPING[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                    deep_compare(q_config, shard_configs[0])
                    for q_config in shard_configs):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme.")
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                Dict[str, Any], self.quant_config.get("layer_quant_config"))
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = cast(str, type(module))
            layer_type_quant_config = cast(
                Dict[str, Any],
                self.quant_config.get("layer_type_quant_config"))
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                Dict[str, Any], self.quant_config.get("global_quant_config"))
            return global_quant_config

    def _get_scheme_from_config(self, config: Dict[str, Any]) -> "QuarkScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported")
        weight_config = cast(Dict[str, Any], config.get("weight"))
        input_config = cast(Dict[str, Any], config.get("input_tensors"))

        if self._is_fp8_w8a8(weight_config, input_config):
            is_fp8_w8a8_supported = self._check_scheme_supported(
                QuarkW8A8Fp8.get_min_capability(), error=False)
            if is_fp8_w8a8_supported:
                weight_qscheme = cast(str, weight_config.get("qscheme"))
                input_static = (input_config is not None and
                                not cast(bool, input_config.get("is_dynamic")))
                return QuarkW8A8Fp8(qscheme=weight_qscheme,
                                    is_static_input_scheme=input_static)
        elif self._is_static_tensor_w8a8(weight_config, input_config):
            weight_qscheme = cast(str, weight_config.get("qscheme"))
            return QuarkW8A8Int8(qscheme=weight_qscheme,
                                 is_static_input_scheme=True,
                                 input_symmetric=input_config.get("symmetric"))

        raise NotImplementedError("No quark compatible scheme was found. "
                                  f"Weight config: {weight_config}, "
                                  f"Input config: {input_config}")

    def get_scheme(self, layer: torch.nn.Module,
                   layer_name: str) -> "QuarkScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)
        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in quark. If this is the case, return its equivalent param name 
        expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if self.kv_cache_group is None or len(self.kv_cache_group) == 0:
            return None

        kv_proj_names = [
            re.split(r"[*.]", kv_cache)[-1] for kv_cache in self.kv_cache_group
        ]
        if name.endswith(".output_scale"):
            if len(kv_proj_names) == 1 and kv_proj_names[0] in name:
                kv_output_scale_name = "." + kv_proj_names[0] + ".output_scale"
                return name.replace(kv_output_scale_name, ".attn.k_scale")

            elif len(kv_proj_names) == 2:
                for kv_proj_name in kv_proj_names:
                    if kv_proj_name in name and kv_proj_name == "k_proj":
                        return name.replace(".k_proj.output_scale",
                                            ".attn.k_scale")
                    elif kv_proj_name in name and kv_proj_name == "v_proj":
                        return name.replace(".v_proj.output_scale",
                                            ".attn.v_scale")

        # If no matches, return None
        return None


class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class QuarkKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: QuarkConfig):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[Dict[str, Any]]):
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_config: the quark kv cache scheme
        """
        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}")

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}")
