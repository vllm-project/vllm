from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    W4A16SPARSE24_SUPPORTED_BITS, WNA16_SUPPORTED_BITS,
    CompressedTensorsScheme, CompressedTensorsW4A16Sparse24,
    CompressedTensorsW8A8Fp8, CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8, CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    CompressionFormat, QuantizationArgs, QuantizationStrategy,
    QuantizationType, find_matched_target, is_activation_quantization_format,
    should_ignore_layer)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsLinearMethod"]


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self,
                 target_scheme_map: Dict[str, Any],
                 ignore: List[str],
                 quant_format: str,
                 kv_cache_scheme: Optional[Dict[str, Any]] = None):

        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "compressed_tensors"

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        # Check if the layer is skipped for quantization.
        # TODO (@robertgshaw2): support module names
        if should_ignore_layer(prefix, ignore=self.ignore):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod(self)
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        target_scheme_map: Dict[str, Any] = dict()
        ignore: List[str] = config.get("ignore", None)
        quant_format: str = config.get("format", None)

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.
        for _, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.parse_obj(
                        quant_config.get("weights"))
                try:
                    target_scheme_map[target][
                        "input_activations"] = QuantizationArgs.parse_obj(
                            quant_config.get("input_activations"))
                except Exception:
                    target_scheme_map[target]["input_activations"] = None

        return cls(target_scheme_map=target_scheme_map,
                   ignore=ignore,
                   quant_format=quant_format,
                   kv_cache_scheme=config.get("kv_cache_scheme"))

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _check_scheme_supported(self,
                                min_capability: int,
                                error: bool = True) -> bool:
        capability = current_platform.get_device_capability()  # type: ignore

        if capability is not None:
            capability = capability[0] * 10 + capability[1]
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.")
            return supported
        else:
            return False

    def _is_static_tensor_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_tensor = (weight_strategy and input_quant.strategy
                     == QuantizationStrategy.TENSOR.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        return is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_fp8_w8a8(self, weight_quant: BaseModel,
                     input_quant: BaseModel) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported.
        is_floating_point = (weight_quant.type == QuantizationType.FLOAT
                             and input_quant.type == QuantizationType.FLOAT)
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = (weight_quant.strategy in [
            QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL
        ])
        if not (is_floating_point and is_symmetric_weight and is_static_weight
                and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.dynamic:
            return True

        # Confirm activation scheme is supported.
        is_symmetric_activation = input_quant.symmetric
        is_per_tensor_activation = (
            input_quant.strategy == QuantizationStrategy.TENSOR)
        return is_symmetric_activation and is_per_tensor_activation

    def _is_fp8_w8a16(self, weight_quant: BaseModel,
                      input_quant: BaseModel) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have floating points.
        if weight_quant.type != QuantizationType.FLOAT:
            return False

        # Confirm weight scheme is supported.
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = (weight_quant.strategy in [
            QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL
        ])
        if not (is_symmetric_weight and is_static_weight
                and is_per_tensor_or_channel_weight):
            return False

        # All conditions satisfied.
        return True

    def _is_wNa16_group_channel(self, weight_quant: BaseModel,
                                input_quant: BaseModel) -> bool:
        input_quant_none = input_quant is None
        is_symmetric = weight_quant.symmetric
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_static = not weight_quant.dynamic

        return (is_channel_group and input_quant_none and is_symmetric
                and is_static)

    def _get_scheme_from_parts(
            self, weight_quant: BaseModel,
            input_quant: BaseModel) -> "CompressedTensorsScheme":

        # Detect If Mixed Precision
        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (self.quant_format == CompressionFormat.marlin_24.value
                    and weight_quant.num_bits in W4A16SPARSE24_SUPPORTED_BITS):
                return CompressedTensorsW4A16Sparse24(
                    strategy=weight_quant.strategy,
                    num_bits=weight_quant.num_bits,
                    group_size=weight_quant.group_size)
            if (self.quant_format == CompressionFormat.pack_quantized.value
                    and weight_quant.num_bits in WNA16_SUPPORTED_BITS):
                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder)

        # Detect If Activation Quantization.
        # TODO @dsikka: clean-up conditions
        if is_activation_quantization_format(self.quant_format):
            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False)
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=(input_quant
                                                and not input_quant.dynamic))
                else:
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=(input_quant
                                                and not input_quant.dynamic))

            if self._is_fp8_w8a16(weight_quant, input_quant):
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=(input_quant
                                            and not input_quant.dynamic))

            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=True)

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False)

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":
        """
        compressed-tensors supports non uniform in the following way:

        ignore: List of layer_names or nn.Module names to be ignored.
        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        We first check whether a layer is in the ignore group and use
        CompressedTensorsUnquantized (i.e. fp16/bf16) scheme for the layer

        We then detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # TODO (@robertgshaw): add compressed-tensors as dep
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this
        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys())

        # Find the quant_scheme
        scheme_dict = self.target_scheme_map[matched_target]
        scheme = self._get_scheme_from_parts(
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"])

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
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


class CompressedTensorsKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from compressed-tensors
    checkpoints.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.validate_kv_cache_scheme(quant_config.kv_cache_scheme)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_scheme(kv_cache_scheme: Optional[Dict[str, Any]]):
        """
        Validator for the kv cache scheme. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_scheme: the compressed-tensors kv cache scheme
        """
        if kv_cache_scheme is None:
            return

        type_ = kv_cache_scheme.get("type")
        num_bits = kv_cache_scheme.get("num_bits")

        if type_ != "float" and num_bits != 8:
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                "num_bits=8, type=float, however "
                f"received num_bits={num_bits}, type={type_}")

        strategy = kv_cache_scheme.get("strategy")
        if strategy != "tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for compressed-tensors KV cache. "
                f"Expected strategy: tensor, found strategy: {strategy}")

        is_symmetric = kv_cache_scheme.get("symmetric")
        if not is_symmetric:
            raise NotImplementedError(
                "Only support symmetric scaling factor "
                "for compressed-tensors KV cache. "
                f"However found symmetric: {is_symmetric}")
