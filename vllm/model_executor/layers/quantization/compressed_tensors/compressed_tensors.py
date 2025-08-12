from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme, CompressedTensorsW4A16,
    CompressedTensorsW4A16Sparse24, CompressedTensorsW8A8DynamicToken,
    CompressedTensorsW8A8StaticTensor)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    CompressionFormat, QuantizationArgs, QuantizationStrategy,
    find_first_name_or_class_match)


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str],
                 quant_format: str):
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details
        self.quant_format = quant_format

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    # Need to figure it out
    def get_min_capability(self) -> int:
        return 60

    def get_name(self) -> str:
        return "compressed_tensors"

    def get_quant_method(
            self, layer: torch.nn.Module
    ) -> Optional["CompressedTensorsLinearMethod"]:
        if isinstance(layer, LinearBase):
            return CompressedTensorsLinearMethod(self)
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        layer_quant_details: Dict[str, Any] = dict()
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
        for key, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                layer_quant_details[target] = {}
                layer_quant_details[target][
                    "weights"] = QuantizationArgs.parse_obj(
                        quant_config.get("weights"))
                try:
                    layer_quant_details[target][
                        "input_activations"] = QuantizationArgs.parse_obj(
                            quant_config.get("input_activations"))
                except Exception:
                    layer_quant_details[target]["input_activations"] = None

        return cls(layer_quant_details=layer_quant_details,
                   ignore=ignore,
                   quant_format=quant_format)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _is_static_tensor_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        is_tensor = (weight_quant.strategy == input_quant.strategy ==
                     QuantizationStrategy.TENSOR.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        is_token_tensor = (weight_quant.strategy
                           == QuantizationStrategy.TENSOR.value) and (
                               input_quant.strategy
                               == QuantizationStrategy.TOKEN.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        return is_8_bits and is_token_tensor and is_symmetric and is_dynamic

    def _is_w4a16(self, weight_quant: BaseModel,
                  input_quant: BaseModel) -> bool:
        input_quant_none = input_quant is None
        is_4_bits = weight_quant.num_bits == 4
        is_symmetric = weight_quant.symmetric
        is_static = not weight_quant.dynamic

        return is_4_bits and input_quant_none and is_symmetric and is_static

    def _get_schema(self, weight_quant: BaseModel,
                    input_quant: BaseModel) -> "CompressedTensorsScheme":

        if self._is_w4a16(weight_quant, input_quant):
            if self.quant_format == CompressionFormat.marlin_24.value:
                return CompressedTensorsW4A16Sparse24(
                    strategy=weight_quant.strategy,
                    num_bits=weight_quant.num_bits,
                    group_size=weight_quant.group_size)
            if self.quant_format == CompressionFormat.pack_quantized.value:
                return CompressedTensorsW4A16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size)

        if self.quant_format == CompressionFormat.int_quantized.value:
            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8StaticTensor()

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8DynamicToken()

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_scheme(self, layer: torch.nn.Module) -> "CompressedTensorsScheme":

        layer_type_name = find_first_name_or_class_match(
            name="",
            module=layer,
            targets=self.layer_quant_details.keys(),
            check_contains=True)

        if layer_type_name is None:
            raise ValueError(f"Could not matching target for layer {layer}")

        layer_quant_details: Dict[str, Any] = self.layer_quant_details.get(
            layer_type_name, None)
        if layer_quant_details is None:
            raise ValueError(
                f"Could not find quantization details for {layer}.")

        return self._get_schema(
            weight_quant=layer_quant_details["weights"],
            input_quant=layer_quant_details["input_activations"])


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

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

        scheme = self.quantization_config.get_scheme(layer=layer)
        scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader)

        layer.scheme = scheme

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme 
        associated with the layer to apply the forward pass with the 
        layer input.  See LinearMethodBase for param details

        """

        if bias is not None:
            raise ValueError("bias is not supported for this linear method")

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x)
