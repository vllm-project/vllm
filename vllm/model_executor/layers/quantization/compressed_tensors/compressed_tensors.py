from typing import Any, Dict, List, Optional

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme, CompressedTensorsW8A8StaticTensor)


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str]):
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16]

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

        for key, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                layer_quant_details[target] = {}
                layer_quant_details[target]["weight"] = quant_config.get(
                    "weights")
                layer_quant_details[target]["input"] = quant_config.get(
                    "input_activations")

        return cls(layer_quant_details=layer_quant_details, ignore=ignore)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _get_schema(self, weight_quant: Dict, input_quant: Dict):
        # TODO: Refactor as additional cases are supported

        weight_bit = weight_quant.get("num_bits")
        input_bit = input_quant.get("num_bits")

        weight_strategy = weight_quant.get("strategy")
        input_strategy = input_quant.get("strategy")

        weight_symmetric = weight_quant.get("symmetric")
        input_symmetric = input_quant.get("symmetric")

        is_8_bits = weight_bit == input_bit == 8
        is_tensor = weight_strategy == input_strategy == "tensor"
        is_symmetric = weight_symmetric and input_symmetric

        if is_8_bits and is_tensor and is_symmetric and \
                torch.cuda.is_available():
            # CompressedTensorsW8A8StaticTensor only supports CUDA path for
            # now.
            return CompressedTensorsW8A8StaticTensor()
        raise NotImplementedError(
            "Scheme not supported. Only CUDA, 8-bit static symmtetric "
            "per tensor quantization is currently supported")

    def get_scheme(self, layer: torch.nn.Module) -> "CompressedTensorsScheme":

        # TODO: update with matching function from `compressed_tensors`
        layer_type_name = None
        layer_name_class = type(layer).__name__.lower()
        for target in self.layer_quant_details:
            if target.lower() in layer_name_class:
                layer_type_name = target
                break
        if layer_type_name is None:
            raise ValueError(f"Could not matching target for layer {layer}")

        layer_quant_details: Dict[str, Any] = self.layer_quant_details.get(
            layer_type_name, None)
        if layer_quant_details is None:
            raise ValueError(
                f"Could not find quantization details for {layer}.")

        return self._get_schema(weight_quant=layer_quant_details["weight"],
                                input_quant=layer_quant_details["input"])


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
        the necessary parameters for the layer.
        """
        weight_loader = extra_weight_attrs.get("weight_loader")

        scheme = self.quantization_config.get_scheme(layer=layer)
        scheme.create_weights(
            layer=layer,
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
        layer input.
        """

        if bias is not None:
            raise ValueError("bias is not supported for this linear method")

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x)
