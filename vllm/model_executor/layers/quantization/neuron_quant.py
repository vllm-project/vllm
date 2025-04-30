# SPDX-License-Identifier: Apache-2.0

import os
from importlib.util import find_spec
from typing import Any, Dict, List, Optional

from torch.nn import Module

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

SUPPORTED_QUANT_DTYPE_LIST = ['s8', 'f8e4m3fn']


class NeuronQuantConfig(QuantizationConfig):
    """Int8 Quantization Config class for Neuron Backend."""

    def __init__(
        self,
        dequant_dtype: str = "f16",
        quantize_method: str = "vector_dynamic",
    ) -> None:
        super().__init__()
        self.quant_dtype = os.getenv("NEURON_QUANT_DTYPE", "s8")
        if self.quant_dtype not in SUPPORTED_QUANT_DTYPE_LIST:
            raise ValueError(
                f"Neuron quantization datatype {self.quant_dtype} is not valid,"
                f" the quantization datatype should match one of the below "
                f"types {SUPPORTED_QUANT_DTYPE_LIST}")
        self.dequant_dtype = dequant_dtype
        self.quantize_method = quantize_method

    def get_name(self) -> QuantizationMethods:
        return "neuron_quant"

    def get_supported_act_dtypes(self) -> List[str]:
        return SUPPORTED_QUANT_DTYPE_LIST

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "This function should not be called with Neuron Backend")

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NeuronQuantConfig":
        quantize_method = cls.get_from_keys(config, ["quantize_method"])
        dequant_dtype = cls.get_from_keys(config, ["dequant_dtype"])
        return cls(dequant_dtype=dequant_dtype,
                   quantize_method=quantize_method)

    def get_quant_method(self, layer: Module, prefix: str) -> Optional[Any]:
        if find_spec("transformers_neuronx") is not None:
            return self.get_quantization_config()
        else:
            raise NotImplementedError(
                "Neuron Quantization is only supported through"
                " transformers_neuronx.")

    def get_quantization_config(self):
        from transformers_neuronx.config import QuantizationConfig
        return QuantizationConfig(quant_dtype=self.quant_dtype,
                                  dequant_dtype=self.dequant_dtype,
                                  quantize_method=self.quantize_method)
