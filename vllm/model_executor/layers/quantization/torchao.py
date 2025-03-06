# SPDX-License-Identifier: Apache-2.0
import importlib.metadata
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from packaging import version
from torch.nn import Module
from torch.nn.parameter import Parameter
from torchao.core.config import AOBaseConfig

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.torchao_utils import (
    torchao_quantize_param_data)
from vllm.model_executor.utils import set_weight_attrs


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    torchao_config: AOBaseConfig
    runtime_quant: bool

    def __init__(self, torchao_config: AOBaseConfig,
                 runtime_quant: bool) -> None:
        self.torchao_config = torchao_config

        # Determine based on torchao version
        self.runtime_quant = runtime_quant
        if not self.runtime_quant:
            torchao_version = version.parse(
                importlib.metadata.version("torchao"))
            assert torchao_version >= version.parse(
                "0.9.0"
            ), "torchao version must be >= 0.9.0 to load quantized models"
            self.quantize_after_loading = torchao_version <= version.parse(
                "0.9.0")

    def __repr__(self) -> str:
        return f"TorchAOConfig({self.torchao_config})"

    def get_name(self) -> str:
        return "torchao"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TorchAOConfig":
        """Create the quant config from an hf model config"""
        from torchao.core.config import config_from_dict
        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        runtime_quant = cls.get_from_keys_or(config, ["runtime_quant"], False)
        assert hf_config is not None, "quant_type must be specified"
        assert (len(hf_config) == 1 and "default" in hf_config
                ), "Expected only one key 'default' in quant_type dictionary"
        quant_type = hf_config["default"]
        ao_config = config_from_dict(quant_type)
        return cls(ao_config, runtime_quant)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["TorchAOLinearMethod"]:
        if isinstance(layer, LinearBase):
            return TorchAOLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class TorchAOLinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        torchao_config: The torchao quantization config, a string
        that encodes the type of quantization and all relevant arguments.
    """

    def __init__(self, quant_config: TorchAOConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # we load quantized model in versions after 0.9.0 and do on the fly
        # quantization in torchao versions before 0.9.0
        if not self.quant_config.runtime_quant:
            weight = torchao_quantize_param_data(
                weight, self.quant_config.torchao_config)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        # We aren't loading a quantized weight, we quantize on the fly
        if self.quant_config.runtime_quant:
            torchao_config = self.quant_config.torchao_config
            layer.weight = torchao_quantize_param_data(layer.weight,
                                                       torchao_config)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
