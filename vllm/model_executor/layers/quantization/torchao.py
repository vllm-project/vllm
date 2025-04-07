# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao."""

    def __init__(self, torchao_config) -> None:
        self.torchao_config = torchao_config

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
        try:
            from torchao.core.config import config_from_dict
        except ImportError as err:
            raise ImportError(
                "Please install torchao>=0.10.0 via "
                "`pip install torchao>=0.10.0` to use torchao quantization."
            ) from err

        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        assert hf_config is not None, "quant_type must be specified"
        assert (len(hf_config) == 1 and "default" in hf_config
                ), "Expected only one key 'default' in quant_type dictionary"
        quant_type = hf_config["default"]
        ao_config = config_from_dict(quant_type)
        return cls(ao_config)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["TorchAOLinearMethod"]:
        if isinstance(layer, LinearBase):
            return TorchAOLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def torchao_quantize_param_data(param: torch.Tensor,
                                torchao_config: Any) -> torch.nn.Parameter:
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to
        use to quantize the Tensor
    """
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_
    assert isinstance(torchao_config, AOBaseConfig)
    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    quantize_(dummy_linear, torchao_config)
    return dummy_linear.weight


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
        weight = torchao_quantize_param_data(weight,
                                             self.quant_config.torchao_config)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
