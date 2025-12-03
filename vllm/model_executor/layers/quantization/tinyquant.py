# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from tinyquant.quantized_linear import QuantizedLinear

logger = init_logger(__name__)


# shard_id -> index mapping
SHARD_ID_MAP = {"q": 0, "k": 1, "v": 2}


def get_shard_index(shard_id) -> int:
    if shard_id is None:
        return 0
    if isinstance(shard_id, str):
        return SHARD_ID_MAP.get(shard_id, 0)
    return int(shard_id)


class TinyQuantConfig(QuantizationConfig):
    """Config class for TinyQuant quantization method."""

    def __init__(self, original_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.original_config = original_config or {}

    def __repr__(self) -> str:
        return "TinyQuantConfig()"

    @classmethod
    def get_name(cls) -> str:
        return "tinyquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json", "quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TinyQuantConfig":        
        return cls(original_config=config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.model_executor.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return TinyQuantLinearMethod(self)
        return None


class TinyQuantLinearMethod(LinearMethodBase):
    """Linear method for TinyQuant. Supports merged layers via sharding."""
    
    def __init__(self, quant_config: TinyQuantConfig):
        self.quant_config = quant_config
    
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
        num_shards = len(output_partition_sizes)
        
        # Create QuantizedLinear for each shard
        layer.tinyquant_layers = [QuantizedLinear.empty() for _ in range(num_shards)]
        
        # Placeholder tq_tensors for vLLM weight loading
        tq_tensors = nn.ParameterDict()
        layer.register_module("tq_tensors", tq_tensors)
        
        # Weight loader writes directly to tinyquant_layers[shard_idx].tq_tensors
        def make_weight_loader(param_name: str):
            def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
                shard_idx = get_shard_index(shard_id)
                # Write directly to the shard's tq_tensors
                # Must wrap in Parameter with requires_grad=False (uint8 doesn't support grads)
                layer.tinyquant_layers[shard_idx].tq_tensors[param_name] = nn.Parameter(
                    loaded_weight.cuda(), requires_grad=False
                )
            return weight_loader
        
        # Register placeholder parameters
        for name, dtype in [("meta", torch.uint8), ("absmax", torch.float32), ("quantized_weight", torch.uint8)]:
            param = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
            set_weight_attrs(param, {"ignore_warning": True, "weight_loader": make_weight_loader(name)})
            tq_tensors[name] = param
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run each shard and concatenate
        outputs = [tq_layer(x) for tq_layer in layer.tinyquant_layers]
        
        output = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
        
        if bias is not None:
            output = output + bias
        
        return output
