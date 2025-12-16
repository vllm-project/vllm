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
        # Extract quantized_tensors mapping: layer_name -> list of param names
        self.quantized_tensors = original_config.get("quantized_tensors", {})

    def __repr__(self) -> str:
        return "TinyQuantConfig()"
    
    def get_quantized_params(self, layer_prefix: str) -> list[str]:
        """Get list of quantized parameter names for a specific layer."""
        return self.quantized_tensors.get(layer_prefix, [])

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
        
        # Get layer prefix (set in LinearBase.__init__)
        prefix = getattr(layer, "prefix", "")
        print("-------------------")
        print("prefix", getattr(layer, "prefix", ""))
        
        # Determine param names from config
        # For merged layers (qkv_proj), check constituent layers (q_proj, k_proj, v_proj)
        param_names = self.quant_config.get_quantized_params(prefix)
        print("param_names", param_names)
        
        if not param_names and num_shards > 1:
            # Try merged layer constituents
            base_prefix = prefix.rsplit(".", 1)[0] if "." in prefix else ""
            layer_name = prefix.rsplit(".", 1)[-1] if "." in prefix else prefix
            print("base_prefix", base_prefix)
            print("layer_name", layer_name)
            
            # Map merged layer names to constituent parts
            merged_layer_map = {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }
            constituent_names = merged_layer_map.get(layer_name, [])
            print("constituent_names", constituent_names)
            
            # Try to find params from first constituent
            for constituent in constituent_names:
                constituent_prefix = f"{base_prefix}.{constituent}" if base_prefix else constituent
                print("constituent_prefix", constituent_prefix)
                param_names = self.quant_config.get_quantized_params(constituent_prefix)
                print("param_names", param_names)
                if param_names:
                    break
            print("-------------------")
        
        # Weight loader writes directly to tinyquant_layers[shard_idx].tq_tensors
        def make_weight_loader(param_name: str):
            def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
                shard_idx = get_shard_index(shard_id)
                # Write directly to the shard's tq_tensors
                layer.tinyquant_layers[shard_idx].tq_tensors[param_name] = nn.Parameter(
                    loaded_weight.cuda(), requires_grad=False
                )
            return weight_loader

        for name in param_names:
            param = nn.Parameter(torch.empty(0), requires_grad=False)
            set_weight_attrs(param, {"ignore_warning": True, "weight_loader": make_weight_loader(name)})
            tq_tensors[name] = param

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run each shard and concatenate
        print("Running TinyQuant forward pass")
        print("layer.tinyquant_layers", layer.tinyquant_layers)
        outputs = [tq_layer(x) for tq_layer in layer.tinyquant_layers]
        
        output = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
        
        if bias is not None:
            output = output + bias
        
        return output
