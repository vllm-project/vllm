from abc import abstractmethod

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs


class MoEMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) MoE methods."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_total_experts: int,
                       intermediate_size: int, hidden_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        """Create weights for a linear layer. 
           The weights will be set as attributes of the layer.
        
        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical 
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, hidden_states: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedMoEMethod(MoEMethodBase):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_total_experts: int,
                       intermediate_size: int, hidden_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        ws = nn.Parameter(
            torch.empty(num_total_experts,
                        2 * intermediate_size,
                        hidden_size,
                        dtype=params_dtype))
        w2s = nn.Parameter(
            torch.empty(num_total_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=params_dtype))
        layer.register_parameter("ws", ws)
        layer.register_parameter("w2s", w2s)
        set_weight_attrs(ws, extra_weight_attrs)
        set_weight_attrs(w2s, extra_weight_attrs)

    def apply(self, layer: torch.nn.Module, hidden_states: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        return fused_moe(hidden_states,
                         layer.ws,
                         layer.w2s,
                         router_logits,
                         layer.top_k,
                         renormalize=True,
                         inplace=True)
