# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing import Optional

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs


class Dense(nn.Module):
    """
    Dense layer for sentence-transformers models.
    
    This layer transforms embeddings from one dimension to another,
    typically used in sentence-transformers models like TencentBAC/Conan-embedding-v1.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Use ReplicatedLinear for the dense transformation
        self.linear = ReplicatedLinear(
            input_size=in_features,
            output_size=out_features,
            bias=bias,
            params_dtype=params_dtype or torch.float32,
            prefix=prefix,
            return_bias=False,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dense layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return self.linear(x)
    
    def load_weights(self, weights):
        """
        Load weights for the dense layer.
        
        Args:
            weights: Iterable of (name, tensor) pairs containing the weights
        """
        loaded_params = set()
        
        for name, weight in weights:
            if name == "dense.weight":
                self.linear.weight.data.copy_(weight)
                loaded_params.add("dense.weight")
            elif name == "dense.bias" and self.linear.bias is not None:
                self.linear.bias.data.copy_(weight)
                loaded_params.add("dense.bias")
                
        return loaded_params 