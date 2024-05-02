from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8DynamicToken"]


class CompressedTensorsW8A8DynamicToken(CompressedTensorsScheme):

    def __init__(self, fake_quant: bool):
        self.fake_quant = fake_quant

    
    def create_weights(self):
        pass 
    
    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor ):
        pass