# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Dict, Any

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

import torch.nn.functional as F

__all__ = ["QuarkW8A8Fp8"]

OCP_MX_BLOCK_SIZE = 32

def unpack_and_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    # Unpack the 4-bit values from each byte
    tensor = tensor.reshape(-1, tensor.shape[-1])
    unpacked = torch.zeros(tensor.shape[0], tensor.shape[1] * 2, dtype=torch.uint8, device=tensor.device)
    unpacked[:, ::2] = (tensor >> 4) & 0x0F  # Extract high 4 bits
    unpacked[:, 1::2] = tensor & 0x0F  # Extract low 4 bits

    # Convert back to int32 and restore the original scaling
    unpacked = unpacked.to(torch.int32)
    unpacked = ((unpacked & 0x07) << 22) | ((unpacked & 0x08) << 28)
    unpacked = unpacked.view(torch.float32) * (2.0**(127 - 1))

    return unpacked



class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, input_quant_spec: Dict[str, Any]):
        try:
            from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
            from quark.torch.quantization.config.config import QuantizationSpec
        except ImportError as err:
            raise ImportError(f"The package `amd-quark` is required to use AMD Quark MX-FP4 models. Please install it with `pip install amd-quark`. Error: {err}")

        self.out_dtype = torch.get_default_dtype()

        self.qscheme = "per_group"

        input_quant_spec = QuantizationSpec.from_dict(input_quant_spec)

        self.input_quantizer = ScaledFakeQuantize(
            quant_obj="activation",
            quant_spec=input_quant_spec
        )
        self.input_quantizer.enable_observer()
        self.input_quantizer.enable_fake_quant()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader
        )
        layer.register_parameter("weight", weight)

        print(f"set weight {weight.data.shape}")

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        input_dtype = x.dtype

        # View input as 2D matrix for fp8 methods
        # input_2d = input.view(-1, input.shape[-1])
        output_shape = [*x.shape[:-1], layer.weight.shape[0]]

        print("x", x.shape, x.dtype)

        # dequantize(quantize(x))
        qdq_x = self.input_quantizer(x.to(torch.float32))

        print("qdq_x", qdq_x.shape, qdq_x.dtype)
        
        dq_weight = unpack_and_dequantize(layer.weight)
        print("dq_weight", dq_weight.shape, dq_weight.dtype)

        print("layer.weight_scale", layer.weight_scale.shape, layer.weight_scale.dtype)

        dq_weight = dq_weight.reshape(dq_weight.shape[0], OCP_MX_BLOCK_SIZE, -1) * layer.weight_scale[:, None, :]
        print("dq_weight", dq_weight.shape, dq_weight.dtype)

        dq_weight = dq_weight.reshape(layer.weight.shape[0], -1)

        print("dq_weight", dq_weight.shape, dq_weight.dtype)

        output = F.linear(qdq_x, dq_weight, bias)
        print("output", output.shape, output.dtype)

        # TODO: handle output quantization to fp8 for KV.

        return output.to(dtype=input_dtype).view(*output_shape)
