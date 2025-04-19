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

class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, weight_quant_spec: Dict[str, Any], input_quant_spec: Dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                           requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                          requires_grad=False)

        # TODO(bowenbao): perform emulation only when native mx kernel is unsupported.
        try:
            from quark.torch.export.nn.modules import realquantizer
            from quark.torch.quantization.config.config import QuantizationSpec
        except ImportError as err:
            raise ImportError(
                f"The package `amd-quark` is required to use AMD Quark MX-FP4 models. Please install it with `pip install amd-quark`. Error: {err}"
            )

        weight_quant_spec = QuantizationSpec.from_dict(self.weight_quant_spec)
        input_quant_spec = QuantizationSpec.from_dict(self.input_quant_spec)

        self.weight_quantizer = realquantizer.get_real_quantizer(
            qspec=weight_quant_spec,
            quantizer=None,
            real_quantized=True,
            reorder=False,  # TODO: load from config
            float_dtype=self.out_dtype,
            scale_shape=layer.weight_scale.shape,
            zero_point_shape=None,
        )
        self.weight_quantizer.scale.data = layer.weight_scale.data

        self.input_quantizer = realquantizer.get_real_quantizer(
            qspec=input_quant_spec,
            quantizer=None,
            real_quantized=False,
            float_dtype=self.out_dtype,
        )

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

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # TODO: observer/quantize kernel unstable when cudagraph is enabled.
        qdq_x = self.input_quantizer(x)
        # TODO: Reference from QParamsLinear.forward. Casting after q/dp is required.
        dq_weight = self.weight_quantizer(layer.weight).to(self.out_dtype)

        return F.linear(qdq_x, dq_weight, bias)
