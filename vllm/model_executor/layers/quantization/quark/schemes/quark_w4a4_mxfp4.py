# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE, per_token_group_quant_mxfp4)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform

__all__ = ["QuarkW4A4MXFP4"]


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.emulate = not current_platform.supports_mx()

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                requires_grad=False)

        if self.emulate:
            try:
                from quark.torch.export.nn.modules import realquantizer
                from quark.torch.quantization.config.config import (
                    QuantizationSpec)
            except ImportError as err:
                raise ImportError(
                    "The package `amd-quark` is required to use AMD Quark "
                    "MX-FP4 models. Please install it with `pip install "
                    "amd-quark`.") from err

            weight_quant_spec = QuantizationSpec.from_dict(
                self.weight_quant_spec)

            weight_quantizer = realquantizer.get_real_quantizer(
                qspec=weight_quant_spec,
                quantizer=None,
                real_quantized=True,
                reorder=False,
                float_dtype=self.out_dtype,
                scale_shape=layer.weight_scale.shape,
                zero_point_shape=None,
            )
            weight_quantizer.scale.data = layer.weight_scale.data

            if not envs.VLLM_QUARK_EMU_MEM_OPT:
                layer.weight = torch.nn.Parameter(
                    weight_quantizer(layer.weight.data).to(self.out_dtype),
                    requires_grad=False,
                )
            else:
                self.weight_quantizer = weight_quantizer
            layer.weight_scale = None

            # This call is necessary to release the scales memory.
            torch.cuda.empty_cache()

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
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
            weight_loader=weight_loader,
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

        if self.emulate:
            if envs.VLLM_QUARK_EMU_MEM_OPT:
                dq_w = self.weight_quantizer(layer.weight).to(self.out_dtype)
            else:
                dq_w = layer.weight
            qdq_x, _ = per_token_group_quant_mxfp4(x, OCP_MX_BLOCK_SIZE)
            return F.linear(qdq_x, dq_w, bias)
        else:
            raise NotImplementedError()
