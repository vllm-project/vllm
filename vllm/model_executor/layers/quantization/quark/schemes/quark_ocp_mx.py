# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4, quant_dequant_mxfp4)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6, quant_dequant_mxfp6)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE, OCP_MX_Scheme)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)

__all__ = ["QuarkOCP_MX"]


class QuarkOCP_MX(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

        self.weight_dtype = weight_quant_spec["dtype"]
        self.input_dtype = input_quant_spec["dtype"]

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype)

        if self.weight_dtype == "fp4":
            self.packed_factor: Union[int, Fraction] = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(dequant_mxfp6,
                                        quant_dtype=self.weight_dtype)

        if self.input_dtype == "fp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(quant_dequant_mxfp6,
                                              quant_dtype=self.input_dtype)

        self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue.")

        if not current_platform.supports_mx():
            self.emulate = True
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision.")
        else:
            self.emulate = True
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                "computation, but kernels are not yet integrated in vLLM. "
                "Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision.")

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "fp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"fp6_e3m2", "fp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                requires_grad=False)

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
                self.get_packed_dim(input_size_per_partition,
                                    self.weight_dtype),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
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
            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)
        else:
            raise NotImplementedError()
