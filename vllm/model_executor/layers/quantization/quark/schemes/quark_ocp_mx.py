# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6,
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


class QuarkOCP_MX(QuarkScheme):
    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any] | None,
        dynamic_mxfp4_quant: bool = False,
    ):
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.dynamic_mxfp4_quant = dynamic_mxfp4_quant
        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype: str | None = None
        if input_quant_spec is not None:
            input_quant = input_quant_spec["dtype"]
            if input_quant == "fp8_e4m3":
                self.input_dtype = "fp8"
            else:
                self.input_dtype = input_quant.replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(
                dequant_mxfp6, quant_dtype=self.weight_dtype.replace("mx", "")
            )

        if self.input_dtype is None:
            self.quant_dequant_func: Callable[[torch.Tensor], torch.Tensor] = (
                lambda x: x
            )  # no input Q/DQ for weight-only
        elif self.input_dtype == "mxfp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(
                quant_dequant_mxfp6, quant_dtype=self.input_dtype.replace("mx", "")
            )

        if input_quant_spec is None:
            self.static_input_scales = False
        else:
            self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # TODO: integrate (or test) mixed-precision kernel.
        self.emulate = not current_platform.supports_mx() or (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        )

        if not self.emulate:
            self.ocp_mx_linear = init_mxfp4_linear_kernel()

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if current_platform.supports_mx() and (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_dynamic_mxfp4_weights_after_loading(
        self, layer: torch.nn.Module
    ) -> None:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        w_q, w_s = dynamic_mxfp4_quant(layer.weight)
        layer.weight_scale = torch.nn.Parameter(w_s.T.contiguous(), requires_grad=False)
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.emulate:
            if self.dynamic_mxfp4_quant:
                self.process_dynamic_mxfp4_weights_after_loading(layer)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
        else:
            if self.dynamic_mxfp4_quant:
                self.process_dynamic_mxfp4_weights_after_loading(layer)
            self.ocp_mx_linear.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if self.dynamic_mxfp4_quant:
            weight = ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, kwargs)
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes

            # WEIGHT
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    self.get_packed_dim(input_size_per_partition, self.weight_dtype),
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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.emulate:
            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)
        return self.ocp_mx_linear.apply_weights(layer, x, bias)
