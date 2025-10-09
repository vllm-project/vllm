# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
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
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


class QuarkOCP_MX(QuarkScheme):
    def __init__(
        self, weight_quant_spec: dict[str, Any], input_quant_spec: dict[str, Any]
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype = input_quant_spec["dtype"].replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: Union[int, Fraction] = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(
                dequant_mxfp6, quant_dtype=self.weight_dtype.replace("mx", "")
            )

        if self.input_dtype == "mxfp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(
                quant_dequant_mxfp6, quant_dtype=self.input_dtype.replace("mx", "")
            )

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

        # ruff: noqa: E501
        self.use_aiter_fp4_asm_gemm = (
            rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
        )  # enable asm kernel using env var

        # if asm is not enabled, use triton if aiter is available
        self.use_aiter_fp4_triton_gemm = (
            rocm_aiter_ops.is_enabled()
            and not rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
        )
        if not self.emulate:
            assert self.use_aiter_fp4_asm_gemm or self.use_aiter_fp4_triton_gemm, (
                f"{self.__class__.__name__} requires AITER to be installed"
            )
            "for non-emulation mode! "
            "ASM and Triton kernels are available "
            "from the AITER package in this mode. "
            "Enable the ASM kernel using VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1; "
            "otherwise, the Triton kernel will be used. "
            "Please refer to https://github.com/ROCm/aiter "
            "for installation details."

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
        else:
            if self.use_aiter_fp4_asm_gemm:
                # shuffle weight scale
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1
                )
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6
                ).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(
                    weight_scale_shuffle, requires_grad=False
                )

                # shuffle weight
                weight_shuffle = layer.weight.data
                weight_shuffle = rocm_aiter_ops.shuffle_weight(weight_shuffle)
                layer.weight = torch.nn.Parameter(weight_shuffle, requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(), requires_grad=False
                )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
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
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # case 1: emulate
        if self.emulate:
            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)

        # case 2: not emulate (aiter asm/ aiter triton)
        else:
            # priority 1: aiter asm
            if self.use_aiter_fp4_asm_gemm:
                return rocm_aiter_ops.asm_fp4_gemm_dynamic_quant(
                    x, layer.weight, layer.weight_scale, self.out_dtype
                )

            # priority 2: aiter triton
            if self.use_aiter_fp4_triton_gemm:
                return rocm_aiter_ops.triton_fp4_gemm_dynamic_qaunt(
                    x, layer.weight, layer.weight_scale, self.out_dtype
                )
