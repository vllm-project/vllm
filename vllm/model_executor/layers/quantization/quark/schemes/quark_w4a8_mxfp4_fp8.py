# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


__all__ = ["QuarkW4A8_MXFP4_FP8"]

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A8_MXFP4_FP8(QuarkScheme):
    """
    - Weights: MXFP4 with E8M0 scales per block of 32
    - Activations: FP8 E4M3 (static per-tensor quantization)

    Uses the AITER Triton kernel and falls back to emulation if AITER not available.
    """

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
    ):
        self.out_dtype = None

        self.weight_dtype = "mxfp4"
        self.packed_factor: Fraction = Fraction(2, 1)  # 2 FP4 values per byte
        self.weight_block_size = OCP_MX_BLOCK_SIZE

        self.is_static_input_scheme = not input_quant_spec.get("is_dynamic")
        self.input_qscheme = input_quant_spec.get("qscheme")  # "per_tensor"

        if not self.is_static_input_scheme:
            raise NotImplementedError(
                "Dynamic FP8 activation quantization is not yet supported "
                "for W4A8. The current implementation expects static per-tensor "
                "FP8 scales stored in the checkpoint."
            )

        kernel_supported_gpu = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx950

            kernel_supported_gpu = on_gfx950()

        self.use_aiter_kernel = (
            is_aiter_found_and_supported()
            and self.is_static_input_scheme
            and kernel_supported_gpu
        )

        if not self.use_aiter_kernel:
            logger.warning_once(
                "[W4A8 MXFP4+FP8] Aiter Triton kernel not found. Using emulation mode."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_packed_dim(self, dim: int) -> int:
        assert dim % 2 == 0, f"Dimension {dim} must be even for MXFP4 packing"
        return dim // 2

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
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # MXFP4 WEIGHT (packed, 2 values per byte)
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                self.get_packed_dim(input_size_per_partition),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE (E8M0 format, per block of 32)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.weight_block_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (FP8 per-tensor static scale)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes),
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            # Initialize to avoid NaN
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Ensuring weights & scales are non-trainable
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

        if self.is_static_input_scheme:
            input_scale = layer.input_scale.data
            # For fused modules (QKV), take the max scale
            if input_scale.numel() != 1:
                input_scale = input_scale.max()

            layer.input_scale = torch.nn.Parameter(
                torch.tensor(input_scale, dtype=torch.float32),
                requires_grad=False,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_aiter_kernel:
            return self._apply_aiter_kernel(layer, x, bias)
        else:
            return self._apply_emulation(layer, x, bias)

    def _apply_aiter_kernel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M = x.shape[0]
        out_dtype = x.dtype if self.out_dtype is None else self.out_dtype

        input_scale = layer.input_scale
        finfo = torch.finfo(torch.float8_e4m3fn)  # [-448, 448]
        x_fp8 = (x / input_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

        # Broadcast per-tensor scale to per-row (M, 1) for Aiter kernel
        x_scales = input_scale.expand(M, 1).to(dtype=torch.float32, device=x.device)

        y = rocm_aiter_ops.gemm_a8wfp4(
            x_fp8, layer.weight, x_scales, layer.weight_scale, out_dtype
        )

        if bias is not None:
            y = y + bias

        return y

    def _apply_emulation(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            dequant_mxfp4,
        )

        weight_dq = dequant_mxfp4(
            layer.weight,
            layer.weight_scale,
            x.dtype,
        )

        input_scale = layer.input_scale
        finfo = torch.finfo(torch.float8_e4m3fn)  # [-448, 448]
        x_fp8 = (x / input_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        x_dq = (x_fp8.to(x.dtype) * input_scale).to(x.dtype)

        return F.linear(x_dq, weight_dq, bias)
