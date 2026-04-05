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
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


__all__ = ["QuarkW4A16_MXFP4_A16"]

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A16_MXFP4_A16(QuarkScheme):
    """
    - Weights: MXFP4 with E8M0 scales per block of 32
    - Activations: FP16/BF16 (no activation quantization, A16)

    Uses the AITER batched_gemm_a16wfp4 kernel when available (ROCm gfx9),
    otherwise falls back to dequant + F.linear (triton/emulation).
    """

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any] | None,
    ):
        self.out_dtype = None

        self.weight_dtype = "mxfp4"
        self.packed_factor: Fraction = Fraction(2, 1)  # 2 FP4 values per byte
        self.weight_block_size = OCP_MX_BLOCK_SIZE

        kernel_supported_gpu = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx950

            kernel_supported_gpu = on_gfx950()

        self.use_aiter_kernel = (
            is_aiter_found_and_supported()
            and kernel_supported_gpu
            and rocm_aiter_ops.is_fp4bmm_enabled()
        )

        if not self.use_aiter_kernel:
            logger.warning_once(
                "[W4A16 MXFP4+A16] Aiter batched_gemm_a16wfp4 not used. "
                "Using dequant + F.linear fallback."
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_aiter_kernel:
            return self._apply_aiter_kernel(layer, x, bias)
        return self._apply_triton_fallback(layer, x, bias)

    def _apply_aiter_kernel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M, K = x.shape[0], x.shape[1]
        N = layer.weight.shape[0]
        out_dtype = x.dtype if self.out_dtype is None else self.out_dtype

        # batched_gemm_a16wfp4 expects 3D (batch, M, K) and (batch, N, K) for X @ W^T
        # For a single linear we use batch=1.
        x_3d = x.unsqueeze(0)  # (1, M, K)
        w = layer.weight  # (N, K/2) packed
        w_scale = layer.weight_scale  # (N, K/32)
        y_3d = torch.empty(
            1, M, N, dtype=out_dtype, device=x.device
        )
        w_3d = w.unsqueeze(0)  # (1, N, K/2)

        rocm_aiter_ops.batched_gemm_a16wfp4(
            x_3d,
            w_3d,
            w_scale,
            y_3d,
            transpose_bm=True,
            prequant=False,
            y_scale=None,
        )
        y = y_3d.squeeze(0)  # (M, N)

        if bias is not None:
            y = y + bias

        return y

    def _apply_triton_fallback(
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
        return F.linear(x, weight_dq, bias)
