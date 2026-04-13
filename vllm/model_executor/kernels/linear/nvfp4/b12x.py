# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
)
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.platforms import current_platform


def _import_b12x_cute_fp4():
    try:
        return importlib.import_module("b12x.cute.fp4")
    except ImportError:
        return None


def _import_b12x_gemm_dense():
    try:
        return importlib.import_module("b12x.gemm.dense")
    except ImportError:
        return None


class B12xNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the optional external b12x SM12x backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x NVFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x NVFP4 kernels require a Blackwell 12x device"
        if _import_b12x_cute_fp4() is None:
            return False, "b12x.cute.fp4 is not importable"
        if _import_b12x_gemm_dense() is None:
            return False, "b12x.gemm.dense is not importable"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        b12x_cute_fp4 = _import_b12x_cute_fp4()
        b12x_gemm_dense = _import_b12x_gemm_dense()
        if b12x_cute_fp4 is None or b12x_gemm_dense is None:
            raise ImportError("b12x is not installed or importable")
        self._as_grouped_scale_view = b12x_cute_fp4.as_grouped_scale_view
        self._swizzle_block_scale = b12x_cute_fp4.swizzle_block_scale
        self._dense_gemm = b12x_gemm_dense.dense_gemm

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            self._swizzle_block_scale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        m = x_2d.shape[0]
        x_packed, x_scale_swizzled = scaled_fp4_quant(
            x_2d,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="cutlass",
        )
        x_packed = pad_nvfp4_activation_for_cutlass(
            x_packed, getattr(layer, "weights_padding_cols", 0)
        )
        k = x_packed.shape[1] * 2
        x_scale = self._as_grouped_scale_view(
            x_scale_swizzled.view(torch.uint8).unsqueeze(0),
            m,
            k,
        )
        weight_scale = self._as_grouped_scale_view(
            layer.weight_scale.view(torch.uint8).unsqueeze(0),
            layer.weight.shape[0],
            k,
        )

        out = torch.empty(
            (m, layer.weight.shape[0], 1),
            device=x.device,
            dtype=output_dtype,
        )
        self._dense_gemm(
            (x_packed.unsqueeze(-1), x_scale),
            (layer.weight.unsqueeze(-1), weight_scale),
            out=out,
            alpha=layer.alpha.view(1),
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype=str(output_dtype).split(".")[-1],
            sf_vec_size=16,
        )
        out = slice_nvfp4_output(out[:, :, 0], output_size)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
