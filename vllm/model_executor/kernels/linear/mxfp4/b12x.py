# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    b12x_warmup_token_counts,
)
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.b12x import get_b12x_intrinsics as _import_b12x_intrinsics

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


def _apply_b12x_mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_storage: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None

    output_size = int(weight.shape[0])
    output_shape = [*x.shape[:-1], output_size]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_packed, x_scale_swizzled = flashinfer_mxfp4_quantize(x_2d, backend="cute-dsl")
    output = blockscaled.mm_mxfp4(
        x_packed,
        x_scale_swizzled,
        weight,
        weight_scale_storage,
        out_dtype=x.dtype,
    )
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def warmup_b12x_mxfp4_linear(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    output_dtype: torch.dtype = torch.bfloat16,
) -> int:
    if not current_platform.is_cuda():
        return 0
    if not current_platform.is_device_capability_family(120):
        return 0
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16
    layer_map: dict[tuple[Any, ...], torch.nn.Module] = {}
    for layer in model.modules():
        if not getattr(layer, "b12x_mxfp4_linear", False):
            continue
        weight = layer.weight
        weight_scale = layer.weight_scale
        n, packed_k = map(int, weight.shape)
        signature = (
            weight.device,
            n,
            packed_k * 2,
            weight.dtype,
            weight_scale.dtype,
            output_dtype,
        )
        layer_map.setdefault(signature, layer)
    if not layer_map:
        return 0
    if _import_b12x_blockscaled() is None:
        return 0

    token_counts = b12x_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    warmed = 0
    last_device: torch.device | None = None

    with torch.inference_mode():
        for signature, layer in layer_map.items():
            weight = layer.weight
            weight_scale = layer.weight_scale
            k = signature[2]
            last_device = weight.device
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, k),
                    dtype=output_dtype,
                    device=weight.device,
                )
                _apply_b12x_mxfp4_linear(
                    source,
                    weight,
                    weight_scale,
                    None,
                )
                warmed += 1

        if warmed > 0 and last_device is not None and last_device.type == "cuda":
            torch.accelerator.synchronize(last_device)

    return warmed


class B12xMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X MXFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X MXFP4 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None or _import_b12x_intrinsics() is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "b12x native MXFP4 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "B12X MXFP4 GEMM requires dynamic MXFP4 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        intrinsics = _import_b12x_intrinsics()
        assert intrinsics is not None
        replace_parameter(
            layer,
            "weight_scale",
            intrinsics.swizzle_block_scale(layer.weight_scale.data),
        )
        layer.b12x_mxfp4_linear = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_b12x_mxfp4_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias,
        )


__all__ = ["B12xMxFp4LinearKernel", "warmup_b12x_mxfp4_linear"]
