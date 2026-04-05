# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.model_executor.utils import replace_parameter


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def _requant_weight_per_tensor(self, layer: torch.nn.Module) -> None:
        device = layer.weight.device
        # Get the max scale on the weight's device
        max_scale = torch.max(layer.weight_scale.data.to(device))
        orig_dtype = layer.weight.dtype
        start_idx = 0
        for index, width in enumerate(layer.logical_widths):
            end_idx = start_idx + width
            if width == 0:
                continue

            shard_weight = layer.weight[start_idx:end_idx, :]
            shard_scale = layer.weight_scale[index].to(device)

            # Dequantize and then requantize with max_scale
            dequantized_shard = shard_weight.to(torch.float32) * shard_scale
            requantized_shard = (dequantized_shard / max_scale).to(orig_dtype)

            layer.weight[start_idx:end_idx, :] = requantized_shard
            start_idx = end_idx

        replace_parameter(layer, "weight_scale", max_scale)
            
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Give xpu only support to per-tensor strategy for now
        # So if we have a fused module (QKV, MLP) with per tensor scales,
        # requantize the weights w/ max scale
        is_compressed_tensors = hasattr(layer, "scheme") and "CompressedTensors" in type(layer.scheme).__name__
        is_fused_per_tensor = (
            hasattr(layer, "logical_widths")
            and len(layer.logical_widths) > 1
            and layer.weight_scale.numel() == len(layer.logical_widths)
            and layer.weight_scale.dim() == 1
        )
        if is_compressed_tensors and is_fused_per_tensor:
            self._requant_weight_per_tensor(layer)
 
        replace_parameter(layer, "weight", layer.weight.t())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass
