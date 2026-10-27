# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class AiterMxfp8LinearKernel(Mxfp8LinearKernel):
    """AITER-based MXFP8 GEMM on ROCm."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "AITER is only supported on ROCm"

        if not current_platform.supports_mx():
            return False, "native MX requires CDNA4 (gfx95x)"

        if envs.VLLM_BATCH_INVARIANT:
            return False, "tuned buckets use split K; not batch invariant"

        if rocm_aiter_ops.is_enabled():
            return True, None
        return False, "AITER not found or not supported on the current platform"

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        N, K = config.weight_shape
        if config.bmm_batch_size is not None:
            return False, "batched MXFP8 GEMM is not supported"
        if K % MXFP8_BLOCK_SIZE != 0:
            return False, f"K={K} is not a multiple of {MXFP8_BLOCK_SIZE}"
        if not rocm_aiter_ops.is_triton_gemm_afp8wfp8_tuned(N, K):
            return False, f"AITER has no tuned MXFP8 config for N={N}, K={K}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k]

        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])
        out = rocm_aiter_ops.mxfp8_gemm_dynamic_quant(
            x2d, layer.weight, layer.weight_scale, x.dtype
        )
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
