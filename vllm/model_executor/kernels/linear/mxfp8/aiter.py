# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class AiterMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 linear on AITER's tuned FlyDSL GEMM (gfx950).

    Weights are (16, 16)-preshuffled once at load; activation and weight
    scales stay row-major E8M0, [M, K / 32] and [N, K / 32].
    """

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
        if not rocm_aiter_ops.is_linear_enabled():
            return False, "AITER linear is disabled or not supported"
        # MiniMax-M3 mono decode re-tiles the MXFP8 QKV, output and separate
        # shared-expert weights from the row-major layout; only the shared
        # experts check for preshuffled weights.
        if os.getenv("VLLM_ROCM_MINIMAX_M3_MONO", "0") in ("1", "true", "True"):
            return False, "VLLM_ROCM_MINIMAX_M3_MONO needs row-major MXFP8 weights"
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        N, K = config.weight_shape
        if config.bmm_batch_size is not None:
            return False, "batched MXFP8 GEMM is not supported"
        # The FlyDSL GEMM writes bf16 only.
        if torch.get_default_dtype() != torch.bfloat16:
            return False, "requires a bf16 model"
        if not rocm_aiter_ops.is_mxfp8_bpreshuffle_gemm_tuned(N, K):
            return False, f"AITER has no tuned MXFP8 config for N={N}, K={K}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        N, K = weight.shape
        weight_scale = layer.weight_scale.data[:N, : K // MXFP8_BLOCK_SIZE]
        replace_parameter(
            layer, "weight", rocm_aiter_ops.shuffle_weight(weight.contiguous())
        )
        replace_parameter(
            layer,
            "weight_scale",
            weight_scale.contiguous().view(torch.float8_e8m0fnu),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x_q, x_scale = mxfp8_e4m3_quantize(x.reshape(-1, x.shape[-1]))
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_q,
            layer.weight,
            x_scale.view(torch.float8_e8m0fnu),
            layer.weight_scale,
            x.dtype,
        )
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
