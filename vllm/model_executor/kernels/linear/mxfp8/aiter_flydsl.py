# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class AiterFlyDSLMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 linear on AITER's tuned FlyDSL GEMM (gfx950).

    Weights are (16, 16)-preshuffled once at load; activation and weight
    scales stay row-major E8M0, [M, K / 32] and [N, K / 32]. Activations are
    MXFP8-quantized here, or arrive pre-quantized as a `QuantizedActivation`
    with the same row-major scales.
    """

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

    def input_quant_key(self) -> QuantKey:
        # Row-major [M, K // 32] E8M0 scales, as RocmDotScaledMxfp8LinearKernel
        # reads them (FlashInfer's are swizzled under the same key).
        return kMxfp8Dynamic

    @staticmethod
    def supports_quantized_input(layer: torch.nn.Module) -> bool:
        """Whether `apply_weights` reads a `QuantizedActivation` for `layer`
        as is; always, since the weight stays fp8."""
        return True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            out_dtype, in_shape = qa.orig_dtype, qa.orig_shape
            x_q, x_scale = qa.data.reshape(-1, in_shape[-1]), qa.scale
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, in_shape = x.dtype, x.shape
            x_q, x_scale = mxfp8_e4m3_quantize(x.reshape(-1, in_shape[-1]))
        # AITER only takes its MXFP8 path for [M, K / 32] E8M0 scales.
        assert x_scale.shape == (x_q.shape[0], x_q.shape[1] // MXFP8_BLOCK_SIZE), (
            f"MXFP8 activation scale {tuple(x_scale.shape)} for {tuple(x_q.shape)}"
        )
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_q,
            layer.weight,
            x_scale.view(torch.float8_e8m0fnu),
            layer.weight_scale,
            out_dtype,
        )
        out = out.reshape(*in_shape[:-1], layer.weight.shape[0])
        if bias is not None:
            out = out + bias
        return out
