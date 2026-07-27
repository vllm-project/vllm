# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

_MXFP4_GROUP_SIZE = 32

# Fake/meta impl so the op is traceable under torch.compile + CUDA graphs.
if hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "mxfp4_gemm_rdna3"):
    try:

        @torch.library.register_fake("_rocm_C::mxfp4_gemm_rdna3")
        def _mxfp4_gemm_rdna3_fake(a, b_q_weight, b_scales_e8m0):
            return a.new_empty((a.shape[0], b_q_weight.shape[1]))
    except RuntimeError:
        pass  # already registered


class Rdna3MxFp4LinearKernel(MxFp4LinearKernel):
    """Weight-only MXFP4 GEMM for RDNA3 (gfx11xx) via the native HIP kernel
    ``_rocm_C.mxfp4_gemm_rdna3`` (scalar GEMV for decode, WMMA for prefill).
    RDNA3 has no native FP4 matmul, so E2M1 weights are dequantized on the fly
    into the f16/bf16 pipeline (W4A16).
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not running on ROCm"
        from vllm.platforms.rocm import on_gfx1100

        if not on_gfx1100():
            return False, "RDNA3 gfx1100 WMMA required"
        if not hasattr(torch.ops._rocm_C, "mxfp4_gemm_rdna3"):
            return False, "_rocm_C.mxfp4_gemm_rdna3 op not built"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # layer.weight: [N, K/2] uint8, two E2M1 codes per byte (low nibble at
        # the even K index). layer.weight_scale: [N, K/32] uint8 E8M0.
        w = layer.weight.data
        scale = layer.weight_scale.data
        N, k_half = w.shape
        K = k_half * 2

        if N % 16 != 0 or K % _MXFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"RDNA3 MXFP4 WMMA kernel needs N%16==0 and K%32==0, got N={N}, K={K}"
            )

        # [N, K/2] uint8 reinterpreted little-endian as int32 is already the
        # [K/8, N] word layout the kernel reads, so repack = view + transpose.
        b_q = w.contiguous().view(torch.int32).t().contiguous()  # [K/8, N]
        b_scale = scale.t().contiguous()  # E8M0 [K/32, N]

        layer.weight = Parameter(b_q, requires_grad=False)
        layer.weight_scale = Parameter(b_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        out = torch.ops._rocm_C.mxfp4_gemm_rdna3(x_2d, layer.weight, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out.view(out_shape)
