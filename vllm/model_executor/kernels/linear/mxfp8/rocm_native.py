# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP8 linear GEMM for AMD CDNA4 (gfx950) via Triton ``tl.dot_scaled``.

Consumes the FP8 E4M3 weights + E8M0 block scales directly (no dequant-to-BF16);
activations are MXFP8-quantized per token. Uses the CDNA4 hardware microscaling
matrix cores. Falls back (via the kernel selector) to the BF16
``EmulationMxfp8LinearKernel`` on archs without native MX or for shapes with
``K % 128 != 0``.
"""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _mxfp8_dot_scaled_linear(
    x: torch.Tensor,  # [M, K] bf16/fp16
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x_q,
        x_scale,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x_q.stride(0),
        x_q.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    return out


class RocmDotScaledMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native CDNA4 (gfx950) MXFP8 linear via Triton ``tl.dot_scaled``."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        # supports_mx() == gfx95x (CDNA4 native microscaling hardware). On other
        # archs dot_scaled would upcast to BF16, so the kernel selector falls
        # through to the BF16 emulation (hipBLASLt) path instead.
        if not current_platform.supports_mx():
            return False, "native MX requires CDNA4 (gfx95x)"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] fp8
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

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
        if x2d.shape[-1] % 128 == 0:
            out = _mxfp8_dot_scaled_linear(x2d, layer.weight, layer.weight_scale)
        else:
            # dot_scaled tiling needs K % 128 == 0; dequantize fallback otherwise.
            w_bf16 = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
            out = torch.nn.functional.linear(x2d, w_bf16).to(x.dtype)
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
