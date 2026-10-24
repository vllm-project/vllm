# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MLA value up-projection + output quantization.

``MLAAttention._v_up_proj`` (see ``mla_attention.py``) runs a per-head batched
GEMM — ``X[N,B,L] @ W_UV[N,L,V] -> O[N,B,V]`` — that lifts the compressed MQA
attention output back to the full value head dimension, immediately before
``o_proj`` consumes it. Today that GEMM always writes a plain bf16/fp16
output; when ``o_proj`` wants a pre-quantized (FP8) activation, a separate
pass quantizes the result afterward (see the ``quant_key is not None`` tail
in ``MLAAttention.forward_impl``).

This module moves that quantization into an explicit model-code call site
right after the up-projection, matching the ``QuantizedActivation`` contract
(``vllm/model_executor/layers/fusion/quant_activation.py``) so ``o_proj`` can
consume the result without re-quantizing. Two code paths exist:

- ``_v_up_proj_fp8_static_quant_fused``: a single Triton kernel launch that
  computes the GEMM and quantizes its own output in one epilogue. Triton can
  only emit the ``float8_e4m3fn`` ("fp8e4nv") format vLLM uses elsewhere on
  sm_90+ (Hopper/Blackwell) — on Ampere this fails to compile
  (``type fp8e4nv not supported in this architecture``), confirmed by actually
  running it on an A100.
- ``_v_up_proj_fp8_static_quant_portable``: the same batched matmul, followed
  immediately by vLLM's existing, hardware-portable
  ``_custom_ops.scaled_fp8_quant`` kernel. Two kernel launches instead of one,
  but it is an explicit call site rather than a compiler-matched pattern
  (the actual ask in vLLM issue #43498), and it runs everywhere vLLM does.

``v_up_proj_fp8_static_quant`` dispatches between the two based on detected
compute capability, so callers do not need to know which backend ran.

Only FP8 static per-tensor quantization is implemented so far. Per-token-block
FP8 and NVFP4 are follow-ups (see the inventory in vLLM issue #43498).
"""

from functools import lru_cache

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

FP8_DTYPE = current_platform.fp8_dtype()
FP8_MIN = float(torch.finfo(FP8_DTYPE).min)
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


@lru_cache(maxsize=1)
def _fused_epilogue_supported() -> bool:
    """True on sm_90+ (Hopper/Blackwell), where Triton can emit fp8e4nv.

    Verified by actually compiling the fused kernel on an A100 (sm_80): it
    fails with "type fp8e4nv not supported in this architecture. The
    supported fp8 dtypes are ('fp8e4b15', 'fp8e5')". We don't try to be
    clever about those legacy encodings here — they don't match vLLM's
    float8_e4m3fn convention used by every FP8 consumer downstream, so an
    Ampere "fused" kernel in a different FP8 format would just move the
    mismatch somewhere else. Fall back to the portable path instead.
    """
    if not current_platform.is_cuda():
        return False
    major, _ = current_platform.get_device_capability()
    return major >= 9


@triton.jit
def _v_up_proj_fp8_static_quant_kernel(
    X,
    W,
    Out,
    OutScale,
    stride_x_n,
    stride_x_b,
    stride_x_l,
    stride_w_n,
    stride_w_l,
    stride_w_v,
    stride_o_b,
    stride_o_n,
    stride_o_v,
    B,
    L,
    V,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_L: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One (head, B-tile, V-tile) program computes a BLOCK_B x BLOCK_V output
    tile of X[n] @ W[n], in fp32, then quantizes it with a single static
    scale before storing.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_v = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_l = tl.arange(0, BLOCK_L)

    x_base = X + pid_n * stride_x_n
    w_base = W + pid_n * stride_w_n

    acc = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for l_start in range(0, L, BLOCK_L):
        l_idx = l_start + offs_l
        x_ptrs = x_base + offs_b[:, None] * stride_x_b + l_idx[None, :] * stride_x_l
        w_ptrs = w_base + l_idx[:, None] * stride_w_l + offs_v[None, :] * stride_w_v
        x_mask = (offs_b[:, None] < B) & (l_idx[None, :] < L)
        w_mask = (l_idx[:, None] < L) & (offs_v[None, :] < V)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    scale = tl.load(OutScale)
    quantized = tl.clamp(acc / scale, FP8_MIN, FP8_MAX).to(Out.dtype.element_ty)

    o_ptrs = (
        Out
        + offs_b[:, None] * stride_o_b
        + pid_n * stride_o_n
        + offs_v[None, :] * stride_o_v
    )
    o_mask = (offs_b[:, None] < B) & (offs_v[None, :] < V)
    tl.store(o_ptrs, quantized, mask=o_mask)


def _v_up_proj_fp8_static_quant_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    output_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-kernel fused ``X[N,B,L] @ W[N,L,V]`` + static FP8 quantization.

    sm_90+ only (see ``_fused_epilogue_supported``).

    Args:
        x: Compressed MQA attention output, ``[N, B, L]`` (bf16/fp16).
        w: Up-projection weight, ``[N, L, V]`` (same dtype as ``x``).
        output_scale: Scalar FP8 static quantization scale (``o / scale``
            matches the convention used elsewhere in this file's quant tail,
            i.e. the inverse of a "quantize by multiplying" scale).
        out: Optional preallocated ``[B, N, V]`` FP8 output buffer.

    Returns:
        The ``[B, N, V]`` FP8 tensor (``out`` if given).

    """
    assert x.ndim == 3 and w.ndim == 3
    N, B, L = x.shape
    Nw, Lw, V = w.shape
    assert Nw == N and Lw == L, f"shape mismatch: x={x.shape} w={w.shape}"
    assert output_scale.numel() == 1

    if out is None:
        out = torch.empty((B, N, V), dtype=FP8_DTYPE, device=x.device)
    else:
        assert out.shape == (B, N, V)

    BLOCK_B = 32
    BLOCK_V = 64
    BLOCK_L = min(128, triton.next_power_of_2(L))

    grid = (N, triton.cdiv(B, BLOCK_B), triton.cdiv(V, BLOCK_V))
    _v_up_proj_fp8_static_quant_kernel[grid](
        x,
        w,
        out,
        output_scale,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        L,
        V,
        BLOCK_B=BLOCK_B,
        BLOCK_V=BLOCK_V,
        BLOCK_L=BLOCK_L,
        FP8_MIN=FP8_MIN,
        FP8_MAX=FP8_MAX,
    )
    return out


def _v_up_proj_fp8_static_quant_portable(
    x: torch.Tensor,
    w: torch.Tensor,
    output_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain batched matmul + vLLM's existing static FP8 quant kernel.

    Two kernel launches, but called back-to-back from one explicit site
    instead of relying on a compiler fusion pass. Runs on any platform
    ``_custom_ops.scaled_fp8_quant`` supports (Ampere included).
    """
    assert x.ndim == 3 and w.ndim == 3
    N, B, L = x.shape
    Nw, Lw, V = w.shape
    assert Nw == N and Lw == L, f"shape mismatch: x={x.shape} w={w.shape}"
    assert output_scale.numel() == 1

    # [N, B, L] @ [N, L, V] -> [N, B, V] -> [B, N, V] -> [B, N*V] for the
    # 2D-input quant kernel's per-tensor static scaling path.
    unquantized = torch.bmm(x, w).transpose(0, 1).reshape(B, N * V)

    quantized, _ = ops.scaled_fp8_quant(unquantized, output_scale)
    quantized = quantized.view(B, N, V)

    if out is not None:
        out.copy_(quantized)
        return out
    return quantized


def v_up_proj_fp8_static_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    output_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``X[N,B,L] @ W[N,L,V]`` + static FP8 quantization of the result.

    Dispatches to the single-kernel fused epilogue on sm_90+, and to the
    portable matmul-then-quantize path everywhere else (see module
    docstring for why the fused path cannot target Ampere directly).
    """
    if _fused_epilogue_supported():
        return _v_up_proj_fp8_static_quant_fused(x, w, output_scale, out)
    return _v_up_proj_fp8_static_quant_portable(x, w, output_scale, out)
