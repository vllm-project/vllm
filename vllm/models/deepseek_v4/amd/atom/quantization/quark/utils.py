# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton
import triton.language as tl
import torch
from aiter import QuantType


@triton.jit
def _weight_dequant_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):  # type: ignore[no-untyped-def]
    """
    Triton kernel for dequantizing FP8 weights using scaling factors.

    This kernel is provided by deepseek-ai for efficient FP8 weight dequantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant_fp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize FP8 weight tensor using inverse scale with Triton kernel.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Optional E8M0 dtype: only available on newer torch builds.
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def weight_dequant_mxfp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Dequantize an MXFP8 weight to the default float dtype."""
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, K = x.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    n_blocks = K // block_size
    assert s.shape == (M, n_blocks), f"scale shape {tuple(s.shape)} != {(M, n_blocks)}"

    if _E8M0_DTYPE is not None and s.dtype == _E8M0_DTYPE:
        # E8M0 dtype decodes straight to the 2**(e-127) multiplier.
        scale = s.to(torch.float32)
    else:
        # Raw E8M0 integer codes stored as uint8 / float.
        scale = torch.exp2(s.to(torch.float32) - 127.0)

    out_dtype = torch.get_default_dtype()
    y = x.to(torch.float32).reshape(M, n_blocks, block_size)
    y = y * scale.unsqueeze(-1)
    return y.reshape(M, K).to(out_dtype)


def quant_mxfp4_online_even(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online MXFP4 weight quant via the aiter HIP kernel with ``Even`` round mode.

    Round-half-to-even on the FP4/E2M1 grid + an E8M0 block scale (note: on
    gfx942 ``Even`` falls back to round-half-away in software). Returns the
    packed weight viewed as ``dtypes.fp4x2`` and the block scale as
    ``dtypes.fp8_e8m0``.

    Shared by the Linear and MoE online-quant paths so both stay in sync.
    ``quant_mxfp4_hip`` requires a 2D contiguous fp16/bf16 input, so we
    normalise the input accordingly before calling it.
    """
    from aiter import dtypes
    from aiter.ops.quant import quant_mxfp4_hip
    from aiter.utility.mx_types import MxScaleRoundModeInt

    q_in = weight.contiguous()
    if q_in.dtype not in (torch.float16, torch.bfloat16):
        q_in = q_in.to(torch.bfloat16)
    q_weight, weight_scale = quant_mxfp4_hip(q_in, round_mode=MxScaleRoundModeInt.Even)
    return q_weight.view(dtypes.fp4x2), weight_scale.view(dtypes.fp8_e8m0)


def quant_weight_online(
    weight: torch.Tensor,
    online_quant_type: QuantType,
    online_quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch online weight quantization by target dtype.

    Single entry point shared by the Linear and MoE online-quant paths so both
    stay in sync:

    - MXFP4 (``dtypes.fp4x2``): use the aiter HIP kernel with ``Even`` round
      mode (:func:`quant_mxfp4_online_even`), matching the offline Quark kernel.
    - FP8 (incl. ptpc_fp8 per-token / per-channel): use the aiter quant
      function resolved from ``get_hip_quant(online_quant_type)``.

    :param weight: The (already dequantized) weight tensor to quantize.
    :param online_quant_type: Online quantization scheme, used to resolve the
        FP8 quant function via ``get_hip_quant``.
    :param online_quant_dtype: Target online quantization dtype.
    :return: ``(q_weight, weight_scale)``.
    """
    from aiter import dtypes, get_hip_quant

    if online_quant_dtype == dtypes.fp4x2:
        return quant_mxfp4_online_even(weight)
    quant_func = get_hip_quant(online_quant_type)
    return quant_func(weight, quant_dtype=online_quant_dtype)
