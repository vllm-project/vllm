# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact correctness for the MoRI EP MXFP4 upscale Triton kernel.

``upscale_mxfp4`` (``vllm...fused_moe.rocm_mori_utils``) dequantizes an MXFP4
(``float4_e2m1fn_x2``) payload + per-32 ``float8_e8m0fnu`` scale back to a float
dtype.  It runs after a MoRI fp4 dispatch when the downstream expert kernel has
no native fp4x2 path (non-mxfp4 weights).

We feed the kernel codes produced by a Python MXFP4 reference quantizer and
assert the kernel decode is bit-identical to the reference decode (the encoding
is shared, so the kernel's only job is to undo it).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
    pytest.skip(
        "torch.float4_e2m1fn_x2 / torch.float8_e8m0fnu not available in this "
        "torch build",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip(
        "MXFP4 upscale Triton kernel needs a CUDA/ROCm device",
        allow_module_level=True,
    )

from vllm.model_executor.layers.fused_moe.rocm_mori_utils import upscale_mxfp4

# MXFP4 (E2M1) magnitude levels, indexed by the 3 low bits of the code.
_MXFP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _mxfp4_quant_reference(x: torch.Tensor, group: int = 32):
    """Reference MXFP4 quantizer: returns ``(payload, scale, dequant_ref)``.

    ``payload`` and ``scale`` use the same on-wire layout the kernel expects;
    ``dequant_ref`` is the exact reference de-quantization the kernel must
    reproduce bit-for-bit.
    """
    M, N = x.shape
    assert N % group == 0
    levels = _MXFP4_LEVELS.to(x.device)

    absmax = x.abs().reshape(M, N // group, group).amax(dim=-1).clamp(min=1e-30)
    fp4_max = 6.0
    exp_f = torch.floor(torch.log2(absmax / fp4_max))
    exp_e8m0 = (exp_f + 127.0).clamp(0.0, 254.0).round().to(torch.int32)
    scale = torch.exp2(exp_e8m0.float() - 127.0)
    xq = x.float() / scale.repeat_interleave(group, dim=-1)

    nearest = (xq.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign_bit = (xq < 0).to(torch.uint8) * 8
    codes = (nearest.to(torch.uint8) | sign_bit).to(torch.uint8)

    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    packed_u8 = (odd.to(torch.int32) << 4 | even.to(torch.int32)).to(torch.uint8)
    payload = packed_u8.view(torch.float4_e2m1fn_x2)
    scale_u8 = exp_e8m0.to(torch.uint8).view(torch.float8_e8m0fnu)

    ref_codes = codes.to(torch.int32)
    ref_signed = torch.where(
        (ref_codes & 8) != 0,
        -levels[ref_codes & 7],
        levels[ref_codes & 7],
    )
    dequant = ref_signed * scale.repeat_interleave(group, dim=-1)
    return payload, scale_u8, dequant


@pytest.mark.parametrize(
    "M, N",
    [
        (16, 32),    # one block per row
        (16, 256),   # standard hidden_dim
        (64, 256),
        (64, 1024),  # multiple BLOCK_N tiles
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    [torch.bfloat16, torch.float16, torch.float32],
)
def test_upscale_mxfp4_bitexact_roundtrip(M, N, out_dtype):
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 4.0
    payload, scale, dequant_ref = _mxfp4_quant_reference(x)

    recv = torch.tensor(M, device=device, dtype=torch.int32)
    out = upscale_mxfp4(payload, scale, recv, out_dtype)

    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    torch.testing.assert_close(
        out.float(),
        dequant_ref.to(out_dtype).float(),
        atol=0.0,
        rtol=0.0,
    )


def test_upscale_mxfp4_partial_recv_matches_full_run():
    torch.manual_seed(0)
    device = "cuda"
    M, N = 32, 128

    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    payload, scale, _ = _mxfp4_quant_reference(x)

    full = upscale_mxfp4(
        payload, scale, torch.tensor(M, device=device, dtype=torch.int32),
        torch.bfloat16,
    )
    keep = M - 5
    partial = upscale_mxfp4(
        payload, scale, torch.tensor(keep, device=device, dtype=torch.int32),
        torch.bfloat16,
    )

    torch.testing.assert_close(
        partial[:keep].float(), full[:keep].float(), atol=0.0, rtol=0.0
    )


def test_upscale_mxfp4_zero_recv_returns_shaped_buffer():
    """Zero received tokens: return a properly-shaped output, no raise."""
    device = "cuda"
    M, N = 8, 64
    payload = torch.zeros(M, N // 2, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    scale = torch.zeros(M, N // 32, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )

    out = upscale_mxfp4(
        payload,
        scale,
        torch.tensor(0, device=device, dtype=torch.int32),
        torch.bfloat16,
    )
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
