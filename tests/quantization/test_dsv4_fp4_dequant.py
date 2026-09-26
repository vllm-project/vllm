# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 -> block-FP8 expert re-encode must be numerically lossless.

The DeepSeek-V4 fp4->fp8 dequant path (``VLLM_DSV4_FP4_DEQUANT=1``) relies on
``cast_mxfp4_to_fp8_block`` being bit-exact against a plain MXFP4 dequant; if it
were lossy the served model would silently lose accuracy.
"""

import pytest
import torch

from vllm.models.deepseek_v4.fp4_dequant import (
    _FP4_LUT,
    cast_mxfp4_to_fp8_block,
)


def _dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    low = (weight & 0x0F).long()
    high = ((weight >> 4) & 0x0F).long()
    vals = torch.stack([_FP4_LUT[low], _FP4_LUT[high]], dim=-1).flatten(1)
    scl = scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return vals * scl


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    bs = 128
    w = weight.float().view(out_dim // bs, bs, in_dim // bs, bs).transpose(1, 2)
    s = scale.float().view(out_dim // bs, in_dim // bs, 1, 1)
    return (w * s).transpose(1, 2).reshape(out_dim, in_dim)


@pytest.mark.parametrize(
    "out_dim,in_dim", [(256, 256), (256, 512), (512, 256), (128, 1024)]
)
def test_cast_mxfp4_to_fp8_block_is_lossless(out_dim: int, in_dim: int):
    torch.manual_seed(0)
    weight = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8)
    # e8m0 exponents around a realistic range (bias 127): 2**-8 .. 2**0.
    scale = torch.randint(127 - 8, 127 + 1, (out_dim, in_dim // 32), dtype=torch.uint8)

    ref = _dequant_mxfp4(weight, scale)
    fp8_w, block_scale = cast_mxfp4_to_fp8_block(weight, scale)
    got = _dequant_fp8_block(fp8_w, block_scale)

    assert fp8_w.dtype == torch.float8_e4m3fn
    assert torch.equal(ref, got)


def test_cast_rejects_unaligned_dims():
    weight = torch.zeros((64, 64), dtype=torch.uint8)  # out=64 not %128
    scale = torch.full((64, 4), 127, dtype=torch.uint8)
    with pytest.raises(ValueError):
        cast_mxfp4_to_fp8_block(weight, scale)
