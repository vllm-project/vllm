# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer E4M3 helpers (TurboQuant FP8 keys below sm_89) vs torch.float8_e4m3fn."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="needs a CUDA device for Triton"
)


def _helpers():
    from vllm.v1.attention.ops.triton_turboquant_decode import _e4m3_decode
    from vllm.v1.attention.ops.triton_turboquant_store import _e4m3_encode

    return _e4m3_encode, _e4m3_decode


def _kernels():
    encode, decode = _helpers()

    @triton.jit
    def enc_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        tl.store(out_ptr + o, encode(tl.load(x_ptr + o, mask=m, other=0.0)), mask=m)

    @triton.jit
    def dec_kernel(b_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        tl.store(out_ptr + o, decode(tl.load(b_ptr + o, mask=m, other=0)), mask=m)

    return enc_kernel, dec_kernel


def _run(kernel, src, out_dtype):
    out = torch.empty(src.shape, dtype=out_dtype, device=src.device)
    kernel[(triton.cdiv(src.numel(), 1024),)](src, out, src.numel(), BLOCK=1024)
    return out


def test_decode_matches_torch_for_every_byte():
    _, dec = _kernels()
    b = torch.arange(256, dtype=torch.uint8, device="cuda")
    got = _run(dec, b, torch.float32)
    want = b.view(torch.float8_e4m3fn).to(torch.float32)
    ok = ~torch.isnan(want)  # 0x7f / 0xff are NaN in E4M3; keys never hold them
    assert torch.equal(got[ok], want[ok])


def test_encode_matches_torch_rounding_and_saturates():
    enc, _ = _kernels()
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.logspace(-4, 3, 4096, device="cuda")
    x = torch.randn(1 << 20, device="cuda", generator=g) * scale.repeat(256)[: 1 << 20]
    # ties (midpoints of neighbouring codes), the +-448 edge and beyond, tiny values
    codes = (
        torch.arange(127, dtype=torch.uint8, device="cuda")
        .view(torch.float8_e4m3fn)
        .float()
    )
    mids = (codes[1:] + codes[:-1]) / 2
    edge = torch.cat(
        [
            mids,
            -mids,
            codes,
            -codes,
            torch.tensor(
                [448.0, 449.0, 1000.0, 1e9, -448.0, -1e9, 0.0, -0.0, 1e-9, -1e-9],
                device="cuda",
            ),
        ]
    )
    x = torch.cat([x, edge])
    got = _run(enc, x, torch.uint8)
    want = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(got, want)
