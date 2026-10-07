# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hardware unpack must equal the software unpack, bit for bit.

The software path is already bit-exact against AITER, so agreement here chains
the hardware converter to the ecosystem reference. A ramp fixture is included
because an ordering mistake in the interleave tree perturbs values without
changing their multiset -- random data would show a subtle mismatch where a
monotonic ramp shows an obvious one.
"""

from __future__ import annotations

import pytest
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops import mxfp4_mla as mx
from vllm.v1.attention.ops.mxfp4_mla_read import (
    unpack_mxfp4_tile,
    unpack_mxfp4_tile_hw,
)


def _on_gfx950() -> bool:
    from vllm.platforms import current_platform

    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 FP4 converts")

BK, L, G = 16, 512, 32


@triton.jit
def _sw_kernel(p_ptr, s_ptr, o_ptr, BK: tl.constexpr, L: tl.constexpr, G: tl.constexpr):
    r = tl.arange(0, BK)[:, None]
    packed = tl.load(p_ptr + r * (L // 2) + tl.arange(0, L // 2)[None, :])
    sc = tl.load(s_ptr + r * (L // G) + tl.arange(0, L // G)[None, :])
    tl.store(
        o_ptr + r * L + tl.arange(0, L)[None, :],
        unpack_mxfp4_tile(packed, sc, BK, L, G),
    )


@triton.jit
def _hw_kernel(p_ptr, s_ptr, o_ptr, BK: tl.constexpr, L: tl.constexpr, G: tl.constexpr):
    r = tl.arange(0, BK)[:, None]
    words = tl.load(p_ptr + r * (L // 8) + tl.arange(0, L // 8)[None, :])
    sc = tl.load(s_ptr + r * (L // G) + tl.arange(0, L // G)[None, :])
    tl.store(
        o_ptr + r * L + tl.arange(0, L)[None, :],
        unpack_mxfp4_tile_hw(words, sc, BK, L, G),
    )


def _run(x: torch.Tensor):
    packed, scales = mx.quantize_pack(x, G)
    packed_d = packed.cuda().contiguous()
    scales_d = scales.cuda().contiguous()
    words_d = packed_d.view(torch.int32)

    sw = torch.empty(BK, L, dtype=torch.bfloat16, device="cuda")
    hw = torch.empty(BK, L, dtype=torch.bfloat16, device="cuda")
    _sw_kernel[(1,)](packed_d, scales_d, sw, BK=BK, L=L, G=G)
    _hw_kernel[(1,)](words_d, scales_d, hw, BK=BK, L=L, G=G)
    torch.accelerator.synchronize()

    ref = mx.unpack_dequantize(packed, scales, G, out_dtype=torch.float32)
    return sw.float().cpu(), hw.float().cpu(), ref


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hw_matches_software_and_reference(seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BK, L, generator=g, dtype=torch.float32)
    x[:, ::97] *= 8.0
    sw, hw, ref = _run(x)

    assert torch.equal(sw, ref), "software path drifted from the reference"
    assert torch.equal(hw, ref), "hardware path disagrees with the reference"
    assert torch.equal(hw, sw)


def test_ramp_catches_an_ordering_mistake():
    """A monotonic ramp makes a wrong interleave tree obvious."""
    x = torch.arange(L, dtype=torch.float32).repeat(BK, 1)
    x = x / x.max() * 6.0
    sw, hw, ref = _run(x)

    assert torch.equal(hw, ref), "hardware path permuted the latent axis"
    row = hw[0]
    assert (row[1:] - row[:-1] >= 0).all(), (
        "hardware unpack produced a non-monotonic row from a ramp: the "
        "interleave tree is assembling values in the wrong order"
    )


def test_every_code_and_a_spread_of_exponents():
    values = torch.tensor(mx.E2M1_VALUES, dtype=torch.float32)
    signed = torch.cat([values, -values])
    row = signed.repeat(L // 16)
    scales = torch.exp2(
        torch.arange(L // G, dtype=torch.float32) - 8.0
    ).repeat_interleave(G)
    x = (row * scales).repeat(BK, 1)

    _, hw, ref = _run(x)
    assert torch.equal(hw, ref)


def test_zero_rows_stay_zero():
    x = torch.zeros(BK, L, dtype=torch.float32)
    _, hw, ref = _run(x)
    assert torch.equal(hw, ref)
    assert (hw == 0).all()
