# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-side weight copies of the DeepSeek-V4.1 gfx942 mono kernels
(vllm/models/deepseek_v41/amd/mono/weights942.py), in torch on the CPU.

  linears  linear_copy and ocp_from_dequant on weights that vLLM's MXFP8
           emulation dequantized at load, and dense_copy's FNUZ bytes,
           codes, K padding and tile-major order.
  experts  moe_copies of vLLM's plain MXFP4 expert tensors: gate/up
           interleave, the padding strip, -0 nibbles, fp4_tile_major and
           codes_tile_rounds.

The layouts are checked against index maps built from the documented
formats, so every byte has to land where the GPU kernels read it.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.deepseek_v41.amd.mono.weights942 import (
    dense_copy,
    fp4_drop_negative_zero_,
    linear_copy,
    moe_copies,
    ocp_from_dequant,
)

# The routed experts' intermediate size at tensor parallel size 4, and the
# size vLLM pads it to.
INTER, INTER_PAD = 576, 640


def _e4m3_bytes(g, n, k):
    """Random OCP e4m3 bytes, -0 (0x80) included, without the NaNs."""
    b = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8)
    nan = (b & 0x7F) == 0x7F
    return torch.where(nan, b ^ 0x01, b)


def _block_codes(g, n, k):
    return torch.randint(110, 140, (n // 32, k // 32), generator=g).to(torch.uint8)


def _remap(t, dst):
    """Scatters each 16-row tile of t: element (r, c) of a tile goes to flat
    offset dst[r, c] of the tile."""
    assert torch.equal(dst.flatten().sort().values, torch.arange(dst.numel()))
    src = t.reshape(-1, dst.numel())
    out = torch.empty_like(src)
    out[:, dst.flatten()] = src
    return out.view(t.shape)


def _dense_map(kp, chunk_q):
    """Lane 16 j + r of a 128-byte step s takes row r's 8-byte chunks at
    K 128 s + 32 b + 8 j, b = 0 .. 3. A step stores the 64 lanes' blocks 0
    and 1, then their blocks 2 and 3, 16 bytes a lane."""
    r = torch.arange(16)[:, None]
    c = torch.arange(kp)[None, :]
    s, b, j, e = c // 128, c % 128 // 32, c % 32 // 8, c % 8
    if chunk_q:
        e = torch.where(e % 2 == 0, e // 2, 4 + e // 2)
    return s * 2048 + b // 2 * 1024 + (16 * j + r) * 16 + b % 2 * 8 + e


def _fp4_map(kb):
    """Lane 16 j + r of a 64-byte step s takes row r's dwords at bytes
    64 s + 16 i + 4 j, i = 0 .. 3, 16 bytes a lane. A last 32-byte step has
    i = 0, 1 and 8 bytes a lane."""
    full = kb // 64 * 64
    r = torch.arange(16)[:, None]
    c = torch.arange(kb)[None, :]
    head = c // 64 * 1024 + (16 * (c % 16 // 4) + r) * 16 + c % 64 // 16 * 4 + c % 4
    t = c - full
    tail = 16 * full + (16 * (t % 16 // 4) + r) * 8 + t // 16 * 4 + t % 4
    return torch.where(c < full, head, tail)


def _codes_map(kc, blocks=8):
    r = torch.arange(16)[:, None]
    c = torch.arange(kc)[None, :]
    return c // blocks * 16 * blocks + r * blocks + c % blocks


def _drop_negative_zero(b):
    lo, hi = b & 0x0F, b >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    return hi << 4 | lo


@pytest.mark.parametrize("k", [576, 1280])
@pytest.mark.parametrize("chunk_q", [False, True])
def test_dense_copy_layout(k, chunk_q):
    g = torch.Generator().manual_seed(k + chunk_q)
    n = 64
    w8, codes = _e4m3_bytes(g, n, k), _block_codes(g, n, k)
    kp = -(-k // 128) * 128
    got, got_codes = dense_copy(w8, codes, chunk_q=chunk_q)

    fnuz = torch.where(w8 == 0x80, torch.zeros_like(w8), w8)
    want = torch.zeros(n, kp, dtype=torch.uint8)
    want[:, :k] = fnuz
    assert torch.equal(got, _remap(want, _dense_map(kp, chunk_q)))
    want_codes = torch.full((n // 32, kp // 32), 128, dtype=torch.uint8)
    want_codes[:, : k // 32] = codes + 1
    assert torch.equal(got_codes, want_codes)


def test_linear_copy_of_vllm_dequant():
    g = torch.Generator().manual_seed(0)
    n, k = 64, 576
    w8, codes = _e4m3_bytes(g, n, k), _block_codes(g, n, k)
    # vLLM repeats each 32 x 32 block's code over its rows and dequantizes
    # the e4m3 weight to bf16 at load on gfx942.
    row_codes = codes.repeat_interleave(32, 0)
    w = dequant_mxfp8_to_bf16(w8.view(torch.float8_e4m3fn), row_codes)

    assert torch.equal(ocp_from_dequant(w, codes), w8)
    want = dense_copy(w8, codes)
    for args in ((w, row_codes), (w, codes), (w8.view(torch.float8_e4m3fn), codes)):
        got = linear_copy(*args)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_linear_copy_refuses_inexact_weights():
    g = torch.Generator().manual_seed(1)
    n, k = 64, 576
    w8, codes = _e4m3_bytes(g, n, k), _block_codes(g, n, k)
    row_codes = codes.repeat_interleave(32, 0)
    w = dequant_mxfp8_to_bf16(w8.view(torch.float8_e4m3fn), row_codes)

    # A bf16 value between two e4m3 values of its block.
    bad = w.clone()
    bad[3, 7] = 1.0625 * 2.0 ** (int(codes[0, 0]) - 127)
    with pytest.raises(ValueError, match="1 of 36864 values are not e4m3"):
        linear_copy(bad, row_codes)
    mixed = row_codes.clone()
    mixed[33, 2] += 1
    with pytest.raises(ValueError, match="not 32 x 32 blocks"):
        linear_copy(w, mixed)
    with pytest.raises(TypeError, match="float16"):
        linear_copy(w.half(), codes)


def test_moe_copies_layout():
    g = torch.Generator().manual_seed(2)
    e, hidden = 2, 512
    kh = hidden // 2
    w13 = torch.randint(0, 256, (e, 2 * INTER_PAD, kh), generator=g, dtype=torch.uint8)
    w13_s = torch.randint(110, 140, (e, 2 * INTER_PAD, hidden // 32), generator=g)
    w2 = torch.randint(0, 256, (e, hidden, INTER_PAD // 2), generator=g).to(torch.uint8)
    w2_s = torch.randint(110, 140, (e, hidden, INTER_PAD // 32), generator=g)
    w13[:, INTER:INTER_PAD] = 0
    w13[:, INTER_PAD + INTER :] = 0
    w2[:, :, INTER // 2 :] = 0
    w13_s, w2_s = w13_s.to(torch.uint8), w2_s.to(torch.uint8)
    inputs = [t.clone() for t in (w13, w13_s, w2, w2_s)]

    got = moe_copies(w13, w13_s, w2, w2_s, INTER)
    for t, before in zip((w13, w13_s, w2, w2_s), inputs):
        assert torch.equal(t, before)

    def interleave(t):
        out = torch.empty(e, 2 * INTER, t.shape[-1], dtype=t.dtype)
        out[:, 0::2] = t[:, :INTER]
        out[:, 1::2] = t[:, INTER_PAD : INTER_PAD + INTER]
        return out

    up_gate = _drop_negative_zero(interleave(w13))
    down = _drop_negative_zero(w2[:, :, : INTER // 2])
    assert torch.equal(got[0], _remap(up_gate, _fp4_map(kh)))
    assert torch.equal(got[1], _remap(interleave(w13_s), _codes_map(hidden // 32)))
    assert torch.equal(got[2], _remap(down, _fp4_map(INTER // 2)))
    assert torch.equal(got[3], w2_s[:, :, : INTER // 32])


@pytest.mark.parametrize("where", ["gate", "up", "down"])
def test_moe_copies_refuses_nonzero_padding(where):
    e, hidden = 1, 256
    w13 = torch.zeros(e, 2 * INTER_PAD, hidden // 2, dtype=torch.uint8)
    w13_s = torch.zeros(e, 2 * INTER_PAD, hidden // 32, dtype=torch.uint8)
    w2 = torch.zeros(e, hidden, INTER_PAD // 2, dtype=torch.uint8)
    w2_s = torch.zeros(e, hidden, INTER_PAD // 32, dtype=torch.uint8)
    if where == "gate":
        w13[0, INTER_PAD - 1, 5] = 0x10
    elif where == "up":
        w13[0, 2 * INTER_PAD - 1, 5] = 0x01
    else:
        w2[0, 9, INTER_PAD // 2 - 1] = 0x01
    with pytest.raises(AssertionError, match="padding is not zero"):
        moe_copies(w13, w13_s, w2, w2_s, INTER)


def test_fp4_drop_negative_zero_count():
    b = torch.arange(256, dtype=torch.uint8)
    want = _drop_negative_zero(b)
    # 16 bytes have a low -0 nibble and 16 a high one.
    assert fp4_drop_negative_zero_(b) == 32
    assert torch.equal(b, want)
