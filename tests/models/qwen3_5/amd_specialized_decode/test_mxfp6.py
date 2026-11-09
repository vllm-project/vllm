# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host side of the MXFP6 E2M3 projections (``runtime/mxfp6.py``): the
rounding, against an E2M3 table, and the packing that the kernels read,
against an unpacker written from the layout's description. Runs on the CPU."""

import pytest
import torch

from vllm.models.qwen3_5.amd.specialized_decode.runtime import mxfp6
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the Qwen3.8 specialized decode is ROCm only",
)

# The 32 E2M3 magnitudes in code order.
E2M3 = [i / 8 for i in range(16)] + [2 + i / 4 for i in range(8)]
E2M3 += [4 + i / 2 for i in range(8)]


def _weight(n: int, k: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(n + k)
    w = torch.randn(n, k, generator=g) * 0.02
    # A few blocks of other magnitudes, an all-zero block and exact ties.
    w[0, :32] = 0.0
    w[1, :32] *= 1000.0
    w[2, 32:64] = torch.linspace(-1.0, 1.0, 32)
    return w.to(torch.bfloat16)


def test_round_e2m3_values():
    """Every rounded value is an E2M3 magnitude times its block's scale, no
    block's maximum exceeds 7.5 times the scale, and rounding twice changes
    nothing."""
    w = _weight(64, 512)
    r = mxfp6.round_e2m3(w)
    assert r.dtype == w.dtype and r.shape == w.shape
    v = r.float().view(64, 16, 32)
    amax = w.float().view(64, 16, 32).abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x80000) & 0x7F800000
    scale = torch.exp2(((bits >> 23) - 129).clamp(-126, 127).float())
    table = torch.tensor(E2M3)
    q = v.abs() / scale
    assert (q.unsqueeze(-1) == table).any(-1).all()
    assert torch.equal(mxfp6.round_e2m3(r), r)
    assert torch.equal(r[0, :32], torch.zeros(32, dtype=r.dtype))


def _unpack(tiles: torch.Tensor, scales: torch.Tensor, n: int, k: int):
    """fp6_pack's layout read back, as its docstring describes it: the tile
    of row group rg and K step st at byte (rg (K / 128) + st) 1536, lane l =
    16 g + r holding row 16 rg + r at K 128 st + 32 g + i as 6-bit code i of
    24 bytes, 16 of them at byte 16 l and 8 at byte 1024 + 8 l; the scale at
    byte (st % 4) of word (rg (K / 512) + st / 4) 64 + l."""
    out = torch.empty(n, k)
    t = tiles.view(n // 16, k // 128, 1536)
    sc = scales.view(n // 16, k // 512, 64, 4)
    table = torch.tensor(E2M3)
    for rg in range(n // 16):
        for st in range(k // 128):
            for lane in range(64):
                g, r = lane // 16, lane % 16
                b = torch.cat(
                    (
                        t[rg, st, 16 * lane : 16 * lane + 16],
                        t[rg, st, 1024 + 8 * lane : 1024 + 8 * lane + 8],
                    )
                ).long()
                bits = sum(int(b[j]) << (8 * j) for j in range(24))
                codes = [(bits >> (6 * i)) & 0x3F for i in range(32)]
                e = int(sc[rg, st // 4, lane, st % 4]) - 127
                vals = [
                    (-1.0 if c & 0x20 else 1.0) * float(table[c & 0x1F]) * 2.0**e
                    for c in codes
                ]
                k0 = 128 * st + 32 * g
                out[16 * rg + r, k0 : k0 + 32] = torch.tensor(vals)
    return out


def test_fp6_pack_layout():
    """fp6_pack holds exactly the values of round_e2m3 in its layout."""
    n, k = 32, 1024
    w = _weight(n, k)
    tiles, scales = mxfp6.fp6_pack(w)
    assert tiles.numel() == n * k * 3 // 4 and scales.numel() == n * k // 32
    got = _unpack(tiles, scales, n, k)
    assert torch.equal(got, mxfp6.round_e2m3(w).float())


def test_projections_regex():
    hits = [
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.3.linear_attn.out_proj",
        "model.layers.3.self_attn.qkv_proj",
        "model.layers.71.self_attn.o_proj",
    ]
    misses = [
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.shared_expert.down_proj",
        "lm_head",
    ]
    assert all(mxfp6.PROJ_RE.search(n) for n in hits)
    assert not any(mxfp6.PROJ_RE.search(n) for n in misses)
