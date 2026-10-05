# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of mono/fp8_attention.py (offline FP8 block-128 attention weights) at the
TP8 shapes: scale shapes / dtypes, rel-L2 within budget, the kernel's own scale
addressing (unit_fp8 for bk=64, unit_fp8x2 for bk=128) == dequant, pack_fp8 is a
permutation, W_UV's uv_scale_rows == 128, zero blocks."""

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import fp8_attention as F
from vllm.models.deepseek_v32.amd.mono.kernel import packing as P

# TP8 per-rank shapes (kernel/glm/kernel.py GEMV sites)
SHAPES = {
    "qkv_a": (2624, 6144),
    "q_b": (2048, 2048),
    "uk": (8 * 512, 192),
    "uv": (8 * 256, 512),
    "o": (6144, 2048),
}
BK = dict(F.ATTENTION_FP8_BLOCKS)


def _rand(rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    # checkpoint-like: N(0, 1/sqrt(K)) with a few large outlier columns
    w = torch.randn(rows, k, generator=g) / k**0.5
    w[:, torch.randint(0, k, (8,), generator=g)] *= 20
    return w.to(torch.bfloat16)


def _kernel_dequant(q, s, bk, scale_rows):
    """The kernel's per-(16-row group, 64-k chunk) scale lookup."""
    rows, K = q.shape
    flat, out = s.reshape(-1), torch.empty(rows, K)
    for rg in range(rows // 16):
        for kc in range(K // 64):
            if bk == 64:
                idx = (rg * 16 // 128) * (K // 64) + kc
            else:
                idx = (rg * 16 // scale_rows) * (K // 128) + kc // 2
            blk = (slice(rg * 16, rg * 16 + 16), slice(kc * 64, kc * 64 + 64))
            out[blk] = q[blk].float() * flat[idx]
    return out


@pytest.mark.parametrize("name", list(SHAPES))
def test_shapes_error_and_kernel_addressing(name):
    (rows, k), bk = SHAPES[name], BK[name]
    w = _rand(rows, k, 100 + list(SHAPES).index(name))
    q, s = F.quant_fp8_block(w, bk)
    assert q.dtype is torch.float8_e4m3fn and q.shape == (rows, k) and q.is_contiguous()
    assert s.dtype is torch.float32 and s.shape == (-(-rows // 128), k // bk)
    assert torch.isfinite(s).all() and (s > 0).all()
    assert rows % 16 == 0 and k % 64 == 0  # pack_fp8 precondition
    assert torch.isfinite(q.float()).all()  # no NaN from an exact-amax element
    d = F.dequant_fp8_block(q, s, bk)
    rel = float((d - w.float()).norm() / w.float().norm())
    assert 5e-3 <= rel <= 4e-2, rel
    scale_rows = rows // s.shape[0] if name == "uv" else 128
    assert scale_rows == 128
    assert torch.equal(_kernel_dequant(q, s, bk, scale_rows), d)
    # pack_fp8 is view(rows/16, 16, k/64, 2, 4, 8) permuted (0,2,4,1,3,5): invert it
    packed = P.pack_fp8(q).view(rows // 16, k // 64, 4, 16, 2, 8)
    unpacked = packed.permute(0, 3, 1, 4, 2, 5).contiguous().view(rows, k)
    assert torch.equal(unpacked, q.view(torch.uint8))


def test_zero_block_and_dict_api():
    w = _rand(256, 256, 3)
    w[:128, :128] = 0
    q, s = F.quant_fp8_block(w)
    assert s[0, 0] == 1.0 and (q[:128, :128].float() == 0).all()
    t = {f"w_{n}": _rand(r, k, 11 + j) for j, (n, (r, k)) in enumerate(SHAPES.items())}
    errs = F.quantize_attention_fp8(t)
    assert set(errs) == set(SHAPES) and all(1e-3 < e < 4e-2 for e in errs.values())
    assert all(
        t[f"w_{n}"].dtype is torch.float8_e4m3fn and f"s_{n}" in t for n in SHAPES
    )
    with pytest.raises(ValueError):
        F.quantize_attention_fp8(t)  # double quantization
