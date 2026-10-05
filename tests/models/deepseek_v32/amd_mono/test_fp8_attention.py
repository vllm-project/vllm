# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of mono/fp8_attention.py (offline FP8 block-128 attention weights) at the
exact TP8 shapes:

* scale shapes [ceil(rows/128), K/bk], FP32 > 0; weights float8_e4m3fn, rows % 16, K %
  64;
* rel-L2(dequant, W) within budget (5e-3..4e-2);
* the kernel's own scale addressing (unit_fp8 for bk=64, unit_fp8x2 for bk=128)
  reproduces dequant exactly for every (16-row group, 64-k chunk);
* pack_fp8 is a pure permutation (the inverse round-trips);
* W_UV's derived uv_scale_rows == 128; zero blocks get scale 1.0 and zero codes.
"""

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


def _kernel_dequant(q, s, bk, scale_rows=128):
    """Emulate the kernel's per-(16-row group, 64-k chunk) scale lookup (unit_fp8 /
    unit_fp8x2)."""
    rows, K = q.shape
    flat = s.reshape(-1)
    out = torch.empty(rows, K)
    qf = q.float()
    for rg in range(rows // 16):
        for kc in range(K // 64):
            if bk == 64:
                idx = (rg * 16 // 128) * (K // 64) + kc * 64 // 64
            else:
                idx = (rg * 16 // scale_rows) * (K // 128) + kc // 2
            out[rg * 16 : (rg + 1) * 16, kc * 64 : (kc + 1) * 64] = (
                qf[rg * 16 : (rg + 1) * 16, kc * 64 : (kc + 1) * 64] * flat[idx]
            )
    return out


def _unpack_fp8(packed_u8, rows, k):
    """Inverse of packing.pack_fp8 for a 2-D matrix: view(rows/16, 16, k/64, 2, 4, 8)
    permuted (0,2,4,1,3,5)."""
    v = packed_u8.view(rows // 16, k // 64, 4, 16, 2, 8)  # permuted shape
    return v.permute(0, 3, 1, 4, 2, 5).contiguous().view(rows, k)


def test_shapes_error_and_kernel_addressing():
    for i, (name, (rows, k)) in enumerate(SHAPES.items()):
        bk = BK[name]
        w = _rand(rows, k, 100 + i)
        q, s = F.quant_fp8_block(w, 128, bk)
        assert (
            q.dtype is torch.float8_e4m3fn
            and q.shape == (rows, k)
            and q.is_contiguous()
        ), name
        assert s.dtype is torch.float32 and s.shape == (-(-rows // 128), k // bk), (
            name,
            s.shape,
        )
        assert torch.isfinite(s).all() and (s > 0).all(), name
        assert rows % 16 == 0 and k % 64 == 0, name  # pack_fp8 precondition
        d = F.dequant_fp8_block(q, s, bk)
        rel = float((d - w.float()).norm() / w.float().norm())
        assert 5e-3 <= rel <= 4e-2, (name, rel)
        # no NaN from an exact-amax element
        assert torch.isfinite(q.float()).all(), name
        kd = _kernel_dequant(
            q, s, bk, scale_rows=rows // s.shape[0] if name == "uv" else 128
        )
        assert torch.equal(kd, d), f"{name}: kernel scale addressing != dequant"
        packed = P.pack_fp8(q)
        assert packed.numel() == rows * k
        assert torch.equal(_unpack_fp8(packed, rows, k), q.view(torch.uint8)), (
            f"{name}: pack_fp8 not invertible"
        )


def test_uv_scale_rows():
    rows, k = SHAPES["uv"]
    _, s = F.quant_fp8_block(_rand(rows, k, 7), 128, BK["uv"])
    assert rows // s.shape[0] == 128


def test_zero_block_and_dict_api():
    w = _rand(256, 256, 3)
    w[:128, :128] = 0
    q, s = F.quant_fp8_block(w, 128, 128)
    assert s[0, 0] == 1.0 and (q[:128, :128].float() == 0).all()
    t = {f"w_{n}": _rand(r, k, 11 + j) for j, (n, (r, k)) in enumerate(SHAPES.items())}
    errs = F.quantize_attention_fp8(t)
    assert set(errs) == set(SHAPES) and all(1e-3 < e < 4e-2 for e in errs.values()), (
        errs
    )
    for n in SHAPES:
        assert t[f"w_{n}"].dtype is torch.float8_e4m3fn and f"s_{n}" in t
    with pytest.raises(ValueError):
        F.quantize_attention_fp8(t)  # double quantization
