# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDNA head-first grid and vec4 stores must not change GDN outputs."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_o, wy_fast
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.triton_utils import triton

if not current_platform.is_cuda_alike():
    pytest.skip("requires a GPU", allow_module_level=True)

BT = 64
H, HG, K, V = 8, 4, 128, 128


def _inputs(T, dtype, cu_seqlens):
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, device="cuda", generator=gen) * scale
        return x.to(dtype)

    nt = (
        triton.cdiv(T, BT)
        if cu_seqlens is None
        else len(prepare_chunk_indices(cu_seqlens, BT))
    )
    k = randn(1, T, HG, K, scale=0.1)
    q = randn(1, T, HG, K, scale=0.1)
    v = randn(1, T, H, V)
    beta = torch.rand(1, T, H, device="cuda", generator=gen).to(dtype)
    g = -torch.rand(1, T, H, device="cuda", generator=gen).cumsum(1) * 0.05
    A = randn(1, T, H, BT, scale=0.1)
    h = randn(1, nt, H, V, K, scale=0.1)
    return q, k, v, beta, g, A, h


# Different num_warps/BK change the reduction order, so both runs must use the
# same config for a bitwise comparison.
PINNED = [
    (wy_fast.recompute_w_u_fwd_kernel, triton.Config({}, num_warps=4, num_stages=1)),
    (
        chunk_o.chunk_fwd_kernel_o,
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
    ),
]


def _run_both(monkeypatch, fn):
    for kernel, cfg in PINNED:
        monkeypatch.setattr(kernel.fn, "configs", [cfg])
        monkeypatch.setattr(kernel.fn, "cache", {})
    outs = []
    for rdna in (False, True):
        monkeypatch.setattr(wy_fast, "is_rdna", rdna)
        monkeypatch.setattr(chunk_o, "is_rdna", rdna)
        outs.append(fn())
    return outs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("T,cu", [(1, None), (65, None), (300, [0, 1, 64, 300])])
def test_recompute_w_u_layout_invariant(monkeypatch, dtype, T, cu):
    cu_seqlens = None if cu is None else torch.tensor(cu, device="cuda").int()
    q, k, v, beta, g, A, h = _inputs(T, dtype, cu_seqlens)
    ref, rdna = _run_both(
        monkeypatch,
        lambda: wy_fast.recompute_w_u_fwd(k, v, beta, g, A, cu_seqlens),
    )
    for x, y in zip(ref, rdna):
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T,cu", [(1, None), (65, None), (300, [0, 1, 64, 300])])
@pytest.mark.parametrize("out", ["none", "same", "fp32", "misaligned"])
def test_chunk_fwd_o_layout_invariant(monkeypatch, dtype, T, cu, out):
    cu_seqlens = None if cu is None else torch.tensor(cu, device="cuda").int()
    q, k, v, beta, g, A, h = _inputs(T, dtype, cu_seqlens)

    def run():
        core_attn_out = None
        if out == "same":
            core_attn_out = torch.full((v.numel(),), float("nan"), device="cuda")
            core_attn_out = core_attn_out.to(v.dtype)
        elif out == "fp32":
            core_attn_out = torch.full(
                (v.numel(),), float("nan"), device="cuda", dtype=torch.float32
            )
        elif out == "misaligned":
            core_attn_out = torch.full(
                (v.numel() + 1,), float("nan"), device="cuda", dtype=v.dtype
            )[1:]
        return chunk_o.chunk_fwd_o(
            q, k, v, h, g, cu_seqlens=cu_seqlens, core_attn_out=core_attn_out
        )

    ref, rdna = _run_both(monkeypatch, run)
    assert torch.isfinite(rdna).all()
    torch.testing.assert_close(rdna, ref, rtol=0, atol=0)
