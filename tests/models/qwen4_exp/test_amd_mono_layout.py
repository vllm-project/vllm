# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash-Next mono kernels' weight addresses against aiter's own shuffles at
TP2: every fp4 byte and e8m0 scale of a routed expert (the intermediate
padded from RI to RIP) and of an online-MXFP4 linear, as the backends leave
them after loading, is where ``layout`` says. Host only."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the AITER weight shuffles are ROCm only", allow_module_level=True)

import aiter.ops.shuffle as shuffle  # noqa: E402
import aiter.utility.fp4_utils as fp4_utils  # noqa: E402

from vllm.models.qwen4_exp.amd.mono import layout as L  # noqa: E402

NE = 2  # experts: enough to catch a wrong expert stride
LAY = L.Layout(2)


def _shuffled(w: torch.Tensor) -> torch.Tensor:
    """``w`` (E, N, K / 2) uint8 through the backend's weight shuffle, as dwords."""
    out = shuffle.shuffle_weight(w.view(torch.float4_e2m1fn_x2), layout=(16, 16))
    return out.view(torch.uint8).reshape(-1).view(torch.int32)


def _check_weight(w: torch.Tensor, base_of, k: int) -> None:
    flat = _shuffled(w)
    groups = w.shape[1] // L.ROWS
    lane = torch.arange(64)
    for e in range(w.shape[0]):
        for rg in range(groups):
            rows = w[e, rg * L.ROWS + lane % 16]  # (64, K / 2)
            for st in range(k // L.FP4_STEP):
                at = L.fp4_tile_dword(base_of(e), rg, k, st, lane * 0) + lane * 4
                got = torch.stack([flat[at + q] for q in range(4)], 1)
                b0 = st * 64 + (lane // 16) * 16  # first byte of the lane's 32 fp4
                want = torch.stack(
                    [rows[i, b0[i] : b0[i] + 16] for i in range(64)]
                ).view(torch.int32)
                assert torch.equal(got, want), (e, rg, st)


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)


def test_shapes():
    assert (LAY.NK, LAY.NV, LAY.QKVZ, LAY.BA, LAY.CONV, LAY.CORE) == (
        8,
        24,
        8192,
        48,
        5120,
        3072,
    )
    assert (LAY.RI, LAY.RIP) == (320, 384)


def test_w13_tiles():
    w = _rand(NE, 2 * LAY.RIP, L.HIDDEN // 2, seed=0)
    _check_weight(w, lambda e: L.w13_base(LAY, e), L.HIDDEN)
    # gate group g and up group g are rows 16 g .. and RIP + 16 g ..
    assert L.w13_group(LAY, 0, 3) * L.ROWS == 48
    assert L.w13_group(LAY, 1, 3) * L.ROWS == LAY.RIP + 48


def test_w2_tiles():
    w = _rand(NE, L.HIDDEN, LAY.RIP // 2, seed=1)
    _check_weight(w, lambda e: L.w2_base(LAY, e), LAY.RIP)


def test_linear_tiles():
    """An online-MXFP4 linear (in_proj_qkvz: K = H; out_proj: K = CORE) is one
    expert of the same layout."""
    for n, k in ((LAY.QKVZ // 16, L.HIDDEN), (L.HIDDEN // 16, LAY.CORE)):
        _check_weight(_rand(1, n, k // 2, seed=k), lambda e: 0, k)


@pytest.mark.parametrize(
    "rows,k,index",
    [
        (2 * LAY.RIP, L.HIDDEN, lambda e, n, kb: L.w13_scale_index(LAY, e, n, kb)),
        (L.HIDDEN, LAY.RIP, lambda e, n, kb: L.w2_scale_index(LAY, e, n, kb)),
    ],
)
def test_scales(rows, k, index):
    """``e8m0_shuffle`` pads K / 32 to a multiple of 8 (w2's 12 blocks to 16)."""
    s = _rand(NE, rows, k // 32, seed=2)
    flat = fp4_utils.e8m0_shuffle(s.view(NE * rows, -1)).reshape(-1)
    assert flat.numel() == NE * rows * L.scale_cols(k)
    n = torch.arange(rows)[:, None]
    kb = torch.arange(k // 32)[None, :]
    for e in range(NE):
        assert torch.equal(flat[index(e, n, kb)], s[e]), e


def test_linear_scales():
    """The online-MXFP4 linears' scales (``AiterMxfp4LinearKernel``'s swizzle)
    at ``scale_index`` with ``scale_cols`` columns, as ``fp4_gemv_loads``
    reads them."""
    for n, k in ((LAY.QKVZ, L.HIDDEN), (L.HIDDEN, LAY.CORE)):
        s = _rand(n, k // 32, seed=n)
        sm, sn = s.shape
        ws = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)
        flat = ws.contiguous().view(-1)
        rows = torch.arange(n)[:, None]
        kb = torch.arange(k // 32)[None, :]
        assert torch.equal(flat[L.scale_index(rows, kb, L.scale_cols(k))], s)
