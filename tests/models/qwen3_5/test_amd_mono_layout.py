# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen mono kernels' weight addresses against aiter's own shuffles, at
Qwen3.8's TP8 shapes: every fp4 byte and e8m0 scale of a routed expert and
of an online-MXFP4 linear, as the backends leave them after loading, is where
``layout`` says. Host only."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the AITER weight shuffles are ROCm only", allow_module_level=True)

import aiter.ops.shuffle as shuffle  # noqa: E402
import aiter.utility.fp4_utils as fp4_utils  # noqa: E402

from vllm.models.qwen3_5.amd.mono import layout as L  # noqa: E402

NE = 2  # experts: enough to catch a wrong expert stride
MOES = {"qwen38": L.MOE}
# the online-MXFP4 linears (rows, K): in_proj_qkvz and out_proj / o_proj
LINEARS = {
    "qwen38": ((L.QKVZ, L.HIDDEN), (L.HIDDEN, L.CORE)),
}


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


@pytest.mark.parametrize("model", MOES)
def test_w13_tiles(model):
    d = MOES[model]
    w = _rand(NE, 2 * d.rip, d.hidden // 2, seed=0)
    _check_weight(w, lambda e: L.w13_base(d, e), d.hidden)
    # gate group g and up group g are rows 16 g .. and RIP + 16 g ..
    assert L.w13_group(d, 0, 3) * L.ROWS == 48
    assert L.w13_group(d, 1, 3) * L.ROWS == d.rip + 48


@pytest.mark.parametrize("model", MOES)
def test_w2_tiles(model):
    d = MOES[model]
    w = _rand(NE, d.hidden, d.rip // 2, seed=1)
    _check_weight(w, lambda e: L.w2_base(d, e), d.rip)


@pytest.mark.parametrize("model", LINEARS)
def test_linear_tiles(model):
    """An online-MXFP4 linear is one expert of the same layout (its first 16
    row groups: enough for the row-group stride)."""
    for n, k in LINEARS[model]:
        _check_weight(_rand(1, min(n, 256), k // 2, seed=k), lambda e: 0, k)


@pytest.mark.parametrize("model", MOES)
@pytest.mark.parametrize("which", ("w13", "w2"))
def test_scales(model, which):
    """``e8m0_shuffle`` pads K / 32 to a multiple of 8 (Flash-Next w2's 12
    blocks to 16)."""
    d = MOES[model]
    if which == "w13":
        rows, k = 2 * d.rip, d.hidden
        index = lambda e, n, kb: L.w13_scale_index(d, e, n, kb)  # noqa: E731
    else:
        rows, k = d.hidden, d.rip
        index = lambda e, n, kb: L.w2_scale_index(d, e, n, kb)  # noqa: E731
    s = _rand(NE, rows, k // 32, seed=2)
    flat = fp4_utils.e8m0_shuffle(s.view(NE * rows, -1)).reshape(-1)
    assert flat.numel() == NE * rows * L.scale_cols(k)
    n = torch.arange(rows)[:, None]
    kb = torch.arange(k // 32)[None, :]
    for e in range(NE):
        assert torch.equal(flat[index(e, n, kb)], s[e]), e


@pytest.mark.parametrize("model", LINEARS)
def test_linear_scales(model):
    """The online-MXFP4 linears' scales (``AiterMxfp4LinearKernel``'s swizzle)
    at ``scale_index`` with ``scale_cols`` columns, as ``fp4_gemv_loads``
    reads them."""
    for n, k in LINEARS[model]:
        s = _rand(n, k // 32, seed=n)
        sm, sn = s.shape
        ws = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)
        flat = ws.contiguous().view(-1)
        rows = torch.arange(n)[:, None]
        kb = torch.arange(k // 32)[None, :]
        assert torch.equal(flat[L.scale_index(rows, kb, L.scale_cols(k))], s)
