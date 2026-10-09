# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8 mono kernels' weight addresses against aiter's own shuffles: every
fp4 byte and e8m0 scale of a routed expert, as ``AITER_MXFP4_MXFP4`` leaves them
after loading, is where ``layout`` says. Host only."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the AITER weight shuffles are ROCm only", allow_module_level=True)

import aiter.ops.shuffle as shuffle  # noqa: E402
import aiter.utility.fp4_utils as fp4_utils  # noqa: E402

from vllm.models.qwen3_5.amd.mono import layout as L  # noqa: E402

NE = 2  # experts: enough to catch a wrong expert stride


def _shuffled(w: torch.Tensor) -> torch.Tensor:
    """``w`` (E, N, K / 2) uint8 through the backend's weight shuffle, as dwords."""
    out = shuffle.shuffle_weight(w.view(torch.float4_e2m1fn_x2), layout=(16, 16))
    return out.view(torch.uint8).reshape(-1).view(torch.int32)


def _check_weight(w: torch.Tensor, base_of, k: int) -> None:
    flat = _shuffled(w)
    groups = w.shape[1] // L.ROWS
    lane = torch.arange(64)
    for e in range(NE):
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


def test_w13_tiles():
    g = torch.Generator().manual_seed(0)
    w = torch.randint(
        0, 256, (NE, 2 * L.RI, L.HIDDEN // 2), dtype=torch.uint8, generator=g
    )
    _check_weight(w, L.w13_base, L.HIDDEN)
    # gate group g and up group g are rows 16 g .. and RI + 16 g ..
    assert L.w13_group(0, 3) * L.ROWS == 48
    assert L.w13_group(1, 3) * L.ROWS == L.RI + 48


def test_w2_tiles():
    g = torch.Generator().manual_seed(1)
    w = torch.randint(0, 256, (NE, L.HIDDEN, L.RI // 2), dtype=torch.uint8, generator=g)
    _check_weight(w, L.w2_base, L.RI)


@pytest.mark.parametrize(
    "rows,k,index",
    [
        (2 * L.RI, L.HIDDEN, L.w13_scale_index),
        (L.HIDDEN, L.RI, L.w2_scale_index),
    ],
)
def test_scales(rows, k, index):
    g = torch.Generator().manual_seed(2)
    s = torch.randint(0, 256, (NE, rows, k // 32), dtype=torch.uint8, generator=g)
    flat = fp4_utils.e8m0_shuffle(s.view(NE * rows, -1)).reshape(-1)
    assert flat.numel() == s.numel()
    n = torch.arange(rows)[:, None]
    kb = torch.arange(k // 32)[None, :]
    for e in range(NE):
        assert torch.equal(flat[index(e, n, kb)], s[e]), e
