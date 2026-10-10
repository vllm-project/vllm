# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the MXFP4 read-side unpack against the PyTorch reference.

Chains store -> gather -> dequantize: the Triton store kernel writes the cache,
the Triton gather unpacks it, and the result must equal what the reference
produces. The reference is in turn bit-exact against AITER, so a pass here ties
the whole path to the ecosystem reference.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops import mxfp4_mla as mx


def _on_rocm_gpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _on_rocm_gpu(), reason="mxfp4_mla is ROCm-only")

LATENT = 512
GROUP = 32
ROW = 272
BLOCK_K = 16


def _ops():
    from vllm.v1.attention.ops.mxfp4_mla_read import gather_mxfp4_rows
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla

    return store_mxfp4_mla, gather_mxfp4_rows


def _latent(rows: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, LATENT, generator=g, dtype=torch.float32)
    x[:, ::97] *= 8.0
    return x


def _store_all(x: torch.Tensor):
    store, _ = _ops()
    num = x.shape[0]
    cache = torch.zeros(num, ROW, dtype=torch.uint8, device="cuda")
    store(x.to(torch.bfloat16).cuda(), torch.arange(num).cuda().to(torch.int64), cache)
    return cache


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gather_matches_reference(seed: int):
    num = 64
    x = _latent(num, seed=seed)
    cache = _store_all(x)

    _, gather = _ops()
    slots = torch.arange(num).cuda().to(torch.int64)
    got = gather(cache, slots, LATENT, BLOCK_K).cpu()

    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP)
    assert torch.equal(got.to(torch.float32), want.to(torch.float32))


def test_nibble_order_is_not_transposed():
    """Catch an interleave-vs-concatenate mistake in the unpack.

    If the two nibble streams were concatenated instead of interleaved, the
    latent dims would be permuted (all evens then all odds). A ramp makes that
    immediately visible, where random data would only show a subtle mismatch.
    """
    x = torch.arange(LATENT, dtype=torch.float32).reshape(1, LATENT)
    x = x / x.max() * 6.0  # keep inside E2M1 range at scale 1
    cache = _store_all(x)

    _, gather = _ops()
    got = gather(cache, torch.zeros(BLOCK_K).cuda().to(torch.int64), LATENT, BLOCK_K)
    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP)

    row = got[0].to(torch.float32).cpu()
    assert torch.equal(row, want[0].to(torch.float32))
    # The unpacked row must be non-decreasing, as the input ramp was. A
    # concatenated unpack would zig-zag.
    assert (row[1:] - row[:-1] >= 0).all(), "latent order was permuted"


def test_scattered_slots_gather_the_right_rows():
    num = 64
    x = _latent(num, seed=3)
    cache = _store_all(x)

    _, gather = _ops()
    perm = torch.randperm(num)[: BLOCK_K * 2]
    got = gather(cache, perm.cuda().to(torch.int64), LATENT, BLOCK_K).cpu()

    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP)[perm]
    assert torch.equal(got.to(torch.float32), want.to(torch.float32))


def test_invalid_slots_read_as_zero():
    """Guarded slots must contribute nothing to either dot."""
    num = 32
    x = _latent(num, seed=4)
    cache = _store_all(x)

    _, gather = _ops()
    slots = torch.arange(BLOCK_K).to(torch.int64)
    slots[3] = -1
    slots[7] = 9999
    got = gather(cache, slots.cuda(), LATENT, BLOCK_K).cpu()

    assert (got[3] == 0).all(), "PAD_SLOT_ID row should be zero"
    assert (got[7] == 0).all(), "out-of-range row should be zero"
    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP)
    for i in (0, 1, 2, 4, 5, 6, 8, 15):
        assert torch.equal(got[i].to(torch.float32), want[i].to(torch.float32))


def test_repeated_slots_are_consistent():
    """The indexer can select the same slot twice; both copies must agree."""
    x = _latent(8, seed=5)
    cache = _store_all(x)

    _, gather = _ops()
    slots = torch.full((BLOCK_K,), 3, dtype=torch.int64)
    got = gather(cache, slots.cuda(), LATENT, BLOCK_K).cpu()
    for i in range(1, BLOCK_K):
        assert torch.equal(got[0], got[i])


def test_all_codes_survive_the_round_trip():
    """Exercise every E2M1 code and a spread of exponents, not just random data."""
    values = torch.tensor(mx.E2M1_VALUES, dtype=torch.float32)
    signed = torch.cat([values, -values])  # 16 values = all codes
    row = signed.repeat(LATENT // 16).reshape(1, LATENT)
    # Scale each group differently so the E8M0 field is exercised too.
    scales = torch.exp2(
        torch.arange(LATENT // GROUP, dtype=torch.float32) - 8.0
    ).repeat_interleave(GROUP)
    x = row * scales

    cache = _store_all(x)
    _, gather = _ops()
    got = gather(cache, torch.zeros(BLOCK_K).cuda().to(torch.int64), LATENT, BLOCK_K)

    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP)
    assert torch.equal(got[0].to(torch.float32).cpu(), want[0].to(torch.float32))


def test_dot_against_unpacked_tile_matches_bf16_reference():
    """The score dot's arithmetic: q . k^T over an unpacked tile.

    The read path relies on the unpacked bf16 tile feeding the existing
    ``tl.dot`` unchanged. Check that a dot over the gathered tile equals a dot
    over the reference-dequantized tile.
    """
    num = 64
    x = _latent(num, seed=6)
    cache = _store_all(x)

    _, gather = _ops()
    slots = torch.arange(num).cuda().to(torch.int64)
    tile = gather(cache, slots, LATENT, BLOCK_K)

    g = torch.Generator().manual_seed(11)
    q = torch.randn(16, LATENT, generator=g, dtype=torch.bfloat16).cuda()

    got = (q.to(torch.float32) @ tile.to(torch.float32).T).cpu()
    ref_tile = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP).cuda()
    want = (q.to(torch.float32) @ ref_tile.to(torch.float32).T).cpu()
    assert torch.equal(got, want)
