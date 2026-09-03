# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm MiniMax-M3 lightning-indexer kernels.

``test_minimax_m3.py`` covers the platform-common kernels; these cover
``vllm/models/minimax_m3/amd/ops/index_topk.py``, which ``common/indexer.py``
dispatches to on ROCm.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton

if not current_platform.is_rocm():
    pytest.skip("ROCm-only indexer kernels", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    TOPK_QUERY_TILE,
    TOPK_QUERY_TILE_SMALL,
    _min_topk_programs,
    _topk_query_tile,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

HEAD_DIM = 128
DEV = "cuda"


def _build(q_lens, prefix_lens, num_idx_heads, cache_dtype, seed=0):
    q_lens = torch.tensor(q_lens, device=DEV, dtype=torch.int32)
    prefix_lens = torch.tensor(prefix_lens, device=DEV, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_seq_len = int(seq_lens.max())
    max_blocks = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_pages = batch * max_blocks
    cu = torch.zeros(batch + 1, device=DEV, dtype=torch.int32)
    cu[1:] = q_lens.cumsum(0)
    g = torch.Generator(device=DEV).manual_seed(seed)
    block_table = torch.randperm(
        num_pages, generator=g, device=DEV, dtype=torch.int32
    ).reshape(batch, max_blocks)
    idx_q = torch.randn(
        int(q_lens.sum()), num_idx_heads, HEAD_DIM, generator=g, device=DEV
    ).to(cache_dtype)
    cache = torch.randn(
        num_pages, SPARSE_BLOCK_SIZE, HEAD_DIM, generator=g, device=DEV
    ).to(cache_dtype)
    return dict(
        q_lens=q_lens,
        prefix_lens=prefix_lens,
        seq_lens=seq_lens,
        cu=cu,
        block_table=block_table,
        idx_q=idx_q,
        cache=cache,
        max_seq_len=max_seq_len,
        num_idx_heads=num_idx_heads,
    )


def _reference_scores(t):
    """Per-(head, token, block) max score, computed densely in fp32."""
    q_lens = t["q_lens"].tolist()
    prefix = t["prefix_lens"].tolist()
    out = []
    for req, q_len in enumerate(q_lens):
        n_blocks = (
            int(t["seq_lens"][req]) + SPARSE_BLOCK_SIZE - 1
        ) // SPARSE_BLOCK_SIZE
        pages = t["block_table"][req, :n_blocks]
        k = t["cache"][pages].reshape(-1, HEAD_DIM).float()
        start = int(t["cu"][req])
        q = t["idx_q"][start : start + q_len].float()
        s = torch.einsum("qhd,kd->hqk", q, k)
        q_pos = prefix[req] + torch.arange(q_len, device=DEV)
        k_pos = torch.arange(k.shape[0], device=DEV)
        s.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -float("inf"))
        out.append(s.reshape(s.shape[0], q_len, n_blocks, SPARSE_BLOCK_SIZE).max(-1)[0])
    return out


def _assert_valid_topk(actual, t, topk, local_blocks):
    """Every stored block must be within the causal window and be a true top-k.

    Validity rather than exact index match: scores tie at the k-th place.
    """
    ref = _reference_scores(t)
    prefix = t["prefix_lens"].tolist()
    for req, per_req in enumerate(ref):
        start = int(t["cu"][req])
        for h in range(per_req.shape[0]):
            for tok in range(per_req.shape[1]):
                valid = (prefix[req] + tok + SPARSE_BLOCK_SIZE) // SPARSE_BLOCK_SIZE
                row = per_req[h, tok, :valid].clone()
                row[max(0, valid - local_blocks) : valid] = 1e29
                sel = [i for i in actual[h, start + tok].tolist() if i >= 0]
                assert len(set(sel)) == len(sel), "duplicate block id"
                assert not sel or max(sel) < valid, "selected a non-causal block"
                assert len(sel) == min(topk, valid), (
                    f"expected {min(topk, valid)} blocks, got {len(sel)}"
                )
                kth = row.sort(descending=True).values[min(topk, valid) - 1]
                assert row[sel].min() >= kth, "dropped a strictly larger score"


@pytest.mark.parametrize("topk", [1, 6, 16, 64])
@pytest.mark.parametrize("num_idx_heads", [1, 2])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prefill_topk_matches_reference(topk, num_idx_heads, cache_dtype):
    t = _build([300, 1, 133], [0, 1024, 4096], num_idx_heads, cache_dtype)
    score = minimax_m3_index_score(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["cu"],
        t["seq_lens"],
        t["prefix_lens"],
        max_query_len=int(t["q_lens"].max()),
        max_seq_len=t["max_seq_len"],
        num_kv_heads=num_idx_heads,
    )
    actual = minimax_m3_index_topk(
        score,
        t["cu"],
        t["prefix_lens"],
        max_query_len=int(t["q_lens"].max()),
        topk=topk,
        init_blocks=0,
        local_blocks=1,
    )
    _assert_valid_topk(actual.cpu(), t, topk, local_blocks=1)


@pytest.mark.parametrize("q_len", [1, 7, 33, 40, 257])
def test_prefill_topk_tail_rows_untouched(q_len):
    """Query lengths that do not divide the tile must not write past the request."""
    topk = 16
    t = _build([q_len, q_len + 1], [0, 512], 1, torch.bfloat16)
    total_q = int(t["q_lens"].sum())
    score = minimax_m3_index_score(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["cu"],
        t["seq_lens"],
        t["prefix_lens"],
        max_query_len=int(t["q_lens"].max()),
        max_seq_len=t["max_seq_len"],
        num_kv_heads=1,
    )
    poison = -12345
    out = torch.full((1, total_q + 64, topk), poison, dtype=torch.int32, device=DEV)
    minimax_m3_index_topk(
        score,
        t["cu"],
        t["prefix_lens"],
        max_query_len=int(t["q_lens"].max()),
        topk=topk,
        init_blocks=0,
        local_blocks=1,
        out=out,
    )
    assert (out[:, total_q:, :] == poison).all(), "wrote past total_q"
    assert (out[:, :total_q, :] != poison).any()
    _assert_valid_topk(out[:, :total_q].cpu(), t, topk, local_blocks=1)


def test_query_tile_always_leaves_a_usable_config():
    """The tile `_topk_query_tile` picks must survive `_topk_prune_configs`.

    These are two halves of one invariant: pick a tile no config can serve and
    the launch dies inside Triton instead of raising an actionable error.
    """
    from vllm.models.minimax_m3.amd.ops.index_topk import (
        _topk_index_kernel,
        _topk_prune_configs,
    )

    configs = _topk_index_kernel.fn.configs
    for topk in (1, 6, 16, 64, 128, 256, 512, 1024):
        for max_query_len in (1, 33, 1024, 8160, 8192, 32768, 73728):
            for batch in (1, 4, 32, 128):
                tile = _topk_query_tile(max_query_len, batch, 1, topk)
                assert tile <= TOPK_QUERY_TILE and tile & (tile - 1) == 0
                kept = _topk_prune_configs(
                    configs,
                    {"topk": topk},
                    BLOCK_SIZE_T=triton.next_power_of_2(topk),
                    BLOCK_SIZE_Q=tile,
                )
                assert kept, f"no config for topk={topk} tile={tile}"
    # Both named widths stay reachable at M3's topk.
    assert _topk_query_tile(32768, 8, 1, 16) == TOPK_QUERY_TILE
    assert _topk_query_tile(1024, 1, 1, 16) == TOPK_QUERY_TILE_SMALL


def test_score_kv_chunks_invariants(monkeypatch):
    """The split the scorer picks must be launchable for any shape and device.

    The kernel divides its causal scan by this count and shifts by its bit
    length, so a zero or a non-power-of-two is a launch failure rather than a
    slow kernel. The CU count is patched so the sweep covers parts this test is
    not running on.
    """
    from vllm.models.minimax_m3.amd.ops import index_topk as ops

    for cus in (64, 80, 128, 256, 304, 1024):
        monkeypatch.setattr(ops, "num_compute_units", lambda _c=cus: _c)
        for num_q_blocks in (1, 2, 7, 64, 512, 1024):
            for batch in (1, 2, 8, 64, 512):
                for heads in (1, 2):
                    for max_block in (0, 1, 2, 7, 8, 63, 64, 1024, 100000):
                        n = ops._score_kv_chunks(num_q_blocks, batch, heads, max_block)
                        assert n >= 1
                        assert n & (n - 1) == 0, f"{n} is not a power of two"
                        assert n <= max(1, max_block)
                        assert n <= ops.PREFILL_SCORE_MAX_KV_CHUNKS


def _wide_tile_query_len() -> int:
    """Shortest single-request prefill that still selects the wide top-k tile.

    The tile widens once the grid can fill the device, so the threshold is a
    property of the GPU rather than a constant: hard-coding a length picks the
    narrow tile on any part with more CUs than the one it was written on.
    """
    return _min_topk_programs() * TOPK_QUERY_TILE


@pytest.mark.parametrize("tail", [0, 33])
def test_prefill_topk_wide_query_tile(tail):
    """Exercise the 32-row tile, which needs a launch big enough to fill the device.

    Every other prefill test here is small enough to fall through to the 8-row
    tile, so without this the shipping configuration is never compiled. ``tail``
    adds rows that do not divide the tile.
    """
    topk = 16
    q_len = _wide_tile_query_len() + tail
    t = _build([q_len], [0], 1, torch.bfloat16, seed=3)
    assert _topk_query_tile(q_len, 1, 1, topk) == TOPK_QUERY_TILE
    score = minimax_m3_index_score(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["cu"],
        t["seq_lens"],
        t["prefix_lens"],
        max_query_len=q_len,
        max_seq_len=t["max_seq_len"],
        num_kv_heads=1,
    )
    poison = -12345
    out = torch.full((1, q_len + 64, topk), poison, dtype=torch.int32, device=DEV)
    minimax_m3_index_topk(
        score,
        t["cu"],
        t["prefix_lens"],
        max_query_len=q_len,
        topk=topk,
        init_blocks=0,
        local_blocks=1,
        out=out,
    )
    assert (out[:, q_len:, :] == poison).all(), "wrote past total_q"
    _assert_valid_topk(out[:, :q_len].cpu(), t, topk, local_blocks=1)


@pytest.mark.parametrize("topk", [16, 64, 128, 256, 512, 1024])
def test_prefill_topk_supports_wide_topk(topk):
    """topk wide enough to force a wide BLOCK_SIZE_K must still launch."""
    t = _build([64, 64], [0, 2048], 1, torch.bfloat16)
    score = minimax_m3_index_score(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["cu"],
        t["seq_lens"],
        t["prefix_lens"],
        max_query_len=64,
        max_seq_len=t["max_seq_len"],
        num_kv_heads=1,
    )
    actual = minimax_m3_index_topk(
        score,
        t["cu"],
        t["prefix_lens"],
        max_query_len=64,
        topk=topk,
        init_blocks=0,
        local_blocks=1,
    )
    assert actual.shape[-1] == topk
    _assert_valid_topk(actual.cpu(), t, topk, local_blocks=1)


def test_prefill_topk_rejects_unservable_topk():
    """No legal config must raise an actionable error, not a compile assert."""
    t = _build([64], [0], 1, torch.bfloat16)
    score = minimax_m3_index_score(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["cu"],
        t["seq_lens"],
        t["prefix_lens"],
        max_query_len=64,
        max_seq_len=t["max_seq_len"],
        num_kv_heads=1,
    )
    with pytest.raises(ValueError, match="no usable config"):
        minimax_m3_index_topk(
            score,
            t["cu"],
            t["prefix_lens"],
            max_query_len=64,
            topk=4096,
            init_blocks=0,
            local_blocks=1,
        )


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("batch", [1, 8, 40])
def test_decode_topk_matches_reference(cache_dtype, batch):
    topk, local_blocks = 16, 1
    ctx = 3000
    t = _build([1] * batch, [ctx - 1] * batch, 1, cache_dtype)
    actual = minimax_m3_index_decode(
        t["idx_q"],
        t["cache"],
        t["block_table"],
        t["seq_lens"],
        max_seq_len=t["max_seq_len"],
        topk=topk,
        init_blocks=0,
        local_blocks=local_blocks,
        num_kv_heads=1,
        decode_query_len=1,
        max_decode_query_len=1,
    )
    _assert_valid_topk(actual.cpu(), t, topk, local_blocks=local_blocks)


def test_fp8_index_cache_insert_clamps_out_of_range():
    """bf16 index keys above the e4m3 max must saturate, not become NaN/inf."""
    from vllm.models.minimax_m3.amd.ops.sparse_pa import minimax_m3_insert_index_cache

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = finfo.min, finfo.max
    index_k = torch.tensor(
        [[1e4] * HEAD_DIM, [-1e4] * HEAD_DIM, [1.5] * HEAD_DIM],
        device=DEV,
        dtype=torch.bfloat16,
    )
    cache = torch.zeros(
        4, SPARSE_BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=torch.float8_e4m3fn
    )
    slots = torch.tensor([0, 1, 2], device=DEV, dtype=torch.int32)
    minimax_m3_insert_index_cache(index_k, cache, slots)
    got = cache[0, :3].float()
    assert torch.isfinite(got).all(), "e4m3 convert produced NaN/inf"
    assert torch.allclose(got[0], torch.full_like(got[0], fp8_max))
    assert torch.allclose(got[1], torch.full_like(got[1], fp8_min))
    assert torch.allclose(got[2], torch.full_like(got[2], 1.5))
