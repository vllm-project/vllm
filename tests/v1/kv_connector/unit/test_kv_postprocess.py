# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NIXL receive post-process helpers run on the connector's registered
caches, which are logical ``[B, H, N, C]`` views whose strides follow the
``KVCacheLayout``. The helpers must reorder the bytes NIXL wrote into each
block's physical memory, so they are checked here against a block whose
logical and physical orders differ (``N != H``)."""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

NUM_BLOCKS, HEADS, TOKENS, HEAD_DIM = 3, 4, 16, 8


def _logical_view(layout: KVCacheLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """A flat buffer and its logical ``[B, H, N, D]`` view with ``layout``
    strides, as ``create_kv_cache_views`` builds it."""
    shape = (NUM_BLOCKS, HEADS, TOKENS, HEAD_DIM)
    order = layout.layer_view_order
    strides = [0] * 4
    stride = 1
    for dim in reversed(order):
        strides[dim] = stride
        stride *= shape[dim]
    raw = torch.zeros(stride, dtype=torch.int64)
    return raw, raw.as_strided(shape, tuple(strides))


def _write_block_bytes(raw: torch.Tensor, block: int, payload: torch.Tensor):
    """Lay ``payload`` into block ``block`` the way an RDMA write would:
    as the block's physical bytes, whatever the logical view says. Only for
    layouts whose block is one contiguous run."""
    n = HEADS * TOKENS * HEAD_DIM
    raw[block * n : (block + 1) * n] = payload.reshape(-1)


def _write_head_run(
    raw: torch.Tensor, cache: torch.Tensor, block: int, head: int, payload
):
    """Lay ``payload`` into the ``(block, head)`` run of a layout that stores
    each head's tokens separately (LHBNC): NIXL registers one region per
    head there, so a write lands inside that run."""
    start = block * cache.stride(0) + head * cache.stride(1)
    raw[start : start + TOKENS * HEAD_DIM] = payload.reshape(-1)


def _truth() -> torch.Tensor:
    """Distinct values per (head, token, elem): logical ``[H, N, D]``."""
    return torch.arange(HEADS * TOKENS * HEAD_DIM).reshape(HEADS, TOKENS, HEAD_DIM)


@pytest.mark.cpu_test
def test_layout_postprocess_on_logical_nhd_view():
    """Remote HND block written raw into a local NHD block (the
    ``enable_permute_local_kv`` path): after the helper the logical view
    must read the true head/token values."""
    layout = KVCacheLayout.LBNHC
    raw, cache = _logical_view(layout)
    truth = _truth()
    # Block 1 holds the HND bytes; block 0 and 2 are untouched sentinels.
    _write_block_bytes(raw, 1, truth)  # HND bytes
    _write_block_bytes(raw, 2, torch.full_like(truth, -1))

    kv_postprocess_layout_on_receive(cache, torch.tensor([1]), layout)

    assert torch.equal(cache[1], truth)
    assert torch.all(cache[0] == 0)
    assert torch.all(cache[2] == -1)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "layout",
    [
        KVCacheLayout.LBHNC,
        KVCacheLayout.BLHNC,
        KVCacheLayout.LBNHC,
        KVCacheLayout.BLNHC,
        KVCacheLayout.LHBNC,
    ],
)
def test_blksize_postprocess_regroups_sub_blocks(layout: KVCacheLayout):
    """Remote blocks are 1/ratio the local size with the same layout. The
    ratio sub-blocks land back to back in each NIXL region of the local
    block; the helper must turn them into one block of the local layout."""
    ratio = 4
    raw, cache = _logical_view(layout)
    truth = _truth()
    sub = TOKENS // ratio
    pieces = [truth[:, i * sub : (i + 1) * sub] for i in range(ratio)]  # [H, sub, D]
    if layout.is_block_contiguous:
        # One HND run per block: sub-blocks are head-major chunks.
        _write_block_bytes(raw, 1, torch.stack([p.reshape(-1) for p in pieces]))
    elif layout.is_block_compact:
        # One NHD run per block: sub-blocks are token-major chunks.
        _write_block_bytes(
            raw, 1, torch.stack([p.permute(1, 0, 2).reshape(-1) for p in pieces])
        )
    else:
        # One run per (block, head): each head's sub-blocks concatenate.
        for h in range(HEADS):
            _write_head_run(raw, cache, 1, h, torch.cat([p[h] for p in pieces]))
    cache[2].fill_(-1)

    kv_postprocess_blksize_on_receive(cache, torch.tensor([1]), ratio, layout)

    assert torch.equal(cache[1], truth)
    assert torch.all(cache[0] == 0)
    assert torch.all(cache[2] == -1)


@pytest.mark.cpu_test
def test_blksize_postprocess_selects_blocks_not_heads_on_lhbnc():
    """LHBNC stores heads outside blocks. Permuting the whole view into
    physical order would put heads on dim 0 and ``index_select`` heads
    numbered by block id; the helper must still address blocks."""
    layout = KVCacheLayout.LHBNC
    raw, cache = _logical_view(layout)
    assert layout.layer_view_order == (1, 0, 2, 3)
    truth = _truth()
    for h in range(HEADS):
        _write_head_run(raw, cache, 1, h, truth[h])
    cache[0].fill_(7)
    cache[2].fill_(-1)

    # Block 1 with every block id in range; nothing may change.
    for block in range(NUM_BLOCKS):
        kv_postprocess_blksize_on_receive(cache, torch.tensor([block]), 2, layout)
        assert torch.equal(cache[1], truth)
        assert torch.all(cache[0] == 7)
        assert torch.all(cache[2] == -1)


@pytest.mark.cpu_test
def test_blksize_and_layout_postprocess_on_logical_nhd_view():
    """Remote HND sub-blocks into a larger local NHD block."""
    ratio = 4
    layout = KVCacheLayout.LBNHC
    raw, cache = _logical_view(layout)
    truth = _truth()
    sub = TOKENS // ratio
    subs = [truth[:, i * sub : (i + 1) * sub].reshape(-1) for i in range(ratio)]
    _write_block_bytes(raw, 1, torch.stack(subs))

    kv_postprocess_blksize_and_layout_on_receive(
        cache, torch.tensor([1]), ratio, layout
    )

    assert torch.equal(cache[1], truth)
    assert torch.all(cache[0] == 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBHNC, KVCacheLayout.LHBNC, KVCacheLayout.BHLNC]
)
def test_layout_postprocess_rejects_non_nhd_targets(layout: KVCacheLayout):
    """The HND -> NHD transposes reinterpret one contiguous NHD block; any
    other local layout is refused rather than reshaped."""
    _, cache = _logical_view(layout)
    with pytest.raises(AssertionError):
        kv_postprocess_layout_on_receive(cache, torch.tensor([1]), layout)
    with pytest.raises(AssertionError):
        kv_postprocess_blksize_and_layout_on_receive(
            cache, torch.tensor([1]), 2, layout
        )


@pytest.mark.cpu_test
def test_postprocess_rejects_non_contiguous_blocks():
    """The helpers only make sense when each block is one contiguous run in
    physical order, which is what NIXL wrote into."""
    layout = KVCacheLayout.LBNHC
    _, cache = _logical_view(layout)
    with pytest.raises(AssertionError):
        kv_postprocess_layout_on_receive(cache[:, ::2], torch.tensor([1]), layout)
