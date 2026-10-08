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
    as the block's physical bytes, whatever the logical view says."""
    n = HEADS * TOKENS * HEAD_DIM
    raw[block * n : (block + 1) * n] = payload.reshape(-1)


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
@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
def test_blksize_postprocess_regroups_sub_blocks(layout: KVCacheLayout):
    """Remote blocks are 1/ratio the local size with the same layout. The
    ratio sub-blocks land back to back in the local block's bytes; the helper
    must turn them into one block of the local layout."""
    ratio = 4
    raw, cache = _logical_view(layout)
    truth = _truth()
    sub = TOKENS // ratio
    # Sub-block i holds tokens [i*sub, (i+1)*sub) in the remote (= local) order.
    heads_outer = layout.layer_view_order.index(1) < layout.layer_view_order.index(2)
    subs = []
    for i in range(ratio):
        piece = truth[:, i * sub : (i + 1) * sub]  # [H, sub, D]
        subs.append(piece if heads_outer else piece.permute(1, 0, 2))
    _write_block_bytes(raw, 1, torch.stack([s.reshape(-1) for s in subs]))

    kv_postprocess_blksize_on_receive(cache, torch.tensor([1]), ratio, layout)

    assert torch.equal(cache[1], truth)
    assert torch.all(cache[0] == 0)


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
def test_postprocess_rejects_non_contiguous_blocks():
    """The helpers only make sense when each block is one contiguous run in
    physical order, which is what NIXL wrote into."""
    layout = KVCacheLayout.LBNHC
    _, cache = _logical_view(layout)
    with pytest.raises(AssertionError):
        kv_postprocess_layout_on_receive(cache[:, ::2], torch.tensor([1]), layout)
