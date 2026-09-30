# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The receive-side layout fix-up must gather the received blocks only once.

`kv_postprocess_layout_on_receive` rewrites the blocks that were just copied
into the local cache so that their layout matches what the local attention
backend expects: it gathers them with `index_select`, permutes the two inner
dimensions and writes them back with `index_copy_`.

`index_select` copies every received block in full, and the function performed
it twice in a row - the second gather immediately overwrote the first with an
identical tensor. On the decode side this runs for every received block set, so
the duplicate doubles a temporary allocation whose size is proportional to the
transferred KV.
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    kv_postprocess_layout_on_receive,
)

_SHAPES = {
    # [num_blocks, n_kv_head, block_size, head_dim]
    4: (6, 2, 4, 3),
    # [num_blocks, kv_dim, n_kv_head, block_size, head_dim]
    5: (6, 2, 2, 4, 3),
}
_INDICES = torch.tensor([1, 3, 4])


class _CountingCache:
    """A real cache tensor that records how often it is gathered / written."""

    def __init__(self, tensor: torch.Tensor, counters: dict[str, int]):
        self._tensor = tensor
        self._counters = counters

    def index_select(self, dim: int, indices) -> torch.Tensor:
        self._counters["index_select"] += 1
        return self._tensor.index_select(dim, indices)

    def index_copy_(self, dim: int, indices, source: torch.Tensor) -> torch.Tensor:
        self._counters["index_copy_"] += 1
        return self._tensor.index_copy_(dim, indices, source)


@pytest.mark.parametrize("ndim", [4, 5], ids=["4d_cache", "5d_cache"])
def test_the_received_blocks_are_gathered_once(ndim):
    counters = {"index_select": 0, "index_copy_": 0}
    cache = torch.arange(
        torch.tensor(_SHAPES[ndim]).prod().item(), dtype=torch.float16
    ).reshape(_SHAPES[ndim])
    original = cache.clone()
    before = cache.index_select(0, _INDICES)

    kv_postprocess_layout_on_receive(_CountingCache(cache, counters), _INDICES)

    after = cache.index_select(0, _INDICES)

    # A layout fix-up permutes data: it neither invents nor drops values.
    assert torch.equal(after.flatten().sort().values, before.flatten().sort().values)
    # The blocks that were not received must be left alone.
    untouched = [i for i in range(_SHAPES[ndim][0]) if i not in _INDICES.tolist()]
    assert torch.equal(cache[untouched], original[untouched])
    # One gather and one write-back is all this rewrite needs.
    assert counters == {"index_select": 1, "index_copy_": 1}
