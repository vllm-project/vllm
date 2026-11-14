# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 indexer's decode row split: the ranks' shares, gathered in rank
order and cut to the step's rows, list every row once and in order, each
share holds whole requests, and steps below the minimum stay replicated.
Host arithmetic only."""

import pytest

from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.indexer import (
    ROW_SHARD_MIN_REQUESTS,
    _decode_row_shard,
)


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("per_request", [1, 6])
def test_shares_rebuild_the_rows(world_size: int, per_request: int) -> None:
    for num_requests in range(1, 41):
        rows = num_requests * per_request
        shards = [
            _decode_row_shard(rows, num_requests, world_size, rank)
            for rank in range(world_size)
        ]
        if num_requests < ROW_SHARD_MIN_REQUESTS:
            assert shards == [None] * world_size
            continue
        gathered: list[int | None] = []
        split = [shard for shard in shards if shard is not None]
        assert len(split) == world_size
        share_rows = {shard[2] for shard in split}
        assert len(share_rows) == 1
        for start, end, share in split:
            assert start % per_request == 0 and end % per_request == 0
            assert end - start <= share
            gathered += list(range(start, end)) + [None] * (share - (end - start))
        assert gathered[:rows] == list(range(rows))


def test_uneven_rows_stay_replicated() -> None:
    assert _decode_row_shard(13, 4, 8, 0) is None
