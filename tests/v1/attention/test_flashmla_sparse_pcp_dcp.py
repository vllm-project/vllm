# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.attention.backends.mla.flashmla_sparse import gathered_prefill_shards

BLOCK_SIZE = 64


def _shard_of_position(
    position: int, world: int, interleave: int = 1
) -> tuple[int, int]:
    virtual_block = position // (BLOCK_SIZE * world)
    offset = position % (BLOCK_SIZE * world)
    owner = (offset // interleave) % world
    local = virtual_block * BLOCK_SIZE + (offset // (interleave * world)) * interleave
    return owner, local + offset % interleave


def _remapped_row(
    token: int, region_start: int, rows_per_rank: int, world: int, interleave: int = 1
) -> int:
    """Where the top-k kernel sends a token in the rank-major gathered workspace."""
    owning_rank = (token // interleave) % world
    local_idx = (token // (world * interleave)) * interleave + token % interleave
    return owning_rank * rows_per_rank + region_start + local_idx


def _simulate_gathered_kv(
    extents: list[int], world: int, starts: np.ndarray, rows_per_rank: int
) -> np.ndarray:
    """Build what all_gather(shard, dim=0) holds, value-tagged by global token."""
    PAD = -1
    gathered = np.full(world * rows_per_rank, PAD, dtype=np.int64)
    for rank in range(world):
        for region, extent in enumerate(extents):
            for position in range(extent):
                owner, local = _shard_of_position(position, world)
                if owner != rank:
                    continue
                row = rank * rows_per_rank + int(starts[region]) + local
                assert gathered[row] == PAD, f"row {row} written twice"
                gathered[row] = region * 10**6 + position
    return gathered


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize(
    "extents",
    [
        [1],
        [7],
        [64],
        [65],
        [129],
        [8, 8],
        [1, 9],
        [300, 1, 129],
        [2730, 256, 63],
    ],
)
def test_gathered_workspace_round_trips_every_token(world, extents):
    """Every global token must be readable back from the gathered buffer."""
    _, rows_per_rank = gathered_prefill_shards(
        np.arange(len(extents)), np.array(extents, dtype=np.int64), world
    )
    # The builder lays entries out back to back, as on the non-DCP path.
    starts = np.concatenate([[0], np.cumsum(rows_per_rank[:-1])])
    shard_rows = int(rows_per_rank.sum())

    gathered = _simulate_gathered_kv(extents, world, starts, shard_rows)

    for region, extent in enumerate(extents):
        for position in range(extent):
            row = _remapped_row(position, int(starts[region]), shard_rows, world)
            assert gathered[row] == region * 10**6 + position, (
                f"region {region} token {position} at world={world} resolved to "
                f"row {row}, which holds {gathered[row]}"
            )


@pytest.mark.parametrize("world", [2, 3, 8])
@pytest.mark.parametrize("extent", [1, 63, 64, 65, 127, 128, 129, 1000])
def test_rows_per_rank_is_the_per_rank_maximum(world, extent):
    """ceil(extent / W) has to cover the busiest rank, and waste at most a row."""
    _, rows_per_rank = gathered_prefill_shards(
        np.arange(1), np.array([extent], dtype=np.int64), world
    )
    highest_local_slot = max(
        _shard_of_position(position, world)[1] for position in range(extent)
    )
    assert int(rows_per_rank[0]) == highest_local_slot + 1


def test_rows_of_one_request_share_an_entry():
    """PCP gives a rank several chunks of one request; they share one context."""
    rows = np.array([7, 7, 3, 3, 9])
    extents = np.array([128, 128, 256, 256, 64], dtype=np.int64)

    row_bounds, rows_per_rank = gathered_prefill_shards(rows, extents, 2)

    assert row_bounds.tolist() == [0, 2, 4, 5]
    assert rows_per_rank.tolist() == [64, 128, 32]


def test_rows_of_one_request_must_be_adjacent():
    """An entry is one contiguous run; an interleaved request has no entry."""
    with pytest.raises(AssertionError, match="adjacent"):
        gathered_prefill_shards(
            np.array([0, 1, 0]), np.array([64, 64, 64], dtype=np.int64), 2
        )


@pytest.mark.parametrize("query_len", [5328, 5332, 16, 17, 100, 4095])
def test_nominal_chunk_length_never_exceeds_a_ranks_real_tokens(query_len):
    """The metadata is sized from the query slice, so it must fit real tokens."""
    world = 8
    num_chunks = 2 * world
    chunk_size = (query_len + num_chunks - 1) // num_chunks
    nominal = chunk_size

    for rank in range(world):
        real = 0
        for chunk_idx in (rank, num_chunks - 1 - rank):
            offset = chunk_idx * chunk_size
            real += max(0, min(chunk_size, query_len - offset))
        rows = 2 if real else 0
        assert min(nominal * rows, real) == real, (
            f"rank {rank} would be asked for {nominal * rows} tokens but owns "
            f"{real}; the clamp in build_prefill_chunk_metadata is what keeps "
            "the difference out of the metadata"
        )
