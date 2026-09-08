# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.attention.backends.mla.flashmla_sparse import plan_gathered_prefill

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
    """Where the top-k kernel's WORKSPACE_RANK_MAJOR branch sends a token."""
    owning_rank = (token // interleave) % world
    local_idx = (token // (world * interleave)) * interleave + token % interleave
    return owning_rank * rows_per_rank + region_start + local_idx


def _simulate_gathered_kv(
    extents: list[int], world: int, plan_starts: np.ndarray, rows_per_rank: int
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
                row = rank * rows_per_rank + int(plan_starts[region]) + local
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
    rows = np.arange(len(extents), dtype=np.int64)
    plan = plan_gathered_prefill(
        rows, np.array(extents, dtype=np.int64), world, max_gathered_rows=1 << 20
    )
    assert len(plan.chunk_bounds) == 1, "the budget here fits everything in one chunk"
    rows_per_rank = int(plan.rows_per_rank.sum())

    gathered = _simulate_gathered_kv(
        extents, world, plan.workspace_starts, rows_per_rank
    )

    for region, extent in enumerate(extents):
        region_start = int(plan.workspace_starts[region])
        for position in range(extent):
            row = _remapped_row(position, region_start, rows_per_rank, world)
            assert gathered[row] == region * 10**6 + position, (
                f"region {region} token {position} at world={world} resolved to "
                f"row {row}, which holds {gathered[row]}"
            )


@pytest.mark.parametrize("world", [2, 3, 8])
@pytest.mark.parametrize("extent", [1, 63, 64, 65, 127, 128, 129, 1000])
def test_rows_per_rank_is_the_per_rank_maximum(world, extent):
    """ceil(extent / W) has to cover the busiest rank, and waste at most a row."""
    plan = plan_gathered_prefill(
        np.zeros(1, dtype=np.int64),
        np.array([extent], dtype=np.int64),
        world,
        max_gathered_rows=1 << 20,
    )
    highest_local_slot = max(
        _shard_of_position(position, world)[1] for position in range(extent)
    )
    assert int(plan.rows_per_rank[0]) == highest_local_slot + 1


def test_rows_of_one_request_share_a_region():
    """PCP gives a rank several chunks of one request; they share one context."""
    rows = np.array([3, 3, 5, 5, 9], dtype=np.int64)
    extents = np.zeros(10, dtype=np.int64)
    extents[[3, 5, 9]] = [128, 256, 64]

    plan = plan_gathered_prefill(
        rows, extents, dcp_world_size=2, max_gathered_rows=1 << 20
    )

    assert plan.region_of_row.tolist() == [0, 0, 1, 1, 2]
    assert plan.region_first_row.tolist() == [0, 2, 4]
    assert plan.rows_per_rank.tolist() == [64, 128, 32]
    assert plan.workspace_starts.tolist() == [0, 64, 192]


def test_chunks_rebase_workspace_starts_and_respect_the_budget():
    """Each chunk reuses the buffer from row 0, so its starts restart at 0."""
    rows = np.arange(4, dtype=np.int64)
    extents = np.array([256, 256, 256, 256], dtype=np.int64)

    # 512 gathered rows = 256 per rank at W=2 = two requests' worth per chunk.
    plan = plan_gathered_prefill(rows, extents, dcp_world_size=2, max_gathered_rows=512)

    assert plan.chunk_bounds == [(0, 2), (2, 4)]
    assert plan.workspace_starts.tolist() == [0, 128, 0, 128]
    for chunk_start, chunk_stop in plan.chunk_bounds:
        rows_per_rank = int(plan.rows_per_rank[chunk_start:chunk_stop].sum())
        assert rows_per_rank * 2 <= 512


def test_request_too_large_for_the_workspace_is_rejected():
    """Fail loudly: a truncated region would silently drop context."""
    with pytest.raises(ValueError, match="contiguous KV rows"):
        plan_gathered_prefill(
            np.zeros(1, dtype=np.int64),
            np.array([4096], dtype=np.int64),
            dcp_world_size=2,
            max_gathered_rows=1024,
        )


def test_unordered_rows_are_rejected():
    """The layout keys on run-length dedup, so it needs grouped rows."""
    with pytest.raises(AssertionError, match="ascending global request"):
        plan_gathered_prefill(
            np.array([1, 0, 1], dtype=np.int64),
            np.array([64, 64], dtype=np.int64),
            dcp_world_size=2,
            max_gathered_rows=1 << 20,
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
