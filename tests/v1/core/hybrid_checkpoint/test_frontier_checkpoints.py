# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints (`mamba_frontier_checkpoint`) on the real
scheduler with an attention group, two align-mode Mamba groups and an indexer
ring, driven step by step on CPU."""

import pytest

from vllm.config import KVEventsConfig, SpeculativeConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request, RequestStatus

from .utils import (
    add_request,
    admitted_at,
    make_hybrid_scheduler,
    make_request,
    mock_connector,
    prompt,
    run,
    step,
)

pytestmark = pytest.mark.cpu_test

BLOCK = 64
RING = 4


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def make_scheduler(frontier: bool = True, **kwargs):
    return make_hybrid_scheduler(
        block_size=BLOCK, ring=RING, mamba_frontier_checkpoint=frontier, **kwargs
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("frontier", [False, True])
def test_next_turn_resumes_at_the_previous_frontier(
    frontier: bool, async_scheduling: bool
):
    scheduler = make_scheduler(frontier=frontier, async_scheduling=async_scheduling)
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens % BLOCK

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=4)
    # Without frontier checkpoints the turn resumes at its prompt's last block
    # boundary, where align mode cached the replay state.
    expected = frontier_tokens if frontier else 128
    assert admitted_at(scheduler, second) == expected


def test_frontier_hits_are_counted_in_prefix_cache_stats():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens

    tokens = list(first.all_token_ids) + prompt(2, 40)
    add_request(scheduler, "t1", tokens, max_tokens=4)
    scheduler.schedule()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.hits) == (1, frontier_tokens)
    assert stats.frontier_hits == frontier_tokens - 2 * BLOCK


def test_resume_inside_an_indexer_group_takes_one_token_per_step():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens % RING

    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    taken = [out.num_scheduled_tokens["t1"] for out in run(scheduler, second)]
    catchup = (-frontier_tokens) % RING
    assert taken[:catchup] == [1] * catchup
    assert taken[catchup] > 1


def test_a_chunk_ending_inside_an_indexer_group_keeps_its_size():
    """Only a resume at a frontier catches up one token per step. A budget
    smaller than the block ends an ordinary chunk inside a group, and the next
    chunk goes on as usual."""
    scheduler = make_scheduler(max_num_batched_tokens=9 * RING + 1)
    request = add_request(scheduler, "r", prompt(1, 150), max_tokens=2)
    taken = [out.num_scheduled_tokens["r"] for out in run(scheduler, request)]
    assert taken[0] % RING
    # The last step decodes the second output token.
    assert 1 not in taken[:-1]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_decode_boundary_state_is_resumable(async_scheduling: bool):
    scheduler = make_scheduler(async_scheduling=async_scheduling)
    first = add_request(scheduler, "t0", prompt(1, 100), max_tokens=60)
    run(scheduler, first)
    assert first.num_computed_tokens > 2 * BLOCK

    # Shares the first request's tokens past its decode boundary at 128, but
    # not its frontier.
    tokens = list(first.all_token_ids[: 2 * BLOCK + 10]) + prompt(3, 50)
    sibling = add_request(scheduler, "s", tokens, max_tokens=2)
    assert admitted_at(scheduler, sibling) == 2 * BLOCK


def test_frontier_goes_whole_when_one_of_its_blocks_is_evicted():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    [entry] = checkpoints.frontiers.entries.values()
    page, *others = entry.blocks.values()

    scheduler.kv_cache_manager.block_pool.evict_blocks({page.block_id})

    assert not checkpoints.frontiers.entries
    assert all(block.block_hash is None for block in others)
    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    assert admitted_at(scheduler, second) == 128


def test_frontier_page_keeps_the_key_of_a_finer_hash_unit():
    """With a prefix-match unit finer than the block, the page holding the
    frontier is already cached under the request's own key at its prompt's
    last unit boundary: the frontier key joins that key, and releasing the
    frontier removes only its own key."""
    unit = 16
    scheduler = make_scheduler(prefix_match_unit=unit, hash_block_size=unit)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    pool = scheduler.kv_cache_manager.block_pool
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens // BLOCK == 150 // BLOCK

    [entry] = checkpoints.frontiers.entries.values()
    page = entry.blocks[0]
    assert page.block_hash is not None
    assert get_block_hash(page.block_hash) == first.block_hashes[150 // unit - 1]
    frontier_page_key = make_block_hash_with_group_id(entry.key, 0)
    assert frontier_page_key in pool.cached_block_hashes_by_block[page.block_id]

    second = add_request(
        scheduler, "t1", list(first.all_token_ids) + prompt(2, 40), max_tokens=4
    )
    assert admitted_at(scheduler, second) == frontier_tokens
    run(scheduler, second)

    # The session moved on: its leaf replacement released the first frontier.
    assert entry.key not in checkpoints.frontiers.entries
    assert page.block_hash is not None
    assert get_block_hash(page.block_hash) == first.block_hashes[150 // unit - 1]
    assert page.block_id not in pool.cached_block_hashes_by_block
    # The page still serves the hit at the prompt's last unit boundary.
    tokens = list(first.all_token_ids[:150]) + prompt(3, 30)
    probe = add_request(scheduler, "p", tokens, max_tokens=1)
    assert admitted_at(scheduler, probe) == 150 // unit * unit


def test_aborted_request_publishes_no_frontier():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=50)
    while first.num_computed_tokens <= 150:
        step(scheduler)
    scheduler.finish_requests("t0", RequestStatus.FINISHED_ABORTED)
    assert scheduler.mamba_checkpoints is not None
    assert not scheduler.mamba_checkpoints.frontiers.entries


def test_session_leaf_is_replaced_unless_another_turn_needs_it():
    scheduler = make_scheduler()
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    entries = checkpoints.frontiers.entries

    def frontier_of(request: Request) -> bytes:
        return frontier_key(sha256, request, request.num_computed_tokens, BLOCK)

    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    tokens = list(first.all_token_ids)
    branch = add_request(scheduler, "b", tokens + prompt(5, 20), max_tokens=3)
    second = add_request(scheduler, "t1", tokens + prompt(2, 20), max_tokens=3)
    run(scheduler, branch, second)
    # Two continuations resumed at it: a branch point stays.
    assert frontier_of(first) in entries

    tokens = list(second.all_token_ids) + prompt(6, 20)
    third = add_request(scheduler, "t2", tokens, max_tokens=3)
    run(scheduler, third)
    # The session moved on from the second turn's frontier.
    assert frontier_of(second) not in entries
    assert frontier_of(third) in entries


def test_next_turn_keeps_the_frontier_a_waiting_request_resumes_at():
    """A request whose lookup resumed it at a frontier and that still waits
    for blocks, as a refused allocation leaves it, keeps that frontier when
    the session's next turn moves its leaf on."""
    scheduler = make_scheduler()
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    key = frontier_key(sha256, first, frontier_tokens, BLOCK)
    tokens = list(first.all_token_ids)

    second = add_request(scheduler, "t1", tokens + prompt(2, 20), max_tokens=3)
    step(scheduler)
    waiting = make_request(scheduler, "w", tokens + prompt(5, 20), max_tokens=1)
    blocks, num_tokens, _ = scheduler.kv_cache_manager.get_computed_blocks(waiting)
    assert num_tokens == frontier_tokens
    checkpoints.admitted(waiting, num_tokens, blocks, 0)
    run(scheduler, second)
    assert key in checkpoints.frontiers.entries

    scheduler.add_request(waiting)
    assert admitted_at(scheduler, waiting) == frontier_tokens


def test_longer_connector_hit_supersedes_the_frontier():
    """The KV connector sees the block-aligned part of a frontier hit and loads
    past the frontier: the request resumes where the load ends, without the
    frontier's partial page, state or ring, and at the state the load caches
    there rather than at its last local block."""
    scheduler = make_scheduler(kv_transfer_config=mock_connector(2 * BLOCK))
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    assert admitted_at(scheduler, first) == 2 * BLOCK
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    # Inside a block and inside an indexer group: the frontier keeps a ring.
    assert frontier_tokens // BLOCK == 2 and frontier_tokens % RING

    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=3)
    assert admitted_at(scheduler, second) == 4 * BLOCK
    assert checkpoints.frontiers.num_hits == 0
    units = checkpoints.graph.units
    # Each load resumed its request at the state it ends with.
    assert units[second.block_hashes[1]].hits == 1
    assert units[second.block_hashes[3]].hits == 1


def test_a_load_resumes_at_the_state_of_its_hash_unit():
    """With a prefix-match unit finer than the block, the state a load ends
    with is cached under the request's hash at that unit (here the sixteenth
    and eighth), not at the block count's unit."""
    unit = 16
    scheduler = make_scheduler(
        kv_transfer_config=mock_connector(2 * BLOCK),
        prefix_match_unit=unit,
        hash_block_size=unit,
    )
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    assert admitted_at(scheduler, first) == 2 * BLOCK
    run(scheduler, first)
    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=3)
    assert admitted_at(scheduler, second) == 4 * BLOCK
    units = checkpoints.graph.units
    assert units[second.block_hashes[2 * BLOCK // unit - 1]].hits == 1
    assert units[second.block_hashes[4 * BLOCK // unit - 1]].hits == 1


@pytest.mark.parametrize(
    "kwargs,needs",
    [
        ({"prefix_cache_retention_interval": BLOCK}, "retention_interval 0"),
        (
            {"long_prefill_token_threshold": BLOCK // 2},
            "long_prefill_token_threshold below the block size",
        ),
        ({"resume_catchup": None}, "resume_catchup"),
        (
            {"kv_transfer_config": mock_connector(0, kv_load_failure_policy="fail")},
            "kv_load_failure_policy 'recompute'",
        ),
        (
            {"kv_transfer_config": mock_connector(0, kv_role="kv_producer")},
            "kv_role 'kv_both'",
        ),
    ],
)
def test_unsupported_configurations_are_refused(kwargs, needs):
    with pytest.raises(ValueError, match=needs):
        make_scheduler(**kwargs)


def test_speculative_decoding_is_refused():
    speculative_config = SpeculativeConfig(method="ngram_gpu", num_speculative_tokens=2)
    with pytest.raises(ValueError, match="needs no speculative decoding"):
        make_scheduler(speculative_config=speculative_config)


def test_kv_cache_events_are_refused():
    kv_events_config = KVEventsConfig(enable_kv_cache_events=True, publisher="null")
    with pytest.raises(
        ValueError,
        match="needs no KV cache events, or kv_events_config.recovery_unit_events",
    ):
        make_scheduler(kv_events_config=kv_events_config)
