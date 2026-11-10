# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GreedyDual-Size eviction (`prefix_cache_eviction="gds"`) of recovery
units on the real scheduler with an attention group and two align-mode Mamba
groups, driven step by step on CPU."""

import itertools
import random
import time

import pytest

from vllm.config import CacheConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint import eviction
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.kv_cache_utils import init_none_hash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .utils import add_request, admitted_at, make_hybrid_scheduler, prompt, run, step

pytestmark = pytest.mark.cpu_test

BLOCK = 64
# A prefill cost model (seconds per token, and per token per 100,000 tokens
# of context) fitted on a hybrid model's prefill steps.
COST = (1.862e-5, 3.33e-6)


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.fixture
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each read of the monotonic clock is a second after the last, so every
    step refreshes the units that waiting requests' hits pin."""
    now = itertools.count(1_000_000.0, 1.0)
    monkeypatch.setattr(time, "monotonic", lambda: next(now))


def make_scheduler(
    cost: tuple[float, float] | None = COST,
    num_blocks: int = 40,
    max_num_batched_tokens: int = 1024,
) -> Scheduler:
    """Frontier checkpoints with GreedyDual-Size eviction on `cost`, or with
    LRU eviction when it is None."""
    return make_hybrid_scheduler(
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        ring=0,
        mamba_frontier_checkpoint=True,
        prefix_cache_eviction="lru" if cost is None else "gds",
        prefill_cost_coefficients=cost,
    )


def finished_sessions(scheduler: Scheduler) -> tuple[Request, Request]:
    """A long session, then a short one, each finished: each leaves a state at
    its prompt's last block boundary and a frontier at its end."""
    long = add_request(scheduler, "long", prompt(1, 1000), max_tokens=5)
    run(scheduler, long)
    short = add_request(scheduler, "short", prompt(2, 200), max_tokens=5)
    run(scheduler, short)
    assert (long.num_computed_tokens, short.num_computed_tokens) == (1004, 204)
    return long, short


def next_turn(scheduler: Scheduler, request_id: str, session: Request) -> Request:
    tokens = list(session.all_token_ids) + prompt(9, 50)
    return add_request(scheduler, request_id, tokens, max_tokens=2)


def cached_units(scheduler: Scheduler) -> list[tuple[int, bool]]:
    """(num_tokens, is a frontier) of every resumable unit."""
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    return sorted((u.num_tokens, u.frontier) for u in checkpoints.graph.units.values())


@pytest.mark.parametrize(
    "cost,units,long_hit",
    [
        # LRU: the least recently freed first, the long session's blocks, so
        # its next turn starts over.
        (None, [(192, False), (204, True), (576, False), (600, True)], 0),
        # The fewest prefill seconds saved per block freed first: the short
        # session's frontier (12 tokens past its block-boundary state), then
        # the filler's frontier as soon as it is cached.
        (COST, [(192, False), (576, False), (960, False), (1004, True)], 1004),
        # Scaling the cost model changes no decision.
        (
            (1000 * COST[0], 1000 * COST[1]),
            [(192, False), (576, False), (960, False), (1004, True)],
            1004,
        ),
    ],
    ids=["lru", "gds", "gds-scaled-cost"],
)
def test_eviction_keeps_the_states_that_save_the_most_prefill_per_block(
    cost: tuple[float, float] | None,
    units: list[tuple[int, bool]],
    long_hit: int,
):
    scheduler = make_scheduler(cost)
    long, _ = finished_sessions(scheduler)
    filler = add_request(scheduler, "filler", prompt(3, 600), max_tokens=1)
    run(scheduler, filler)

    assert cached_units(scheduler) == units
    assert admitted_at(scheduler, next_turn(scheduler, "long-1", long)) == long_hit


@pytest.mark.usefixtures("fast_clock")
@pytest.mark.parametrize("waiting", [True, False])
def test_waiting_request_keeps_the_state_it_resumes_at(waiting: bool):
    scheduler = make_scheduler()
    _, short = finished_sessions(scheduler)
    add_request(scheduler, "filler", prompt(3, 600), max_tokens=1)
    if not waiting:
        # The filler takes its blocks before the next turn arrives.
        step(scheduler)
    resumed = admitted_at(scheduler, next_turn(scheduler, "short-1", short))
    # Waiting while the filler takes its blocks, the next turn pins the short
    # session's frontier; arriving after, it finds that cheapest unit gone and
    # resumes at the block-boundary state below it.
    assert resumed == (204 if waiting else 192)


@pytest.mark.usefixtures("fast_clock")
def test_pins_are_refreshed_within_a_lookup_budget(monkeypatch: pytest.MonkeyPatch):
    """The refresh looks up the waiting queue's front first, until the hash
    units of the requests it looked up exceed the budget."""
    scheduler = make_scheduler()
    long, short = finished_sessions(scheduler)
    next_turn(scheduler, "long-1", long)
    next_turn(scheduler, "short-1", short)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    gds = checkpoints.eviction
    gds.new_step()
    assert set(gds._pinned_blocks.values()) == {0, 1}
    # The long session's next turn walks 1,054 // 64 + 1 = 17 units.
    monkeypatch.setattr(eviction, "_PIN_UNITS", 20)
    gds.new_step()
    assert set(gds._pinned_blocks.values()) == {0}


def test_a_state_key_that_never_becomes_a_unit_drops_its_path():
    """A Mamba group cached a state the other group did not: once its key is
    removed, the policy no longer keeps the path recorded for it."""
    scheduler = make_scheduler()
    long, _ = finished_sessions(scheduler)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    gds = checkpoints.eviction
    key = long.block_hashes[7]
    assert key not in gds.units
    gds.record(key, 8 * BLOCK, long)
    assert key in gds.paths
    mamba_group = gds.mamba[0]
    gds.keys_removed([make_block_hash_with_group_id(key, mamba_group)])
    assert key not in gds.paths


@pytest.mark.usefixtures("fast_clock")
def test_a_prefix_cache_reset_drops_the_pins():
    scheduler = make_scheduler()
    _, short = finished_sessions(scheduler)
    next_turn(scheduler, "short-1", short)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    checkpoints.eviction.new_step()
    assert checkpoints.eviction._pinned_blocks
    assert scheduler.reset_prefix_cache()
    assert not checkpoints.eviction._pinned_blocks


@pytest.mark.usefixtures("fast_clock")
def test_exhausted_pool_takes_the_waiting_queue_back_pins_first():
    """A long prefill that finds no free unit left takes pinned blocks: those
    the last waiting request resumes at go before the ones the head resumes
    at, so the head, admitted next, still resumes at its frontier."""
    scheduler = make_scheduler(num_blocks=26, max_num_batched_tokens=256)
    sessions = []
    for i, request_id in enumerate(("a", "b")):
        session = add_request(scheduler, request_id, prompt(1 + i, 200), max_tokens=5)
        run(scheduler, session)
        sessions.append(session)
    add_request(scheduler, "filler", prompt(3, 500), max_tokens=1)
    head = next_turn(scheduler, "a-1", sessions[0])
    back = next_turn(scheduler, "b-1", sessions[1])
    hits = {}
    while RequestStatus.WAITING in (head.status, back.status):
        for new in step(scheduler).scheduled_new_reqs:
            hits[new.req_id] = new.num_computed_tokens
    assert (hits["a-1"], hits["b-1"]) == (204, 0)


@pytest.mark.parametrize("cost", [None, COST], ids=["lru", "gds"])
def test_sessions_under_pressure_keep_the_pool_consistent(
    cost: tuple[float, float] | None,
):
    """Multi-turn sessions, forks of earlier prefixes and new prompts, two at a
    time, in a pool that holds a few of them. After every step each resumable
    unit is cached in every Mamba group and the free queue holds exactly the
    unreferenced blocks; once all requests finish every block is free."""
    rng = random.Random(0)
    scheduler = make_scheduler(cost, num_blocks=90, max_num_batched_tokens=512)
    pool = scheduler.kv_cache_manager.block_pool
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    sessions = [prompt(100 + i, rng.randrange(80, 700)) for i in range(6)]
    hits = 0
    num_steps = 0
    for turn in range(30):
        batch: list[tuple[int, Request]] = []
        for j in range(rng.randrange(1, 3)):
            i = rng.randrange(len(sessions))
            tokens = sessions[i]
            if rng.random() < 0.2:
                tokens = tokens[: rng.randrange(1, len(tokens))]
            request = add_request(
                scheduler,
                f"r{turn}-{j}",
                tokens + prompt(1000 + 2 * turn + j, rng.randrange(1, 300)),
                max_tokens=rng.randrange(1, 90),
            )
            batch.append((i, request))
        while not all(request.is_finished() for _, request in batch):
            output = step(scheduler)
            num_steps += 1
            assert num_steps < 5000
            hits += sum(
                new.num_computed_tokens > 0 for new in output.scheduled_new_reqs
            )
            assert all(checkpoints.complete(key) for key in checkpoints.graph.units)
            free = pool.free_block_queue.get_all_free_blocks()
            assert all(block.ref_cnt == 0 for block in free)
            assert len(free) == sum(
                block.ref_cnt == 0 for block in pool.blocks if not block.is_null
            )
        for i, request in batch:
            tokens = list(request.all_token_ids)
            sessions[i] = tokens if len(tokens) < 2000 else prompt(200 + turn, 300)
    assert hits > 0
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    if cost is not None:
        assert checkpoints.eviction is not None
        assert checkpoints.eviction.num_evicted_units > 0


def leaf_last_scheduler(leaf_last: bool, idle: float | None = None) -> Scheduler:
    return make_hybrid_scheduler(
        block_size=BLOCK,
        num_blocks=40,
        ring=0,
        mamba_frontier_checkpoint=True,
        prefix_cache_eviction="gds",
        prefill_cost_coefficients=COST,
        prefix_cache_leaf_last=leaf_last,
        prefix_cache_leaf_idle_seconds=idle,
    )


@pytest.mark.parametrize("leaf_last", [False, True])
def test_leaf_last_ranks_inner_states_below_every_leaf(leaf_last: bool):
    """The long session's state at its prompt's last block boundary lies
    under its frontier (inner): by GreedyDual-Size alone it outranks the short
    session's frontier, which saves almost nothing; with leaf-last tiers it
    goes first."""
    scheduler = leaf_last_scheduler(leaf_last)
    long, _short = finished_sessions(scheduler)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    eviction = checkpoints.eviction
    inner = eviction.units[long.block_hashes[1000 // BLOCK - 1]]
    [short_frontier] = [
        u for u in eviction.units.values() if u.frontier and u.num_tokens == 204
    ]
    inner_score = eviction._score(inner)
    leaf_score = eviction._score(short_frontier)
    if leaf_last:
        assert (inner.tier, short_frontier.tier) == (1, 2)
        assert inner_score < leaf_score
    else:
        assert inner_score > leaf_score


def test_an_idle_leaf_ranks_as_inner(monkeypatch: pytest.MonkeyPatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    scheduler = leaf_last_scheduler(True, idle=60.0)
    finished_sessions(scheduler)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    eviction = checkpoints.eviction
    [frontier] = [
        u for u in eviction.units.values() if u.frontier and u.num_tokens == 204
    ]
    assert eviction._tier(frontier) == 2
    clock[0] += 61.0
    assert eviction._tier(frontier) == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"prefix_cache_leaf_last": True}, "prefix_cache_leaf_last requires"),
        (
            {
                "prefix_cache_eviction": "gds",
                "mamba_frontier_checkpoint": True,
                "prefill_cost_coefficients": COST,
                "prefix_cache_leaf_idle_seconds": 60.0,
            },
            "requires prefix_cache_leaf_last",
        ),
        (
            {"prefix_cache_eviction": "gds", "prefill_cost_coefficients": COST},
            "requires mamba_frontier_checkpoint",
        ),
        (
            {"prefix_cache_eviction": "gds", "mamba_frontier_checkpoint": True},
            "requires prefill_cost_coefficients",
        ),
        ({"prefill_cost_coefficients": (0.0, 1e-6)}, "need d > 0 and e >= 0"),
        ({"prefill_cost_coefficients": (1e-5, -1e-6)}, "need d > 0 and e >= 0"),
    ],
)
def test_invalid_eviction_configurations_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CacheConfig(**kwargs)


def test_evicting_a_frontier_keeps_the_page_key_of_a_finer_hash_unit():
    """With a prefix-match unit finer than the block, the frontier's page is
    also cached under the request's own key at its prompt's last unit
    boundary. Evicting the frontier removes only the frontier's key from the
    page, which is neither freed nor counted in the frontier's size."""
    unit = 16
    scheduler = make_hybrid_scheduler(
        block_size=BLOCK,
        num_blocks=40,
        max_num_batched_tokens=1024,
        ring=0,
        mamba_frontier_checkpoint=True,
        prefix_cache_eviction="gds",
        prefill_cost_coefficients=COST,
        prefix_match_unit=unit,
        hash_block_size=unit,
    )
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=5)
    run(scheduler, first)
    entry = checkpoints.frontiers.entries[
        frontier_key(sha256, first, first.num_computed_tokens, unit)
    ]
    page = entry.blocks[0]
    own_key = make_block_hash_with_group_id(first.block_hashes[150 // unit - 1], 0)
    assert page.block_hash == own_key
    gds = checkpoints.eviction
    frontier = gds.units[entry.key]
    assert gds._size(frontier) == len(gds._state_blocks(frontier)) - 1

    freed = gds._evict(frontier)
    assert page.block_id not in {b.block_id for b in freed}
    assert entry.key not in checkpoints.frontiers.entries
    assert page.block_hash == own_key
    tokens = list(first.all_token_ids[:150]) + prompt(3, 30)
    probe = add_request(scheduler, "p", tokens, max_tokens=1)
    assert admitted_at(scheduler, probe) == 150 // unit * unit
