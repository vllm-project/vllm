# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache events of frontier checkpoints with
`kv_events_config.recovery_unit_events`: recovery units, frontier keys and
the cause of every removal, on the real scheduler driven step by step on CPU."""

import itertools
import random
import time
from typing import Any

import msgspec
import pytest

from vllm.config import KVEventsConfig
from vllm.distributed.kv_events import (
    REMOVAL_CAUSES,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    ExternalBlockHash,
    KVCacheEvent,
    KVEventBatch,
    RecoveryUnitRemoved,
    RecoveryUnitStored,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    init_none_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler

from .utils import add_request, make_hybrid_scheduler, prompt, run, step

pytestmark = pytest.mark.cpu_test

BLOCK = 64
# A prefill cost model (seconds per token, and per token per 100,000 tokens
# of context) fitted on a hybrid model's prefill steps.
COST = (1.862e-5, 3.33e-6)
EVENTS = KVEventsConfig(enable_kv_cache_events=True, publisher="null")
UNIT_EVENTS = KVEventsConfig(
    enable_kv_cache_events=True, publisher="null", recovery_unit_events=True
)


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.fixture
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each read of the monotonic clock is a second after the last, so runs
    that read it equally often see the same times."""
    now = itertools.count(1_000_000.0, 1.0)
    monkeypatch.setattr(time, "monotonic", lambda: next(now))


class EventRecorder:
    """Takes the scheduler's KV cache event batches in place of a publisher."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.batches: list[list[KVCacheEvent]] = []

    def publish(self, batch: KVEventBatch) -> None:
        self.batches.append(list(batch.events))

    def shutdown(self) -> None:
        pass

    @property
    def events(self) -> list[KVCacheEvent]:
        """Every event so far, with those no step has published yet."""
        self.take_pending()
        return [event for batch in self.batches for event in batch]

    def take_pending(self) -> None:
        pending = self.scheduler.kv_cache_manager.take_events()
        if pending:
            self.batches.append(pending)


def make_scheduler(
    kv_events_config: KVEventsConfig | None,
    frontiers: bool = True,
    num_blocks: int = 90,
    max_num_batched_tokens: int = 512,
    hash_unit: int = BLOCK,
) -> tuple[Scheduler, EventRecorder]:
    """With `frontiers`, frontier checkpoints with leaf-last GreedyDual-Size
    eviction; without, prefix caching alone. `hash_unit` is the prefix-match
    unit."""
    cache_kwargs: dict[str, Any] = {}
    if hash_unit != BLOCK:
        cache_kwargs["prefix_match_unit"] = hash_unit
    if frontiers:
        cache_kwargs = dict(
            mamba_frontier_checkpoint=True,
            prefix_cache_eviction="gds",
            prefill_cost_coefficients=COST,
            prefix_cache_leaf_last=True,
            **cache_kwargs,
        )
    scheduler = make_hybrid_scheduler(
        block_size=BLOCK,
        hash_block_size=hash_unit,
        num_blocks=num_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        ring=0,
        kv_events_config=kv_events_config,
        **cache_kwargs,
    )
    recorder = EventRecorder(scheduler)
    scheduler.kv_event_publisher = recorder
    return scheduler, recorder


def drive(scheduler: Scheduler, turns: int = 30) -> list[Any]:
    """Multi-turn sessions, forks of earlier prefixes and new prompts, two at a
    time, in a pool that holds a few of them. Returns every step's scheduled
    tokens per request and the prefix-cache hits of the requests it admitted."""
    rng = random.Random(0)
    sessions = [prompt(100 + i, rng.randrange(80, 700)) for i in range(6)]
    decisions: list[Any] = []
    for turn in range(turns):
        batch = []
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
                session_id=f"s{i}",
            )
            batch.append((i, request))
        while not all(request.is_finished() for _, request in batch):
            output = step(scheduler)
            decisions.append(
                (
                    sorted(output.num_scheduled_tokens.items()),
                    sorted(
                        (new.req_id, new.num_computed_tokens)
                        for new in output.scheduled_new_reqs
                    ),
                )
            )
            assert len(decisions) < 5000
        for i, request in batch:
            tokens = list(request.all_token_ids)
            sessions[i] = tokens if len(tokens) < 2000 else prompt(200 + turn, 300)
    return decisions


def test_unit_events_need_kv_cache_events():
    scheduler, _ = make_scheduler(UNIT_EVENTS)
    assert scheduler.mamba_checkpoints is not None
    assert scheduler.mamba_checkpoints.unit_events
    # Without KV cache events the field publishes nothing.
    scheduler, _ = make_scheduler(KVEventsConfig(recovery_unit_events=True))
    assert scheduler.mamba_checkpoints is not None
    assert not scheduler.mamba_checkpoints.unit_events
    assert not scheduler.kv_cache_manager.block_pool.removal_causes


@pytest.mark.parametrize("config", [EVENTS, UNIT_EVENTS], ids=["off", "on"])
def test_without_frontier_checkpoints_kv_cache_events_are_unchanged(
    config: KVEventsConfig,
):
    """No removal names a cause and no new kind of event appears, with the
    field off or on."""
    scheduler, recorder = make_scheduler(config, frontiers=False)
    drive(scheduler, turns=10)
    events = recorder.events
    removed = [e for e in events if isinstance(e, BlockRemoved)]
    assert removed and all(e.cause is None for e in removed)
    for event in events:
        assert isinstance(event, BlockStored | BlockRemoved | AllBlocksCleared)
        assert getattr(event, "kv_cache_spec_kind", None) != "frontier"


@pytest.mark.usefixtures("fast_clock")
def test_recovery_unit_events_change_no_decision():
    """Eviction, admission and scheduling never read the events: the same
    sessions schedule the same tokens, hit the same prefixes and keep the same
    units with them as with KV cache events off."""
    quiet, _ = make_scheduler(None)
    published, recorder = make_scheduler(UNIT_EVENTS)
    assert drive(quiet) == drive(published)
    assert recorder.events
    a, b = quiet.mamba_checkpoints, published.mamba_checkpoints
    assert a is not None and a.eviction is not None
    assert b is not None and b.eviction is not None
    assert sorted(a.graph.units) == sorted(b.graph.units)
    assert a.eviction.num_evicted_units == b.eviction.num_evicted_units > 0


def test_a_finished_turn_publishes_its_frontier_and_the_next_supersedes_it():
    scheduler, recorder = make_scheduler(
        UNIT_EVENTS, num_blocks=200, max_num_batched_tokens=1024
    )
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "a", prompt(1, 1000), max_tokens=5, session_id="s")
    run(scheduler, first)
    events = recorder.events
    units = [e for e in events if isinstance(e, RecoveryUnitStored)]
    # A state at the prompt's last block boundary and a frontier at the end.
    assert [(e.num_tokens, e.frontier, e.session_id) for e in units] == [
        (960, False, "s"),
        (1004, True, "s"),
    ]
    frontier = units[1].unit_hash
    [entry] = [
        f
        for key, f in checkpoints.frontiers.entries.items()
        if maybe_convert_block_hash(key) == frontier
    ]
    stored = [
        e
        for e in events
        if isinstance(e, BlockStored) and e.kv_cache_spec_kind == "frontier"
    ]
    assert {e.group_idx for e in stored} == set(entry.blocks)
    for e in stored:
        assert e.block_hashes == [frontier]
        assert (e.token_ids, e.block_size, e.session_id) == ([], 1004, "s")
        assert e.parent_block_hash == maybe_convert_block_hash(
            first.block_hashes[1004 // BLOCK - 1]
        )
    assert not any(isinstance(e, RecoveryUnitRemoved) for e in events)
    assert not any(isinstance(e, BlockRemoved) and e.cause is None for e in events)

    tokens = list(first.all_token_ids) + prompt(9, 50)
    second = add_request(scheduler, "b", tokens, max_tokens=2, session_id="s")
    run(scheduler, second)
    events = recorder.events
    [removed] = [e for e in events if isinstance(e, RecoveryUnitRemoved)]
    assert (removed.unit_hash, removed.num_tokens, removed.frontier) == (
        frontier,
        1004,
        True,
    )
    assert removed.cause == "superseded"
    keys_removed = {
        e.group_idx
        for e in events
        if isinstance(e, BlockRemoved) and e.block_hashes == [frontier]
    }
    assert keys_removed == set(entry.blocks)
    assert all(
        e.cause == "superseded"
        for e in events
        if isinstance(e, BlockRemoved) and e.block_hashes == [frontier]
    )


def test_a_unit_whose_block_is_rewritten_goes_as_replaced():
    """When the block pool replaces a block's contents, the unit its key held
    goes with cause "replaced", including from the removal listener."""
    scheduler, recorder = make_scheduler(
        UNIT_EVENTS, num_blocks=200, max_num_batched_tokens=1024
    )
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    request = add_request(scheduler, "a", prompt(1, 1000), max_tokens=5)
    run(scheduler, request)
    pool = scheduler.kv_cache_manager.block_pool
    # A unit goes as soon as one of its Mamba groups loses its key.
    group_id = min(checkpoints.mamba_groups)
    key = request.block_hashes[960 // BLOCK - 1]
    assert key in checkpoints.graph.units
    block = pool.cached_block_hash_to_block.get_one_block(
        make_block_hash_with_group_id(key, group_id)
    )
    assert block is not None
    recorder.take_pending()
    pool.cache_partial_block(
        request=request,
        block=block,
        num_tokens=896,
        kv_cache_group_id=group_id,
        block_size=BLOCK,
        replace_existing_hashes=True,
    )
    assert key not in checkpoints.graph.units
    unit_hash = maybe_convert_block_hash(key)
    [removed] = [e for e in recorder.events if isinstance(e, RecoveryUnitRemoved)]
    assert (removed.unit_hash, removed.cause) == (unit_hash, "replaced")
    assert any(
        isinstance(e, BlockRemoved)
        and e.block_hashes == [unit_hash]
        and e.cause == "replaced"
        for e in recorder.events
    )


def test_a_decode_boundary_state_is_published_with_its_session():
    """A state the output crossed a block boundary with is cached from a
    scratch block, with no request: it still names the request's session."""
    scheduler, recorder = make_scheduler(
        UNIT_EVENTS, num_blocks=200, max_num_batched_tokens=1024
    )
    request = add_request(
        scheduler, "a", prompt(1, 1000), max_tokens=40, session_id="s"
    )
    run(scheduler, request)
    units = [e for e in recorder.events if isinstance(e, RecoveryUnitStored)]
    assert [(e.num_tokens, e.frontier, e.session_id) for e in units] == [
        (960, False, "s"),
        (1024, False, "s"),
        (1039, True, "s"),
    ]


@pytest.mark.parametrize("hash_unit", [BLOCK, 16])
def test_events_track_every_recovery_unit_and_frontier_key(hash_unit: int):
    """Replaying the events gives exactly the resumable units and frontier
    keys the cache holds: each is published when it is stored, with its
    session, and when it goes, with a cause. A unit goes with the cause of
    the removal of its own key, in the same step."""
    scheduler, recorder = make_scheduler(UNIT_EVENTS, hash_unit=hash_unit)
    drive(scheduler)
    recorder.take_pending()
    num_chosen = 0
    for batch in recorder.batches:
        key_causes = {
            (key, event.cause)
            for event in batch
            if isinstance(event, BlockRemoved)
            for key in event.block_hashes
        }
        for event in batch:
            if isinstance(event, RecoveryUnitRemoved):
                # A released unit may keep a block a running request holds.
                if event.cause != "superseded":
                    assert (event.unit_hash, event.cause) in key_causes
                if event.num_blocks is not None:
                    # The unit the eviction chose, by its tier.
                    assert event.cause == f"evicted-{event.tier or 'unranked'}"
                    assert event.num_blocks > 0
                    num_chosen += 1
    assert num_chosen > 0
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    units: dict[ExternalBlockHash, RecoveryUnitStored] = {}
    frontier_keys: set[tuple[ExternalBlockHash, int | None]] = set()
    ever_frontier: set[tuple[ExternalBlockHash, int | None]] = set()
    causes: set[str] = set()
    for event in recorder.events:
        if isinstance(event, RecoveryUnitStored):
            assert event.unit_hash not in units
            assert event.session_id in {f"s{i}" for i in range(6)}
            units[event.unit_hash] = event
        elif isinstance(event, RecoveryUnitRemoved):
            stored = units.pop(event.unit_hash)
            assert (event.num_tokens, event.frontier) == (
                stored.num_tokens,
                stored.frontier,
            )
            assert event.cause in REMOVAL_CAUSES
            causes.add(event.cause)
        elif isinstance(event, BlockStored) and event.kv_cache_spec_kind == "frontier":
            [key] = event.block_hashes
            assert event.token_ids == [] and event.session_id is not None
            assert (key, event.group_idx) not in frontier_keys
            frontier_keys.add((key, event.group_idx))
            ever_frontier.add((key, event.group_idx))
        elif isinstance(event, BlockRemoved):
            assert event.cause in REMOVAL_CAUSES
            for key in event.block_hashes:
                if (key, event.group_idx) in ever_frontier:
                    frontier_keys.remove((key, event.group_idx))
    assert set(units) == {maybe_convert_block_hash(k) for k in checkpoints.graph.units}
    cached = scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    assert frontier_keys == {
        (maybe_convert_block_hash(get_block_hash(k)), get_group_id(k))
        for k in cached._cache
        if get_block_hash(k) in checkpoints.frontiers.entries
    }
    assert "superseded" in causes
    assert any(cause.startswith("evicted-") for cause in causes)


class _BlockRemovedBefore(KVCacheEvent, tag="BlockRemoved"):  # type: ignore[call-arg]
    """`BlockRemoved` as consumers written before `cause` decode it."""

    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None
    ownership: str | None = None


def test_a_removal_without_a_cause_encodes_as_before():
    encoder = msgspec.msgpack.Encoder()
    removal = BlockRemoved(block_hashes=[7], medium="GPU", group_idx=1)
    assert encoder.encode(removal) == encoder.encode(
        _BlockRemovedBefore(block_hashes=[7], medium="GPU", group_idx=1)
    )
    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockRemoved(block_hashes=[7], medium="GPU", group_idx=1, cause="orphan"),
            RecoveryUnitStored(unit_hash=7, num_tokens=1004, frontier=True),
            RecoveryUnitRemoved(
                unit_hash=7,
                num_tokens=1004,
                frontier=True,
                cause="evicted-leaf",
                tier="leaf",
                num_blocks=3,
            ),
        ],
    )
    assert msgspec.msgpack.decode(encoder.encode(batch), type=KVEventBatch) == batch
