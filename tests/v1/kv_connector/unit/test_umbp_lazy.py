# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy offload: store cached blocks only once they near eviction.

The scheduler scans the eviction end of the real `BlockPool` free queue, so
these tests drive a real pool and real `Request` objects through it.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler as umbp_scheduler
from tests.v1.core.utils import new_kv_cache_group
from tests.v1.kv_connector.umbp_test_utils import (
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    complete_store,
    make_scheduler_output,
)
from tests.v1.kv_connector.unit.utils import create_vllm_config, set_mtp_config
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    RankTopology,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request

_BLOCK_SIZE = 16


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        umbp_scheduler, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    return clock


@pytest.fixture
def lazy_scheduler(clock):
    """Build lazy schedulers bound to a real `BlockPool`; close them all."""
    made = []

    def make(
        handle=None,
        *,
        vllm_config=None,
        kv_cache_config=None,
        codec=None,
        topology=None,
        num_gpu_blocks=8,
        **extra,
    ):
        scheduler = UMBPStoreConnectorScheduler(
            vllm_config
            or _vllm_config({"mode": "embedded", "lazy_offload": True, **extra}),
            kv_cache_config or _kv_cache_config(),
            _SchedulerHandle({}) if handle is None else handle,
            codec or BlockIdentityCodec(UMBPNamespace("lazy")),
            topology,
        )
        made.append(scheduler)
        pool = BlockPool(
            num_gpu_blocks, enable_caching=True, hash_block_size=_BLOCK_SIZE
        )
        scheduler.bind_gpu_block_pool(pool)
        return scheduler, pool

    yield make
    for scheduler in made:
        scheduler.close()


def _request(request_id: str, num_blocks: int, first_token: int = 0) -> Request:
    """A prompt of `num_blocks` full blocks; `first_token` sets its hashes."""
    init_none_hash(sha256)
    return Request(
        request_id,
        list(range(first_token, first_token + num_blocks * _BLOCK_SIZE)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )


def _cache(pool: BlockPool, request: Request, blocks, num_cached: int = 0) -> None:
    pool.cache_full_blocks(request, blocks, num_cached, len(blocks), _BLOCK_SIZE, 0)


def _finish(
    scheduler,
    pool: BlockPool,
    request: Request,
    blocks: list[KVCacheBlock] | None = None,
    num_cached: int = 0,
) -> list[KVCacheBlock]:
    """Compute `request` into `blocks`, then finish and free it as core does."""
    if blocks is None:
        blocks = pool.get_new_blocks(len(request.block_hashes))
    request.num_computed_tokens = request.num_tokens
    _cache(pool, request, blocks, num_cached)
    scheduler.request_finished(request, ([block.block_id for block in blocks],))
    pool.free_blocks(reversed(blocks))
    return blocks


def _recycle(pool: BlockPool, request: Request) -> KVCacheBlock:
    """Evict the next free block and cache `request`'s first block in it."""
    [block] = pool.get_new_blocks(1)
    _cache(pool, request, [block])
    return block


def _alias(pool: BlockPool, block: KVCacheBlock, request: Request) -> None:
    """Make `block` also reachable from `request`'s first block hash."""
    pool.cache_partial_block(request, block, _BLOCK_SIZE, 0, _BLOCK_SIZE)


def _build(scheduler, output=None):
    return scheduler.build_connector_meta(output or make_scheduler_output())


def _planned(metadata) -> list[tuple[int, str]]:
    return [(plan.block_id, plan.key) for plan in metadata.store_plans]


def _key(scheduler, request: Request, index: int = 0) -> str:
    return scheduler.codec.key(request.block_hashes[index], 0)


def _free_queue(pool: BlockPool) -> list[int]:
    return [block.block_id for block in pool.free_block_queue.iter_blocks_after(None)]


def _take_empty_blocks(pool: BlockPool) -> None:
    """Allocate every uncached free block, as a busy engine would."""
    empty = [
        block
        for block in pool.free_block_queue.iter_blocks_after(None)
        if block.block_hash is None
    ]
    pool.get_new_blocks(len(empty))


def test_lazy_offload_defers_store_until_blocks_enter_free_queue(lazy_scheduler):
    scheduler, pool = lazy_scheduler()
    request = _request("r", 2)
    blocks = pool.get_new_blocks(2)
    block_ids = ([block.block_id for block in blocks],)
    scheduler.update_state_after_alloc(request, KVCacheBlocks((blocks,)), 0)
    running = make_scheduler_output(
        new_reqs=[
            NewRequestData.from_request(request, block_ids, request.prompt_token_ids)
        ],
        scheduled_tokens={"r": 32},
    )

    assert _build(scheduler, running).store_plans == []

    head, tail = _finish(scheduler, pool, request, blocks)

    assert _planned(_build(scheduler)) == [
        (tail.block_id, _key(scheduler, request, 1)),
        (head.block_id, _key(scheduler, request, 0)),
    ]


def test_lazy_store_pins_blocks_until_the_store_completes(lazy_scheduler):
    """The worker reads in-flight blocks, so core must not hand them out."""
    scheduler, pool = lazy_scheduler()
    _finish(scheduler, pool, _request("r", 2))
    metadata = _build(scheduler)
    planned = {plan.block_id for plan in metadata.store_plans}
    assert len(planned) == 2

    assert not planned & set(_free_queue(pool))
    handed_out = pool.get_new_blocks(pool.get_num_free_blocks())
    assert not planned & {block.block_id for block in handed_out}

    complete_store(scheduler, metadata.store_event)
    assert set(_free_queue(pool)) == planned


def test_lazy_offload_rechecks_residency_after_the_pool_evicts(lazy_scheduler, clock):
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(handle)
    request = _request("r", 1)
    _finish(scheduler, pool, request)
    key = _key(scheduler, request)

    first = _build(scheduler)
    assert [plan.key for plan in first.store_plans] == [key]
    complete_store(scheduler, first.store_event)
    handle.hits[key] = True
    # The completed store confirmed the key, so steps before the recheck do not
    # look it up again.
    for _ in range(3):
        assert _build(scheduler).store_plans == []
    assert len(handle.queries) == 1

    handle.hits[key] = False
    clock.now += umbp_scheduler._LAZY_RECHECK_S
    again = _build(scheduler)

    assert [plan.key for plan in again.store_plans] == [key]
    assert len(handle.queries) == 2


def test_lazy_offload_repairs_a_missing_nonzero_rank(lazy_scheduler, clock):
    topology = RankTopology(tp_size=2)
    codec = BlockIdentityCodec(UMBPNamespace("lazy"))
    request = _request("r", 1)
    rank_keys = codec.keys_for_topology(request.block_hashes[0], topology, (0,))
    handle = _SchedulerHandle({rank_keys[0]: True})
    scheduler, pool = lazy_scheduler(
        handle,
        vllm_config=_vllm_config(
            {"lazy_offload": True}, tensor_parallel_size=2, world_size=2
        ),
        codec=codec,
        topology=topology,
    )
    [block] = _finish(scheduler, pool, request)

    first = _build(scheduler)
    assert [plan.block_id for plan in first.store_plans] == [block.block_id]
    assert handle.queries == [list(rank_keys)]
    complete_store(scheduler, first.store_event, workers=2)

    # Rank 1 still misses the object at the recheck, so it is stored again.
    clock.now += umbp_scheduler._LAZY_RECHECK_S
    repair = _build(scheduler)
    assert [plan.block_id for plan in repair.store_plans] == [block.block_id]
    complete_store(scheduler, repair.store_event, workers=2)

    handle.hits = dict.fromkeys(rank_keys, True)
    clock.now += umbp_scheduler._LAZY_RECHECK_S
    assert _build(scheduler).store_plans == []
    assert len(handle.queries) == 3


class _UnreachableHandle(_SchedulerHandle):
    """Fails lookups until `down` is cleared: per-key `None`, or raising."""

    def __init__(self, raises: bool):
        super().__init__({})
        self.raises = raises
        self.down = True

    def lookup(self, keys):
        if not self.down:
            return super().lookup(keys)
        self.queries.append(list(keys))
        if self.raises:
            raise ConnectionError("UMBP server unreachable")
        return [None] * len(keys)


@pytest.mark.parametrize("raises", [False, True], ids=["unavailable", "raises"])
def test_lazy_offload_backs_off_while_the_pool_is_unavailable(
    lazy_scheduler, clock, raises
):
    """An unreachable pool is not a miss: back off instead of storing."""
    handle = _UnreachableHandle(raises)
    scheduler, pool = lazy_scheduler(handle)
    [block] = _finish(scheduler, pool, _request("r", 1))

    assert _build(scheduler).store_plans == []
    assert scheduler._lazy_lookup_failures == 1
    # Until the backoff passes, steps neither look up nor keep the engine busy.
    assert _build(scheduler).store_plans == []
    assert len(handle.queries) == 1
    assert not scheduler.has_pending_push_work()

    handle.down = False
    clock.now += umbp_scheduler._LAZY_RETRY_INITIAL_S
    retried = _build(scheduler)

    assert [plan.block_id for plan in retried.store_plans] == [block.block_id]
    assert scheduler._lazy_lookup_failures == 0


def test_lazy_offload_backs_off_failed_stores_on_a_dead_server(lazy_scheduler, clock):
    """A killed standalone server answers lookups with misses and fails stores."""
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(handle, mode="standalone")
    request = _request("r", 1)
    [block] = _finish(scheduler, pool, request)

    first = _build(scheduler)
    assert [plan.block_id for plan in first.store_plans] == [block.block_id]
    complete_store(scheduler, first.store_event, failed={(0, request.block_hashes[0])})
    # Until the backoff passes, the window is not stored again.
    assert _build(scheduler).store_plans == []
    assert len(handle.queries) == 1
    assert not scheduler.has_pending_push_work()

    clock.now += umbp_scheduler._LAZY_RETRY_INITIAL_S
    retried = _build(scheduler)

    assert [plan.block_id for plan in retried.store_plans] == [block.block_id]


def test_lazy_offload_includes_secondary_block_hashes(lazy_scheduler):
    scheduler, pool = lazy_scheduler()
    primary, secondary = _request("a", 1), _request("b", 1, first_token=100)
    [block] = _finish(scheduler, pool, primary)
    _alias(pool, block, secondary)

    assert set(_planned(_build(scheduler))) == {
        (block.block_id, _key(scheduler, primary)),
        (block.block_id, _key(scheduler, secondary)),
    }


def test_lazy_offload_rescans_a_block_that_gains_an_alias(lazy_scheduler):
    """An aliased block never settles, since it can gain keys while reused."""
    scheduler, pool = lazy_scheduler()
    [block] = _finish(scheduler, pool, _request("a", 1))
    _alias(pool, block, _request("b", 1, first_token=100))
    metadata = _build(scheduler)
    complete_store(scheduler, metadata.store_event)
    assert _build(scheduler).store_plans == []

    gained = _request("c", 1, first_token=200)
    _alias(pool, block, gained)
    # The block stays put, so a window-moving step (here a preemption) arms the
    # rescan that picks up its new key.
    metadata = _build(scheduler, make_scheduler_output(preempted=["other"]))

    assert _planned(metadata) == [(block.block_id, _key(scheduler, gained))]


@pytest.mark.parametrize("aliased_head", [False, True])
def test_lazy_offload_closes_full_attention_prefix(lazy_scheduler, aliased_head):
    """A tail in the window pulls its prefix head into the store, tail first."""
    scheduler, pool = lazy_scheduler(lazy_offload_max_blocks=1)
    request = _request("r", 2)
    head, tail = pool.get_new_blocks(2)
    if aliased_head:
        # The head was cached under another hash first; ours is its alias.
        _cache(pool, _request("other", 1, first_token=100), [head])
        _alias(pool, head, request)
    _finish(scheduler, pool, request, [head, tail], num_cached=int(aliased_head))
    _take_empty_blocks(pool)
    assert _free_queue(pool) == [tail.block_id, head.block_id]

    assert _planned(_build(scheduler)) == [
        (tail.block_id, _key(scheduler, request, 1)),
        (head.block_id, _key(scheduler, request, 0)),
    ]


def test_lazy_offload_drops_recycled_prefix_chain_tail(lazy_scheduler):
    """A recycled tail must not pull in the old chain's head."""
    scheduler, pool = lazy_scheduler(num_gpu_blocks=3)
    head, tail = _finish(scheduler, pool, _request("old", 2))
    # A running request hits the head, so core recycles the tail.
    pool.touch([head])
    new = _request("new", 1, first_token=100)
    assert _recycle(pool, new) is tail
    pool.free_blocks([tail])

    assert _planned(_build(scheduler)) == [(tail.block_id, _key(scheduler, new))]
    assert tail.block_id not in scheduler._lazy_prefix_chains_by_block


def test_lazy_offload_skips_recycled_prefix_head(lazy_scheduler):
    """A recycled head is not stored under the old chain's key."""
    scheduler, pool = lazy_scheduler(num_gpu_blocks=3)
    old = _request("old", 2)
    head, tail = _finish(scheduler, pool, old)
    # A running request hits the tail, so core recycles the head.
    pool.touch([tail])
    new = _request("new", 1, first_token=100)
    assert _recycle(pool, new) is head
    pool.free_blocks([head])
    pool.free_blocks([tail])

    assert _planned(_build(scheduler)) == [
        (head.block_id, _key(scheduler, new)),
        (tail.block_id, _key(scheduler, old, 1)),
    ]


def test_lazy_offload_rescans_a_recycled_settled_block(lazy_scheduler):
    old, new = _request("old", 1), _request("new", 1, first_token=100)
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(handle, num_gpu_blocks=2)
    handle.hits[_key(scheduler, old)] = True
    [settled] = _finish(scheduler, pool, old)
    assert _build(scheduler).store_plans == []

    [recycled] = _finish(scheduler, pool, new)
    assert recycled is settled

    assert _planned(_build(scheduler)) == [(recycled.block_id, _key(scheduler, new))]


@pytest.mark.parametrize("activity", ["new_request", "preempted", "new_blocks"])
def test_lazy_offload_scans_when_allocation_moves_the_window(lazy_scheduler, activity):
    """Allocation pushes cached blocks toward eviction without any finish."""
    scheduler, pool = lazy_scheduler(num_gpu_blocks=4, lazy_offload_max_blocks=1)
    request = _request("r", 1)
    [block] = _finish(scheduler, pool, request)
    assert _build(scheduler).store_plans == []

    other = _request("other", 2, first_token=100)
    block_ids = ([block.block_id for block in pool.get_new_blocks(2)],)
    if activity == "new_request":
        output = make_scheduler_output(
            new_reqs=[
                NewRequestData.from_request(other, block_ids, other.prompt_token_ids)
            ],
            scheduled_tokens={"other": 32},
        )
    elif activity == "preempted":
        output = make_scheduler_output(preempted=["other"])
    else:
        output = make_scheduler_output(
            cached_reqs=CachedRequestData(["other"], [[]], [block_ids], [16], [0]),
            scheduled_tokens={"other": 16},
        )

    assert _planned(_build(scheduler, output)) == [
        (block.block_id, _key(scheduler, request))
    ]


def test_lazy_offload_rescans_after_pinning_the_window(lazy_scheduler):
    """Pinning the window moves the next block into it within the same tick."""
    scheduler, pool = lazy_scheduler(num_gpu_blocks=3, lazy_offload_max_blocks=1)
    [first] = _finish(scheduler, pool, _request("first", 1))
    [second] = _finish(scheduler, pool, _request("second", 1, first_token=100))

    assert [plan.block_id for plan in _build(scheduler).store_plans] == [first.block_id]
    assert [plan.block_id for plan in _build(scheduler).store_plans] == [
        second.block_id
    ]


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_lazy_offload_does_not_store_running_requests(lazy_scheduler, lazy):
    """Eager mode stores each prefill chunk; lazy mode waits for the free queue."""
    scheduler, pool = lazy_scheduler(lazy_offload=lazy)
    request = _request("r", 3)
    blocks = pool.get_new_blocks(3)
    scheduler.update_state_after_alloc(request, KVCacheBlocks((blocks[:1],)), 0)
    _build(
        scheduler,
        make_scheduler_output(
            new_reqs=[
                NewRequestData.from_request(
                    request, ([blocks[0].block_id],), request.prompt_token_ids
                )
            ],
            scheduled_tokens={"r": 16},
        ),
    )
    new_block_ids = [block.block_id for block in blocks[1:]]
    chunk = make_scheduler_output(
        cached_reqs=CachedRequestData(["r"], [[]], [(new_block_ids,)], [16], [0]),
        scheduled_tokens={"r": 32},
    )

    planned = [plan.block_id for plan in _build(scheduler, chunk).iter_store_plans()]

    assert planned == ([] if lazy else new_block_ids)


def test_lazy_offload_does_not_resubmit_an_in_flight_key(lazy_scheduler):
    """Two blocks with one key are stored once, even across steps."""
    scheduler, pool = lazy_scheduler()
    _finish(scheduler, pool, _request("a", 1))
    _finish(scheduler, pool, _request("b", 1))

    assert len(_build(scheduler).store_plans) == 1
    assert _build(scheduler).store_plans == []


def test_lazy_offload_skips_non_transferable_groups(lazy_scheduler):
    spec = AttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["transferred"], spec),
            new_kv_cache_group(["local"], spec, enable_kv_transfer=False),
        ],
    )
    scheduler, pool = lazy_scheduler(kv_cache_config=config)
    request = _request("r", 1)
    transferred, local = pool.get_new_blocks(2)
    _cache(pool, request, [transferred])
    pool.cache_full_blocks(request, [local], 0, 1, _BLOCK_SIZE, 1)
    pool.free_blocks([transferred, local])

    assert _planned(_build(scheduler)) == [
        (transferred.block_id, _key(scheduler, request))
    ]


def test_lazy_target_matches_cache_group_geometry():
    config = _hybrid_kv_cache_config()
    swa = SlidingWindowManager.layer_config(
        AttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16),
        sliding_window=40,
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[*config.kv_cache_groups, new_kv_cache_group(["swa"], swa)],
    )

    target = UMBPStoreConnectorScheduler._estimate_lazy_target_blocks(
        config, max_num_batched_tokens=64, dcp_size=1
    )

    # Twice the blocks one step can evict: full attention + Mamba + SWA.
    assert target == 2 * (cdiv(64, 16) + 2 + (cdiv(40, 16) + 1))


def test_lazy_offload_is_disabled_for_consumer_role(lazy_scheduler):
    config = _vllm_config({"mode": "embedded", "lazy_offload": True})
    config.kv_transfer_config.kv_role = "kv_consumer"
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(handle, vllm_config=config)
    _finish(scheduler, pool, _request("r", 1))

    assert _build(scheduler).store_plans == []
    assert handle.queries == []


@pytest.mark.parametrize("value", [-1, 1.5, "4", True])
def test_lazy_offload_rejects_invalid_scan_limit(lazy_scheduler, value):
    with pytest.raises(ValueError, match="lazy_offload_max_blocks"):
        lazy_scheduler(lazy_offload_max_blocks=value)


def test_lazy_offload_zero_scan_limit_disables_the_scan(lazy_scheduler):
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(handle, lazy_offload_max_blocks=0)
    _finish(scheduler, pool, _request("r", 1))

    assert _build(scheduler).store_plans == []
    assert handle.queries == []


def test_lazy_offload_looks_up_tp_replicated_groups_once(lazy_scheduler):
    codec = BlockIdentityCodec(UMBPNamespace("lazy"), replicated_groups=frozenset({0}))
    topology = RankTopology(tp_size=2)
    request = _request("r", 1)
    keys = codec.keys_for_topology(request.block_hashes[0], topology, (0,))
    handle = _SchedulerHandle({})
    scheduler, pool = lazy_scheduler(
        handle,
        vllm_config=_vllm_config(
            {"lazy_offload": True}, tensor_parallel_size=2, world_size=2
        ),
        codec=codec,
        topology=topology,
    )
    [block] = _finish(scheduler, pool, request)

    metadata = _build(scheduler)

    assert len(keys) == 1
    assert handle.queries == [list(keys)]
    assert [plan.block_id for plan in metadata.store_plans] == [block.block_id]
    assert scheduler._lazy_lookup_failures == 0


def test_lazy_store_returns_blocks_behind_free_empty_blocks(lazy_scheduler):
    scheduler, pool = lazy_scheduler(num_gpu_blocks=40, lazy_offload_max_blocks=40)
    a = _finish(scheduler, pool, _request("a", 4))
    b = _finish(scheduler, pool, _request("b", 4, first_token=100))
    before = _free_queue(pool)

    metadata = _build(scheduler)
    assert {plan.block_id for plan in metadata.store_plans} == {
        block.block_id for block in a + b
    }
    complete_store(scheduler, metadata.store_event)

    assert _free_queue(pool) == before
    # A one-block allocation takes an empty block, not A's oldest one.
    assert pool.get_new_blocks(1)[0].block_hash is None
    assert all(block.block_hash is not None for block in a)


def test_lazy_store_frees_pulled_in_prefix_blocks_to_the_mru_end(lazy_scheduler):
    scheduler, pool = lazy_scheduler(num_gpu_blocks=40, lazy_offload_max_blocks=1)
    head, middle, tail = _finish(scheduler, pool, _request("x", 3))
    others = [
        _finish(scheduler, pool, _request(f"c{index}", 1, first_token=100 * index))[0]
        for index in range(1, 4)
    ]
    # Another request hits the prefix, so it moves to the MRU end.
    pool.touch([head, middle])
    pool.free_blocks([middle, head])
    _take_empty_blocks(pool)
    before = _free_queue(pool)
    assert before == [
        tail.block_id,
        *(other.block_id for other in others),
        middle.block_id,
        head.block_id,
    ]

    metadata = _build(scheduler)
    assert [plan.block_id for plan in metadata.store_plans] == [
        tail.block_id,
        middle.block_id,
        head.block_id,
    ]
    complete_store(scheduler, metadata.store_event)

    assert _free_queue(pool) == before


def test_lazy_prefix_chain_skips_blocks_without_lookahead_hash(lazy_scheduler):
    """A block whose lookahead tokens never arrived has no hash to chain."""
    config = create_vllm_config(
        kv_connector="UMBPStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lazy_offload": True},
    )
    set_mtp_config(config)
    kv_config = _kv_cache_config()
    kv_config.hash_block_size = _BLOCK_SIZE
    scheduler, _ = lazy_scheduler(vllm_config=config, kv_cache_config=kv_config)
    assert scheduler.lazy_offload
    init_none_hash(sha256)
    request = Request(
        "r",
        list(range(2 * _BLOCK_SIZE)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256, 1),
    )
    # Aborted before sampling: the second block's hash needs the next token.
    request.num_computed_tokens = 2 * _BLOCK_SIZE
    assert len(request.block_hashes) == 1
    scheduler.request_finished(request, ([3, 4],))
    assert set(scheduler._lazy_prefix_chains_by_block) == {3}
