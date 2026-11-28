# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests: a reported primary HIT must survive until its consumer
takes it over, and a request that never consumes its HITs must not leak pins.

Production crash (v0.32.0, TieringOffloadingSpec with an fs secondary tier,
5 engine crashes on 2026-10-24):

    AssertionError: Chunk ... not found in cache   (cpu/manager.py, _prepare_load)

A fs->RAM promotion allocates a primary slot with prepare_write(), which
evicts an idle chunk. When the evicted chunk is one that a lookup of the
same scheduler step already reported as HIT, the later prepare_load() (or
create_store_job()'s prepare_read()) asserts on the missing chunk.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    StoreTrigger,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="test")
_NUM_PRIMARY = 5


def _keys(ids):
    return [make_offload_key(str(i).encode(), 0) for i in ids]


def _end_step(manager):
    manager.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    list(manager.take_events())


@pytest.fixture
def setup():
    region = MagicMock()
    view = memoryview(torch.zeros((_NUM_PRIMARY, 16), dtype=torch.int8).numpy())
    region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=_NUM_PRIMARY, mmap_region=region
    )
    tier = ExampleSecondaryTierManager(
        offloading_spec=MagicMock(), primary_kv_view=view, tier_type="example"
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    manager.on_new_request(_CTX)

    # Fill the primary tier, then let the cascades finish so every primary
    # chunk is idle (ref_cnt 0) and evictable.
    resident = _keys(range(_NUM_PRIMARY))
    manager.prepare_store(resident, _CTX, trigger=StoreTrigger.ON_COMPUTE)
    manager.complete_store(
        resident, _CTX, trigger=StoreTrigger.ON_COMPUTE, success=True
    )
    _end_step(manager)
    _end_step(manager)
    assert primary._num_evictable_cache_chunks == _NUM_PRIMARY

    # A chunk that only lives in the secondary tier.
    secondary_only = _keys([100])[0]
    tier.chunks[secondary_only] = True
    return manager, primary, tier, resident, secondary_only


def test_promotion_does_not_evict_a_hit_reported_this_step(setup):
    """The production race: within one step, a lookup counts a primary HIT,
    a later lookup starts a promotion that needs one primary slot, then the
    scheduler loads what it counted."""
    manager, primary, tier, resident, secondary_only = setup
    oldest = resident[0]  # first LRU eviction candidate

    assert manager.lookup(oldest, _CTX) is LookupResult.HIT
    assert manager.lookup(secondary_only, _CTX) is LookupResult.HIT_PENDING

    # Before the fix the promotion had evicted `oldest` and this raised
    # "Chunk ... not found in cache".
    spec = manager.prepare_load([oldest], _CTX)

    # 1 -> 2 -> 1: the load holds its own ref, the lookup pin is released.
    assert primary._policy.get(oldest).ref_cnt == 1
    assert len(spec.chunk_ids) == 1
    assert primary._policy.get(oldest).chunk_id == spec.chunk_ids[0]

    manager.complete_load([oldest], _CTX)
    assert primary._policy.get(oldest).ref_cnt == 0


def test_create_store_job_after_lookups(setup):
    """The P2P/KVCR shape: a caller looks keys up through the manager and
    pins the HITs with create_store_job() (see ParentManager)."""
    manager, primary, tier, resident, secondary_only = setup
    oldest = resident[0]
    ctx = ReqContext(req_id="external:0")
    manager.on_new_request(ctx)

    # Same walk as KVCRSecondaryTierManager._resolve_pin: collect HITs,
    # letting later lookups initiate promotions in between.
    hit_keys = []
    for key in [oldest, secondary_only]:
        if manager.lookup(key, ctx) is LookupResult.HIT:
            hit_keys.append(key)

    # Before the fix the promotion had evicted `oldest` and prepare_read()
    # inside create_store_job raised "Chunk ... not found in cache".
    job = manager.create_store_job(hit_keys, ctx)
    assert list(job.keys) == [oldest]

    # 1 -> 2 -> 1: the job holds its own ref, the lookup pin is released.
    assert primary._policy.get(oldest).ref_cnt == 1

    # Finish the job the way a tier does: the completion's complete_read()
    # releases the job's ref. The first step-end submits the pending
    # promotion and skips the poll (its gate is held), the second drains
    # both completions.
    tier.completed_jobs.append(JobResult(job_id=job.job_id, success=True))
    _end_step(manager)
    _end_step(manager)
    assert primary._policy.get(oldest).ref_cnt == 0


def test_aborted_request_releases_its_pins(setup):
    """A request that reports a HIT and is then aborted (no prepare_load,
    no create_store_job) must not leak its pins: the primary tier would
    silently stop evicting those chunks."""
    manager, primary, tier, resident, secondary_only = setup
    oldest = resident[0]
    ctx_aborted = ReqContext(req_id="aborted")
    manager.on_new_request(ctx_aborted)

    assert manager.lookup(oldest, ctx_aborted) is LookupResult.HIT
    assert manager.lookup(secondary_only, ctx_aborted) is LookupResult.HIT_PENDING
    assert primary._policy.get(oldest).ref_cnt == 1  # pinned by the lookup

    # The request disappears. The next request's first HIT releases the
    # previous request's pins; the chunk becomes evictable again.
    ctx_next = ReqContext(req_id="next")
    manager.on_new_request(ctx_next)
    assert manager.lookup(resident[2], ctx_next) is LookupResult.HIT

    assert primary._policy.get(oldest).ref_cnt == 0
    # Observable release (not just bookkeeping): a promotion may evict the
    # chunk again. The secondary tier holds every resident key, so another
    # secondary-only key promotes and takes `oldest`'s slot.
    another_secondary = _keys([101])[0]
    tier.chunks[another_secondary] = True
    assert manager.lookup(another_secondary, ctx_next) is LookupResult.HIT_PENDING
    assert primary._policy.get(oldest) is None
    assert manager._lookup_hits_owner is None or (
        manager._lookup_hits_owner.req_id == "next"
    )


def test_pins_are_released_at_schedule_end(setup):
    """Whatever a step pinned and never consumed is released at
    on_schedule_end(): the chunk is evictable again on the next step."""
    manager, primary, tier, resident, secondary_only = setup
    oldest = resident[0]

    assert manager.lookup(oldest, _CTX) is LookupResult.HIT
    # The promotion this lookup starts is what pins `oldest` (ref 1).
    assert manager.lookup(secondary_only, _CTX) is LookupResult.HIT_PENDING
    assert primary._policy.get(oldest).ref_cnt == 1
    _end_step(manager)

    assert primary._policy.get(oldest).ref_cnt == 0
    assert manager._lookup_hits_owner is None
    assert not manager._lookup_hits
    assert not manager._pinned_lookup_hits

    # Next step: nothing was consumed, so a promotion may evict it again.
    another_secondary = _keys([101])[0]
    tier.chunks[another_secondary] = True
    _end_step(manager)  # drain the pending promotion from the lookup above
    assert manager.lookup(another_secondary, _CTX) is LookupResult.HIT_PENDING
    assert primary._policy.get(oldest) is None


def test_pins_are_released_at_poll_round_end(setup):
    """A tier-poll round serves external requests; anything its lookups
    left tracked is released at the end of the round."""
    manager, primary, tier, resident, secondary_only = setup
    oldest = resident[0]
    ctx = ReqContext(req_id="external:1")
    manager.on_new_request(ctx)

    assert manager.lookup(oldest, ctx) is LookupResult.HIT
    # The promotion this lookup starts is what pins `oldest` (ref 1).
    assert manager.lookup(secondary_only, ctx) is LookupResult.HIT_PENDING
    assert primary._policy.get(oldest).ref_cnt == 1

    manager.poll_tiers()

    assert primary._policy.get(oldest).ref_cnt == 0
    assert manager._lookup_hits_owner is None


def test_promotion_degrades_to_miss_when_every_idle_chunk_was_hit(setup):
    """When every evictable chunk was a reported HIT this step, the
    promotion cannot allocate a slot and the key is reported as a plain
    MISS (recompute), not a crash."""
    manager, primary, tier, resident, secondary_only = setup
    for key in resident:
        assert manager.lookup(key, _CTX) is LookupResult.HIT

    assert manager.lookup(secondary_only, _CTX) is LookupResult.MISS

    manager.prepare_load(resident, _CTX)
    manager.complete_load(resident, _CTX)
    for key in resident:
        assert primary._policy.get(key).ref_cnt == 0


def test_reset_cache_asserts_when_tracking_survives_the_step(setup):
    """reset_cache() never runs inside a schedule step, so lookup-hit
    tracking that survived one is a pin leak and must fail loudly."""
    manager, primary, tier, resident, secondary_only = setup
    assert manager.lookup(resident[0], _CTX) is LookupResult.HIT

    with pytest.raises(AssertionError, match="lookup-hit tracking"):
        manager.reset_cache()


def test_reset_cache_passes_after_a_completed_step(setup):
    manager, primary, tier, resident, secondary_only = setup
    assert manager.lookup(resident[0], _CTX) is LookupResult.HIT
    _end_step(manager)
    manager.reset_cache()
