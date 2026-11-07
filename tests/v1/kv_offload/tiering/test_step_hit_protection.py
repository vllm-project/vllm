# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a promotion must not evict a primary HIT that a lookup
already reported in the same scheduler step.

Production crash (v0.32.0, TieringOffloadingSpec with an fs secondary tier):
    AssertionError: Chunk ... not found in cache   (cpu/manager.py:221)
raised from prepare_load() in the scheduler's update_state_after_alloc().
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
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
    manager.prepare_store(resident, _CTX)
    manager.complete_store(resident, _CTX, success=True)
    _end_step(manager)
    _end_step(manager)
    assert primary._num_evictable_cache_chunks == _NUM_PRIMARY

    # A chunk that only lives in the secondary tier.
    secondary_only = _keys([100])[0]
    tier.chunks[secondary_only] = True
    return manager, resident, secondary_only


def test_promotion_does_not_evict_a_hit_reported_this_step(setup):
    manager, resident, secondary_only = setup
    oldest = resident[0]  # first LRU eviction candidate

    # Same step: a lookup counts `oldest` as HIT, then a later lookup hits
    # the secondary tier and starts a promotion that needs one primary slot.
    assert manager.lookup(oldest, _CTX) is LookupResult.HIT
    assert manager.lookup(secondary_only, _CTX) is LookupResult.HIT_PENDING

    # The scheduler now loads what it counted. Before the fix this raised
    # "Chunk ... not found in cache" because the promotion evicted `oldest`.
    manager.prepare_load([oldest], _CTX)
    manager.complete_load([oldest], _CTX)


def test_promotion_degrades_to_miss_when_every_idle_chunk_was_hit(setup):
    manager, resident, secondary_only = setup
    for key in resident:
        assert manager.lookup(key, _CTX) is LookupResult.HIT

    # No idle chunk may be evicted, so the promotion cannot allocate a slot
    # and the key is reported as a plain MISS (recomputed), not a crash.
    assert manager.lookup(secondary_only, _CTX) is LookupResult.MISS

    manager.prepare_load(resident, _CTX)
    manager.complete_load(resident, _CTX)


def test_protection_is_released_at_schedule_end(setup):
    manager, resident, secondary_only = setup
    assert manager.lookup(resident[0], _CTX) is LookupResult.HIT
    _end_step(manager)

    # Next step: nothing was counted yet, so the promotion may evict again.
    assert manager.lookup(secondary_only, _CTX) is LookupResult.HIT_PENDING
    assert not manager._step_hit_keys
