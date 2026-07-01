# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


REQ = ReqContext(req_id="test")


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_emit_four_counters_with_policy_label(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    # Two misses
    assert mgr.lookup(key(1), REQ) == LookupResult.MISS
    assert mgr.lookup(key(2), REQ) == LookupResult.MISS

    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,)) == 2
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_HIT, {}).get((policy,)) == 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_MISS, {}).get((policy,)) == 2
    assert data.get(CPUOffloadingMetrics.BLOCK_EVICTION, {}).get((policy,)) == 0


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_hits_plus_misses_equals_lookups(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    # Fill one block so a hit is possible
    ps = mgr.prepare_store([key(1)], REQ)
    assert ps is not None
    mgr.complete_store([key(1)], REQ, success=True)
    # 1 hit, 2 misses -> 3 lookups
    mgr.lookup(key(1), REQ)
    mgr.lookup(key(2), REQ)
    mgr.lookup(key(3), REQ)

    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    lookups = data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,), 0)
    hits = data.get(CPUOffloadingMetrics.CPU_BLOCK_HIT, {}).get((policy,), 0)
    misses = data.get(CPUOffloadingMetrics.CPU_BLOCK_MISS, {}).get((policy,), 0)
    assert lookups == hits + misses
    assert hits == 1
    assert misses == 2


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_deltas_reset_each_call(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    mgr.lookup(key(1), REQ)  # MISS
    mgr.get_stats()  # flush
    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    # No new activity → counter for this policy label should be 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,), 0) == 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_HIT, {}).get((policy,), 0) == 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_MISS, {}).get((policy,), 0) == 0
    assert data.get(CPUOffloadingMetrics.BLOCK_EVICTION, {}).get((policy,), 0) == 0


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_eviction_counter_matches_evicted_key_count(policy: str):
    # Fill 2/2 blocks, then request 2 more -> forces eviction of 2.
    mgr = CPUOffloadingManager(num_blocks=2, cache_policy=policy)
    ps = mgr.prepare_store([key(1), key(2)], REQ)
    assert ps is not None
    mgr.complete_store([key(1), key(2)], REQ, success=True)
    # These 2 new keys need 2 blocks; the manager must evict 2.
    ps2 = mgr.prepare_store([key(3), key(4)], REQ)
    assert ps2 is not None
    assert len(ps2.evicted_keys) == 2

    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    assert data.get(CPUOffloadingMetrics.BLOCK_EVICTION, {}).get((policy,)) == 2
