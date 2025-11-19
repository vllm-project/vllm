# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm.v1.core.eviction_policies import FrequencyCostEvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def test_frequency_cost_eviction_orders_by_score():
    policy = FrequencyCostEvictionPolicy(block_size=16, alpha=2.0)

    blocks = []
    now = time.monotonic()
    # Create three cached-free blocks with different access patterns
    for i, (age, access) in enumerate([(10.0, 1), (5.0, 1), (5.0, 10)]):
        b = KVCacheBlock(block_id=i)
        # mark as free and cached by simulating a non-None hash
        b._block_hash = b"dummy_hash"  # type: ignore[attr-defined]
        b.ref_cnt = 0
        # manually set tracking attributes used by the policy
        b.first_access_ts = now - age  # type: ignore[attr-defined]
        b.access_count = access  # type: ignore[attr-defined]
        blocks.append(b)
        policy.on_block_release(b)

    evicted = policy.get_eviction_candidates(3)
    # The block with lowest frequency/age should be first (age=10, access=1)
    assert evicted[0] == 0
    # The most frequently accessed among recent ones should be retained longer
    assert set(evicted) == {0, 1, 2}


def test_policy_remove_block():
    policy = FrequencyCostEvictionPolicy(block_size=16)
    b = KVCacheBlock(block_id=42)
    b._block_hash = b"dummy"  # type: ignore[attr-defined]
    b.ref_cnt = 0
    b.first_access_ts = time.monotonic() - 1.0  # type: ignore[attr-defined]
    b.access_count = 5  # type: ignore[attr-defined]
    policy.on_block_release(b)

    # Removing the block should make it unselectable
    policy.remove_block(b)
    selected = policy.get_eviction_candidates(1)
    assert 42 not in selected
