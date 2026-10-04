# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprioritized chunks are taken last, within a bounded search."""

import pytest

from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import (
    DEPRIORITIZED_SCAN_BUDGET,
    ChunkStatus,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

POLICIES = [LRUCachePolicy, ARCCachePolicy]


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def fill(policy, count: int) -> list[OffloadKey]:
    keys = []
    for i in range(count):
        k = key(i)
        chunk = ChunkStatus(chunk_id=i)
        chunk.ref_cnt = 0  # stored and idle, i.e. evictable
        policy.insert(k, chunk)
        policy.mark_evictable(k)
        keys.append(k)
    return keys


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_a_plain_chunk_dies_before_a_deprioritized_one(policy_cls):
    policy = policy_cls(cache_capacity=8)
    keys = fill(policy, 4)
    deprioritized = {keys[0], keys[1]}

    evicted = policy.evict(2, set(), deprioritized)

    assert evicted is not None
    assert {k for k, _ in evicted} == {keys[2], keys[3]}


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_deprioritized_chunks_are_taken_when_nothing_else_is_left(policy_cls):
    """The preference must not turn into a refusal to evict."""
    policy = policy_cls(cache_capacity=8)
    keys = fill(policy, 3)

    evicted = policy.evict(2, set(), set(keys))

    assert evicted is not None
    assert len(evicted) == 2


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_the_search_is_bounded_on_a_fully_deprioritized_cache(policy_cls):
    """A cache made entirely of deprioritized chunks must still evict fast.

    The budget is what keeps an all-in-flight tier from turning every store
    into a walk over the whole cache.
    """
    policy = policy_cls(cache_capacity=4096)
    keys = fill(policy, DEPRIORITIZED_SCAN_BUDGET * 4)

    evicted = policy.evict(1, set(), set(keys))

    assert evicted is not None
    assert len(evicted) == 1


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_a_failed_eviction_changes_nothing(policy_cls):
    """Asking for more than exists must leave every chunk where it was."""
    policy = policy_cls(cache_capacity=8)
    keys = fill(policy, 2)

    assert policy.evict(5, set(), {keys[0]}) is None

    # Everything is still there and still evictable.
    for k in keys:
        assert policy.get(k) is not None
    evicted = policy.evict(2, set(), set())
    assert evicted is not None and len(evicted) == 2
