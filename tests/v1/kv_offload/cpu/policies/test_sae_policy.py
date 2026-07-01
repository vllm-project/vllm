# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def make_block(block_id: int) -> BlockStatus:
    return BlockStatus(block_id)


def test_construction_and_missing_key_returns_none():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.get(key(1)) is None


def test_first_insert_opens_new_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    assert policy._open_sid == 0
    assert policy._sid_to_keys == {0: [key(1)]}
    assert policy._key_to_sid == {key(1): 0}


def test_consecutive_inserts_join_open_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    assert policy._open_sid == 0
    assert policy._sid_to_keys == {0: [key(1), key(2)]}


def test_remove_closes_open_session_and_cleans_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.remove(key(1))
    assert policy._open_sid is None
    assert policy._sid_to_keys == {}
    assert policy._key_to_sid == {}
    assert policy.get(key(1)) is None


def test_insert_after_remove_opens_new_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.remove(key(1))
    policy.insert(key(2), make_block(1))
    assert policy._open_sid == 1  # sid_counter incremented
    assert policy._sid_to_keys == {1: [key(2)]}


def test_insert_seeds_initial_hits_from_ghost_sum():
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=2.0)
    policy._key_ghost[key(1)] = 4.0
    policy._key_ghost[key(2)] = 6.0
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    # (4.0 + 6.0) / 2.0 = 5.0
    assert policy._sid_stats[0]["hits"] == 5.0
