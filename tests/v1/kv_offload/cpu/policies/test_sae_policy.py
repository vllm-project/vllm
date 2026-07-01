# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict

from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def make_block(block_id: int) -> BlockStatus:
    return BlockStatus(block_id)


def make_ready_block(block_id: int) -> BlockStatus:
    b = BlockStatus(block_id)
    b.ref_cnt = 0
    return b


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


def test_touch_bumps_hits_and_last_touch_and_closes_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    policy.touch([key(1), key(2)])
    sid = 0
    assert policy._sid_stats[sid]["hits"] == 2.0
    assert policy._sid_stats[sid]["last_touch"] == 1
    assert policy._open_sid is None
    # Next insert opens a fresh session
    policy.insert(key(3), make_block(2))
    assert policy._open_sid == 1


def test_touch_ignores_unknown_keys():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.touch([key(1), key(99)])
    assert policy._sid_stats[0]["hits"] == 1.0


def test_clear_resets_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.touch([key(1)])
    policy._key_ghost[key(2)] = 5.0
    policy.clear()
    assert policy._blocks == {}
    assert policy._sid_to_keys == {}
    assert policy._key_to_sid == {}
    assert policy._sid_stats == {}
    assert policy._key_ghost == {}
    assert policy._evictable_keys == OrderedDict()
    assert policy._open_sid is None
    assert policy._last_event == "clear"


def test_get_hit_accumulates_ghost_score():
    policy = SAECachePolicy(cache_capacity=4, ghost_hit_weight=3.0)
    policy.insert(key(1), make_ready_block(0))
    policy.get(key(1))
    policy.get(key(1))
    assert policy._key_ghost[key(1)] == 6.0


def test_get_miss_accumulates_ghost_score():
    policy = SAECachePolicy(cache_capacity=4, ghost_miss_weight=0.5)
    policy.get(key(1))
    policy.get(key(1))
    assert policy._key_ghost[key(1)] == 1.0


def test_decay_runs_every_interval_and_prunes_low_ghosts():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=3,
        decay_factor=0.5,
        ghost_hit_weight=1.0,
        ghost_miss_weight=1.0,
    )
    policy.insert(key(1), make_block(0))
    policy._sid_stats[0]["hits"] = 10.0
    policy._key_ghost[key(2)] = 0.05  # non-resident
    policy._key_ghost[key(1)] = 4.0  # resident
    # 3 gets triggers one decay tick
    policy.get(key(1))
    policy.get(key(1))
    policy.get(key(1))
    assert policy._sid_stats[0]["hits"] == 5.0
    # resident key(1) accumulated 3 hits before decay: (4.0 + 3.0) * 0.5 = 3.5
    assert policy._key_ghost[key(1)] == 3.5
    # non-resident key(2) with score 0.05 -> 0.025 < 0.01? no, still 0.025 >= 0.01
    # Adjust: set higher threshold test
    assert key(2) in policy._key_ghost


def test_decay_prunes_below_threshold():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=1,
        decay_factor=0.1,
    )
    policy._key_ghost[key(99)] = 0.05  # non-resident
    policy.get(key(1))  # triggers decay; 0.05 * 0.1 = 0.005 < 0.01
    assert key(99) not in policy._key_ghost
