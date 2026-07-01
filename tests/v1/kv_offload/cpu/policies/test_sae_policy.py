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


def test_session_hits_truncated_to_int_on_close():
    """Reference computes initial_hits = int(ghost_sum / ghost_norm). The port
    accumulates the ghost seed as a float while the session is open, then
    truncates to int when the session closes (touch/evict/remove/clear)."""
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=3.0)
    policy._key_ghost[key(1)] = 5.0
    policy._key_ghost[key(2)] = 5.0
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    # While open, accumulated float: (5.0 + 5.0) / 3.0 = 3.333...
    assert policy._sid_stats[0]["hits"] > 3.0
    policy.touch([])  # closes the session -> seal
    # int(3.333...) = 3
    assert policy._sid_stats[0]["hits"] == 3
    assert isinstance(policy._sid_stats[0]["hits"], int)


def test_touch_bumps_hits_once_per_unique_session_and_closes_it():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    policy.touch([key(1), key(2)])
    sid = 0
    # Reference bumps hits once per unique session in the batch, not once
    # per key — touching two keys from the same session bumps hits by 1.
    assert policy._sid_stats[sid]["hits"] == 1
    assert policy._sid_stats[sid]["last_touch"] == 1
    assert policy._open_sid is None
    # Next insert opens a fresh session
    policy.insert(key(3), make_block(2))
    assert policy._open_sid == 1


def test_touch_bumps_hits_once_per_distinct_session():
    """Two sessions touched in the same batch each get +1."""
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.touch([])  # close session 0
    policy.insert(key(2), make_block(1))
    policy.touch([key(1), key(2)])  # touches sessions 0 and 1
    assert policy._sid_stats[0]["hits"] == 1
    assert policy._sid_stats[1]["hits"] == 1


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


def test_decay_truncates_session_hits_to_int():
    """Reference does int(hits * decay_factor); the port must too."""
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=1,
        decay_factor=0.9,
    )
    policy.insert(key(1), make_block(0))
    policy._sid_stats[0]["hits"] = 7
    policy.get(key(1))  # triggers decay: int(7 * 0.9) = int(6.3) = 6
    assert policy._sid_stats[0]["hits"] == 6
    assert isinstance(policy._sid_stats[0]["hits"], int)


def test_mark_evictable_adds_to_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.mark_evictable(key(1))
    assert key(1) in policy._evictable_keys


def test_mark_non_evictable_removes_from_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.mark_evictable(key(1))
    policy.mark_non_evictable(key(1))
    assert key(1) not in policy._evictable_keys


def test_mark_non_evictable_missing_key_is_safe():
    policy = SAECachePolicy(cache_capacity=4)
    # Should not raise
    policy.mark_non_evictable(key(99))


def score_of(policy: SAECachePolicy, sid: int) -> float:
    stats = policy._sid_stats[sid]
    pos_bonus = 30000.0 / (1.0 + stats["start_pos"] / 8.0)
    freq_bonus = stats["hits"] * 1500.0
    return stats["last_touch"] + freq_bonus + pos_bonus


def test_evict_returns_empty_when_n_zero():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.evict(0, set()) == []


def test_evict_returns_none_when_insufficient_idle_blocks():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    # Only 1 idle block, but need 2
    assert policy.evict(2, set()) is None


def test_evict_walks_sessions_worst_first():
    policy = SAECachePolicy(cache_capacity=4)
    # Session 0 (worst): low hits
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    policy.touch([])  # close session 0 without bumping hits
    # Session 1 (best): high hits
    policy.insert(key(2), make_ready_block(1))
    policy.mark_evictable(key(2))
    policy._sid_stats[1]["hits"] = 1000.0
    # Evict 1 -> should come from session 0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert len(evicted) == 1
    assert evicted[0][0] == key(1)


def test_evict_skips_protected_keys():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    assert policy.evict(1, {key(1)}) is None


def test_evict_skips_non_evictable_keys():
    policy = SAECachePolicy(cache_capacity=4)
    b = BlockStatus(0)
    b.ref_cnt = 1  # not evictable (ref_cnt != 0)
    policy.insert(key(1), b)
    # Never marked evictable
    assert policy.evict(1, set()) is None


def test_evict_removes_evicted_keys_from_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert key(1) not in policy._blocks
    assert key(1) not in policy._key_to_sid
    assert key(1) not in policy._evictable_keys
    # sid 0 was emptied, so it should be gone
    assert 0 not in policy._sid_to_keys


def test_evict_admission_gate_denies_when_worst_incumbent_score_exceeds_baseline():
    """Reference gate: new_score = logical_timer + 30000.0. When the worst
    incumbent's score is above that baseline, the gate denies eviction."""
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    # Push incumbent's score well above the new-session baseline via hits
    # (freq_bonus = hits * 1500.0). 30 hits -> 45000 freq_bonus + 30000 pos
    # -> 75000, easily above the ~30000 baseline for the would-be session.
    policy._sid_stats[0]["hits"] = 30
    result = policy.evict(1, set())
    assert result is None


def test_evict_admission_gate_allows_when_baseline_beats_worst_incumbent():
    """When incumbent's score is below new_score = logical_timer + 30000,
    the gate admits and eviction proceeds."""
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    # Incumbent has 0 hits: score = 0 (last_touch) + 0 (freq) + 30000 (pos)
    # = 30000. New session baseline = logical_timer + 30000 = 0 + 30000 =
    # 30000. new_score >= worst_score -> gate allows.
    policy._sid_stats[0]["hits"] = 0
    result = policy.evict(1, set())
    assert result is not None
    assert len(result) == 1


def test_evict_admission_gate_ignores_ghost_scores():
    """Reference explicitly excludes ghost scores from the gate. A high
    ghost score on `protected` keys must NOT sway the gate decision."""
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_ready_block(0))
    policy.mark_evictable(key(1))
    # Incumbent's score is high enough that the gate would deny WITHOUT
    # ghost scores helping the newcomer.
    policy._sid_stats[0]["hits"] = 30  # score ~= 75000
    # Load a massive ghost score on the would-be new session. If ghost
    # were factored into the gate, this would allow eviction; the fix
    # says it must still deny.
    policy._key_ghost[key(2)] = 10_000_000.0
    result = policy.evict(1, {key(2)})
    assert result is None


def test_cpu_offloading_manager_accepts_sae_policy_and_kwargs():
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    mgr = CPUOffloadingManager(
        num_blocks=4,
        cache_policy="sae",
        policy_kwargs={"decay_interval": 42, "decay_factor": 0.5},
    )
    assert isinstance(mgr._policy, SAECachePolicy)
    assert mgr._policy._decay_interval == 42
    assert mgr._policy._decay_factor == 0.5


def test_cpu_offloading_manager_defaults_still_work_for_lru():
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

    mgr = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    assert mgr._policy is not None
    assert isinstance(mgr._policy, LRUCachePolicy)
