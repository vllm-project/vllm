# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy

_CTX = ReqContext(req_id="test")


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def make_block(block_id: int) -> BlockStatus:
    return BlockStatus(block_id)


def make_ready_block(block_id: int) -> BlockStatus:
    b = BlockStatus(block_id)
    b.ref_cnt = 0
    return b


def open_and_insert(
    policy: SAECachePolicy,
    keys_and_blocks: list[tuple[OffloadKey, BlockStatus]],
    *,
    req_id: str = "r0",
    start_pos: int = 0,
) -> None:
    """Test helper: brackets an insert batch with open_session/close_session
    exactly the way `CPUOffloadingManager.prepare_store` does. Use this in
    place of a bare `policy.insert(...)` call — the policy now assumes the
    manager opens a session first."""
    policy.open_session(ReqContext(req_id=req_id), start_pos)
    try:
        for k, b in keys_and_blocks:
            policy.insert(k, b)
    finally:
        policy.close_session()


def test_construction_and_missing_key_returns_none():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.get(key(1)) is None


def test_open_insert_close_creates_a_session():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))])
    assert policy.open_session_id is None  # sealed by close_session
    assert policy.session_keys == {0: [key(1)]}
    assert policy.key_to_session == {key(1): 0}


def test_two_inserts_within_one_batch_join_open_session():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0)), (key(2), make_block(1))])
    assert policy.session_keys == {0: [key(1), key(2)]}
    assert policy.key_to_session == {key(1): 0, key(2): 0}


def test_two_batches_open_two_sessions_when_not_merging():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))], req_id="r0")
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r1")
    # Two distinct sids, one per batch, since no record_lookup preceded
    # either open_session.
    assert policy.session_keys == {0: [key(1)], 1: [key(2)]}


def test_remove_cleans_state():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))])
    policy.remove(key(1))
    assert policy.session_keys == {}
    assert policy.key_to_session == {}
    assert policy.get(key(1)) is None


def test_record_lookup_counts_pending_blocks():
    """`record_lookup` must count blocks that exist but are not yet
    ``is_ready`` (a `BlockStatus` in the ``-1 ref_cnt = HIT_PENDING``
    state). Otherwise the recorded ``hit_count`` falls short of the
    scheduler's ``hit_count`` — which counts HIT_PENDING as a hit —
    and the positional check in ``open_session`` always misses when a
    request's prefix contains any block whose store hasn't completed
    yet. Empirically this drove `pos_drift_mean ≈ 67 blocks` in the
    multi-turn benchmark before the fix."""
    policy = SAECachePolicy(cache_capacity=4)
    # make_block(0) is a BlockStatus in the "not ready" state
    # (ref_cnt = -1) — the state a stored block is in between
    # prepare_store and complete_store.
    open_and_insert(policy, [(key(1), make_block(0))])
    assert policy.blocks[key(1)].is_ready is False
    policy.record_lookup(key(1), ReqContext(req_id="r_pending"))
    assert policy.pending_merge_pointers == {"r_pending": 0}


def test_record_lookup_hit_populates_lookup_state():
    """Only genuine request-lookup hits (via record_lookup, called from
    manager.lookup) may install a merge candidate — plain get() is a
    pure existence check. State is keyed by req_id (sid, hit_count) so
    concurrent requests do not overwrite each other."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r_lookup"))
    assert policy.pending_merge_pointers == {"r_lookup": 0}


def test_get_hit_does_not_populate_lookup_state():
    """get() is a pure existence check — it must NOT install a merge
    candidate. Otherwise the manager's `already stored?` filter, which
    walks every input key with get(), would merge every new store into
    whichever session happened to own a prefix hit."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.get(key(1))  # hit — but on the non-lookup path
    assert policy.pending_merge_pointers == {}


def test_record_lookup_miss_does_not_clear_lookup_state():
    """The req_id's merge candidate must survive a trailing miss on the
    lookup path: the scheduler's prefix lookup calls the manager on each
    key until a miss ends the run, so hits (early keys) then a miss
    (first uncached key) must not lose the merge candidate the earlier
    hit installed. The miss also must NOT bump hit_count."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r_lookup"))  # installs candidate
    policy.record_lookup(
        key(99), ReqContext(req_id="r_lookup")
    )  # miss on unrelated key
    assert policy.pending_merge_pointers == {"r_lookup": 0}


def test_record_lookup_state_is_keyed_per_req_id():
    """Two concurrent requests hitting different sessions must not
    overwrite each other's merge candidates — the single-slot design
    lost this under multi-turn load."""
    policy = SAECachePolicy(cache_capacity=8)
    open_and_insert(policy, [(key(1), make_ready_block(0))], req_id="rA")
    open_and_insert(policy, [(key(2), make_ready_block(1))], req_id="rB")
    policy.record_lookup(key(1), ReqContext(req_id="A"))
    policy.record_lookup(key(2), ReqContext(req_id="B"))
    # Both entries live. The sids come from the order of open_and_insert
    # calls above: sid 0 owns key(1), sid 1 owns key(2). Each got exactly
    # one hit so hit_count == 1 for both.
    assert policy.pending_merge_pointers == {"A": 0, "B": 1}


def test_merge_fires_within_request():
    """A same-req_id lookup hit followed by an open_session merges
    into the recorded session as long as the sid is still resident.
    The port drops the reference's ``_last_lookup_count == start_pos``
    positional check (see test_merge_fires_regardless_of_start_pos_drift)."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r_same"))
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r_same", start_pos=1)
    # Merged into sid 0.
    assert policy.session_keys[0] == [key(1), key(2)]
    assert policy.key_to_session[key(2)] == 0
    assert policy.next_session_id == 1  # no new sid was created
    # open_session consumed the entry.
    assert policy.pending_merge_pointers == {}


def test_merge_does_not_fire_across_req_ids():
    """A lookup on request A must NOT cause request B's store to merge
    into A's session. Per-req_id state makes this structurally
    impossible."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="A"))
    open_and_insert(policy, [(key(2), make_block(1))], req_id="B", start_pos=0)
    # New session opened, not merged.
    assert policy.key_to_session[key(2)] != policy.key_to_session[key(1)]
    assert policy.next_session_id == 2
    # Request A's entry is still live; only B was consumed.
    assert policy.pending_merge_pointers == {"A": 0}


def test_merge_fires_regardless_of_start_pos_drift():
    """The reference algorithm required ``_last_lookup_count == start_pos``.
    We dropped that: the vllm 0.23 scheduler churns the cache between
    a request's lookup and its store (see bench run #6:
    ``pos_drift_mean=58.5``, majority-negative), so the positional
    check made merges effectively impossible. Now the merge fires as
    long as the recorded sid is still resident, whether or not
    ``hit_count == start_pos``.
    """
    policy = SAECachePolicy(cache_capacity=8)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r"))
    # hit_count == 1 but store at start_pos == 5: still merges.
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r", start_pos=5)
    assert policy.key_to_session[key(2)] == policy.key_to_session[key(1)]
    # The entry is consumed on merge; a follow-up store in the same
    # req_id has no live entry left and therefore opens fresh.
    open_and_insert(policy, [(key(3), make_block(2))], req_id="r", start_pos=6)
    assert policy.key_to_session[key(3)] != policy.key_to_session[key(1)]


def test_merge_fires_with_negative_drift():
    """Negative drift (``start_pos < hit_count``) is the dominant
    failure mode under concurrent traffic — the lookup counted N
    blocks but by store time some were evicted, so the store sees
    fewer already-stored blocks. This must still merge into the
    recorded sid, otherwise the algorithm's continuation machinery
    is unreachable in practice."""
    policy = SAECachePolicy(cache_capacity=8)
    open_and_insert(
        policy,
        [(key(1), make_ready_block(0)), (key(2), make_ready_block(1))],
    )
    policy.record_lookup(key(1), ReqContext(req_id="r"))
    policy.record_lookup(key(2), ReqContext(req_id="r"))  # hit_count == 2
    # start_pos=0 (all lookup blocks got evicted before our store) —
    # negative drift, but the sid is still resident, so merge.
    open_and_insert(policy, [(key(3), make_block(2))], req_id="r", start_pos=0)
    assert policy.key_to_session[key(3)] == policy.key_to_session[key(1)]


def test_merge_does_not_fire_when_recorded_sid_is_gone():
    """If the recorded sid has been evicted entirely (removed from
    ``_sid_stats``), there is no session to merge into and
    ``open_session`` must open a fresh one instead."""
    policy = SAECachePolicy(cache_capacity=8)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r"))
    # Simulate the recorded sid being evicted away.
    recorded_sid = policy.pending_merge_pointers["r"]
    policy.session_stats.pop(recorded_sid, None)
    policy.session_keys.pop(recorded_sid, None)
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r", start_pos=1)
    assert policy.key_to_session[key(2)] != recorded_sid


def test_merge_survives_intervening_touches_from_other_requests():
    """Under multi-turn load, other requests' `_touch` calls run between
    my lookup and my prepare_store — bumping the logical timer past the
    reference's `<= 1` window. The per-req_id design removes that
    dependency: as long as my sid is still resident, my merge still
    fires no matter how many timer ticks happened in between."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="me"))
    # Simulate many other-request `touch()` calls bumping the timer.
    for _ in range(20):
        policy.touch([], _CTX)
    open_and_insert(policy, [(key(2), make_block(1))], req_id="me", start_pos=1)
    # Merge still fires.
    assert policy.key_to_session[key(2)] == policy.key_to_session[key(1)]


def test_on_request_finished_drops_stale_merge_pointer():
    """When a request finishes, its per-request state must be released
    so the dict doesn't grow unboundedly across long runs."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r_done"))
    assert "r_done" in policy.pending_merge_pointers
    policy.on_request_finished(ReqContext(req_id="r_done"))
    assert "r_done" not in policy.pending_merge_pointers
    # Idempotent — calling twice or on unknown req_ids is safe.
    policy.on_request_finished(ReqContext(req_id="r_done"))
    policy.on_request_finished(ReqContext(req_id="never_seen"))


def test_merge_skips_ghost_reseed():
    """Merging only bumps last_touch; unlike a fresh session, it must not
    add the merged key's ghost score to hits."""
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=1.0)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r_same"))
    policy.ghost_scores[key(2)] = 50.0  # would be a large reseed if not skipped
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r_same", start_pos=1)
    assert policy.session_stats[0]["hits"] == 0


def test_insert_seeds_initial_hits_from_ghost_sum():
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=2.0)
    policy.ghost_scores[key(1)] = 4.0
    policy.ghost_scores[key(2)] = 6.0
    open_and_insert(policy, [(key(1), make_block(0)), (key(2), make_block(1))])
    # (4.0 + 6.0) / 2.0 = 5.0.
    assert policy.session_stats[0]["hits"] == pytest.approx(5.0)


def test_session_hits_seeded_from_ghost_scores_stay_float():
    """``insert`` seeds ``hits`` from each key's ghost score divided
    by ``ghost_norm`` — a fractional quantity. Both the seed and the
    later per-touch increments live on ``hits`` as float; the score
    math is float either way and no truncation is applied."""
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=3.0)
    policy.ghost_scores[key(1)] = 5.0
    policy.ghost_scores[key(2)] = 5.0
    policy.open_session(ReqContext(req_id="r0"), 0)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    # Accumulated float: (5.0 + 5.0) / 3.0 = 3.333...
    assert policy.session_stats[0]["hits"] == pytest.approx(10.0 / 3.0)
    policy.close_session()
    # close_session no longer mutates hits.
    assert policy.session_stats[0]["hits"] == pytest.approx(10.0 / 3.0)


def test_new_session_records_prefix_depth():
    """Fresh sessions store the batch's prefix depth (num_blocks_in_cache
    at open_session time) so the session score reflects how deep the
    request's already-cached prefix was."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))], req_id="r0", start_pos=7)
    assert policy.session_stats[0]["prefix_depth"] == 7


def test_touch_bumps_hits_once_per_unique_session():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0)), (key(2), make_block(1))])
    policy.touch([key(1), key(2)], _CTX)
    # Reference bumps hits once per unique session in the batch, not once
    # per key — touching two keys from the same session bumps hits by 1.
    assert policy.session_stats[0]["hits"] == 1
    assert policy.session_stats[0]["last_touch"] == 1


def test_touch_bumps_hits_once_per_distinct_session():
    """Two sessions touched in the same batch each get +1."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))], req_id="rA")
    open_and_insert(policy, [(key(2), make_block(1))], req_id="rB")
    policy.touch([key(1), key(2)], _CTX)  # touches sessions 0 and 1
    assert policy.session_stats[0]["hits"] == 1
    assert policy.session_stats[1]["hits"] == 1


def test_touch_ignores_unknown_keys():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))])
    policy.touch([key(1), key(99)], _CTX)
    assert policy.session_stats[0]["hits"] == 1


def test_clear_resets_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.record_lookup(key(1), ReqContext(req_id="r0"))
    open_and_insert(policy, [(key(2), make_block(1))], req_id="r0", start_pos=1)
    policy.ghost_scores[key(2)] = 5.0
    policy.clear()
    assert policy.blocks == {}
    assert policy.session_keys == {}
    assert policy.key_to_session == {}
    assert policy.session_stats == {}
    assert policy.ghost_scores == {}
    assert policy.evictable_blocks == OrderedDict()
    assert policy.open_session_id is None
    assert policy.pending_merge_pointers == {}
    assert policy.open_session_is_merged is False


def test_touch_hit_accumulates_ghost_score():
    """Ghost scoring moved from get() to touch() — get() has no batch
    context, so it can't be trusted with per-position weighting; touch() is
    the only method that receives a real key batch."""
    policy = SAECachePolicy(cache_capacity=4, ghost_hit_weight=3.0)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.touch([key(1)], _CTX)
    policy.touch([key(1)], _CTX)
    # Single-key batches -> pos_weight(0) == 1.0 each time.
    assert policy.ghost_scores[key(1)] == 6.0


def test_touch_miss_accumulates_ghost_score():
    policy = SAECachePolicy(cache_capacity=4, ghost_miss_weight=0.5)
    policy.touch([key(1)], _CTX)
    policy.touch([key(1)], _CTX)
    assert policy.ghost_scores[key(1)] == 1.0


def test_touch_ghost_bonus_decreases_with_position():
    """_pos_weight(i) weights earlier keys in a touch() batch more heavily,
    matching the reference's per-batch position weighting (now recoverable
    since touch(), unlike get(), receives the whole batch at once)."""
    policy = SAECachePolicy(cache_capacity=4, ghost_hit_weight=10.0)
    open_and_insert(
        policy,
        [(key(1), make_ready_block(0)), (key(2), make_ready_block(1))],
    )
    policy.touch([key(1), key(2)], _CTX)
    assert policy.ghost_scores[key(1)] == 10.0  # pos_weight(0) == 1.0
    assert policy.ghost_scores[key(2)] == 10.0 * policy._position_weight(1)
    assert policy.ghost_scores[key(1)] > policy.ghost_scores[key(2)]


def test_touch_reclassifies_hit_and_miss_per_key():
    """touch()'s key batch (the group's known offload_keys) isn't
    necessarily the exact set get() classified as hits this pass, so touch()
    must reclassify hit/miss per key from residency rather than trusting an
    externally supplied hit count."""
    policy = SAECachePolicy(
        cache_capacity=4, ghost_hit_weight=10.0, ghost_miss_weight=1.0
    )
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    # key(2) was never inserted -> not resident -> miss, even though it's
    # touched in the same batch as a hit.
    policy.touch([key(1), key(2)], _CTX)
    assert policy.ghost_scores[key(1)] == 10.0  # hit, pos_weight(0) == 1.0
    assert policy.ghost_scores[key(2)] == 1.0 * policy._position_weight(1)  # miss


def test_decay_runs_every_interval():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=3,
        decay_factor=0.5,
        ghost_hit_weight=1.0,
        ghost_miss_weight=1.0,
    )
    open_and_insert(policy, [(key(1), make_block(0))])
    policy.session_stats[0]["hits"] = 10
    policy.ghost_scores[key(1)] = 4.0  # resident
    # 3 touches trigger one decay tick. Each touch also bumps sid 0's hits
    # by 1 (key(1) belongs to sid 0): 10 -> 11 -> 12 -> 13, then decayed.
    policy.touch([key(1)], _CTX)
    policy.touch([key(1)], _CTX)
    policy.touch([key(1)], _CTX)
    assert policy.session_stats[0]["hits"] == pytest.approx(13 * 0.5)
    # resident key(1) accumulated 3 hits before decay: (4.0 + 3.0) * 0.5 = 3.5
    assert policy.ghost_scores[key(1)] == 3.5


def test_decay_prunes_below_threshold():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=1,
        decay_factor=0.1,
    )
    policy.ghost_scores[key(99)] = 0.05  # non-resident
    policy.touch([key(1)], _CTX)  # triggers decay; 0.05 * 0.1 = 0.005 < 0.01
    assert key(99) not in policy.ghost_scores


def test_mark_evictable_adds_to_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))])
    policy.mark_evictable(key(1))
    assert key(1) in policy.evictable_blocks


def test_mark_non_evictable_removes_from_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_block(0))])
    policy.mark_evictable(key(1))
    policy.mark_non_evictable(key(1))
    assert key(1) not in policy.evictable_blocks


def test_mark_non_evictable_missing_key_is_safe():
    policy = SAECachePolicy(cache_capacity=4)
    # Should not raise
    policy.mark_non_evictable(key(99))


def test_evict_returns_empty_when_n_zero():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.evict(0, set(), ReqContext(req_id="r0"), 0) == []


def test_evict_returns_none_when_insufficient_idle_blocks():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Only 1 idle block, but need 2 -> insufficient
    # Zero hits, so admission gate allows; the failure must be
    # `insufficient`, not `gate_refused`.
    result = policy.evict(2, set(), ReqContext(req_id="r0"), 0)
    assert result is None


def test_evict_walks_sessions_worst_first():
    policy = SAECachePolicy(cache_capacity=4)
    # Session 0 (worst): low hits
    open_and_insert(policy, [(key(1), make_ready_block(0))], req_id="rA")
    policy.mark_evictable(key(1))
    # Session 1 (best): high hits
    open_and_insert(policy, [(key(2), make_ready_block(1))], req_id="rB")
    policy.mark_evictable(key(2))
    policy.session_stats[1]["hits"] = 1000
    # Evict 1 -> should come from session 0
    evicted = policy.evict(1, set(), ReqContext(req_id="rC"), 0)
    assert evicted is not None
    assert len(evicted) == 1
    assert evicted[0][0] == key(1)


def test_evict_skips_protected_keys():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    assert policy.evict(1, {key(1)}, ReqContext(req_id="rZ"), 0) is None


def test_evict_skips_non_evictable_keys():
    policy = SAECachePolicy(cache_capacity=4)
    b = BlockStatus(0)
    b.ref_cnt = 1  # not evictable (ref_cnt != 0)
    open_and_insert(policy, [(key(1), b)])
    # Never marked evictable
    assert policy.evict(1, set(), ReqContext(req_id="rZ"), 0) is None


def test_evict_removes_evicted_keys_from_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    evicted = policy.evict(1, set(), ReqContext(req_id="rZ"), 0)
    assert evicted is not None
    assert key(1) not in policy.blocks
    assert key(1) not in policy.key_to_session
    assert key(1) not in policy.evictable_blocks
    # sid 0 was emptied, so it should be gone
    assert 0 not in policy.session_keys


def test_admission_gate_denies_when_worst_incumbent_score_exceeds_baseline():
    """Reference gate: new_score = logical_timer + 30000/(1 + start_pos/8).
    When the worst incumbent's score is above that baseline, the gate
    denies eviction."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Push incumbent's score well above the new-session baseline via hits
    # (freq_bonus = hits * 1500.0). 30 hits -> 45000 freq_bonus + 30000 pos
    # -> 75000, easily above the ~30000 baseline for the would-be session
    # at start_pos=0.
    policy.session_stats[0]["hits"] = 30
    result = policy.evict(1, set(), ReqContext(req_id="new"), 0)
    assert result is None


def test_admission_gate_allows_when_baseline_beats_worst_incumbent():
    """When incumbent's score is below new_score = logical_timer + 30000,
    the gate admits and eviction proceeds."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Incumbent has 0 hits: score = 0 (last_touch) + 0 (freq) + 30000 (pos)
    # = 30000. New session baseline = logical_timer + 30000/(1+0/8) = 30000.
    # new_score >= worst_score -> gate allows.
    policy.session_stats[0]["hits"] = 0
    result = policy.evict(1, set(), ReqContext(req_id="new"), 0)
    assert result is not None
    assert len(result) == 1


def test_admission_gate_ignores_ghost_scores():
    """Reference explicitly excludes ghost scores from the gate. A high
    ghost score on `protected` keys must NOT sway the gate decision."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Incumbent's score is high enough that the gate would deny WITHOUT
    # ghost scores helping the newcomer.
    policy.session_stats[0]["hits"] = 30  # score ~= 75000
    policy.ghost_scores[key(2)] = 10_000_000.0
    result = policy.evict(1, {key(2)}, ReqContext(req_id="new"), 0)
    assert result is None


def test_admission_gate_baseline_uses_real_start_pos():
    """When start_pos > 0, the baseline shrinks (30000/(1 + start_pos/8)),
    so a session that would have squeaked past the gate at start_pos=0
    can be refused at a larger start_pos, matching the reference's
    prefix-depth-aware asymmetry."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Make incumbent's score exactly the baseline at start_pos=0.
    policy.session_stats[0]["hits"] = 0
    # At start_pos=0: baseline = 30000, worst = 30000 -> admits.
    assert policy.evict(1, set(), ReqContext(req_id="r0"), 0) is not None
    # Re-prime the same setup for start_pos comparison.
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    policy.session_stats[policy.key_to_session[key(1)]]["hits"] = 0
    # At large start_pos, baseline shrinks below 30000; worst still ~30000.
    # But we also need worst's start_pos to be 0 for this test to expose the
    # asymmetry — that's the setup we have. Baseline at start_pos=1000 is
    # ~30000/126 ~= 238, below worst's 30000, so gate should refuse.
    result = policy.evict(1, set(), ReqContext(req_id="r0"), 1000)
    assert result is None


def test_evict_skips_admission_gate_when_merging():
    """A live merge candidate (same-req_id lookup hit whose sid is
    still resident) bypasses the gate entirely, matching the
    reference's `if not is_merging and needed > 0`. The positional
    check was dropped, so any ``start_pos`` triggers the merge bypass
    as long as the sid still exists."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    # Score high enough that the gate would normally deny.
    policy.session_stats[0]["hits"] = 30
    policy.record_lookup(key(1), ReqContext(req_id="r_merge"))
    # start_pos need not match hit_count anymore.
    result = policy.evict(1, set(), ReqContext(req_id="r_merge"), 9)
    assert result is not None


def test_evict_gate_does_not_skip_for_wrong_req_id():
    """A recorded lookup on request A must NOT let request B bypass the
    gate — that would recreate the cross-request leak."""
    policy = SAECachePolicy(cache_capacity=4)
    open_and_insert(policy, [(key(1), make_ready_block(0))])
    policy.mark_evictable(key(1))
    policy.session_stats[0]["hits"] = 30  # gate would refuse
    policy.record_lookup(key(1), ReqContext(req_id="A"))
    result = policy.evict(1, set(), ReqContext(req_id="B"), 0)
    assert result is None


def test_manager_prepare_store_does_not_leak_hits_across_requests():
    """The manager's prepare_store calls _policy.get on every input key
    for its `already stored?` filter. Those hits must NOT set the
    lookup pointer — only manager.lookup (via record_lookup) should.
    Additionally, a lookup on request A must not cause request B's
    prepare_store to merge into A's session."""
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    mgr = CPUOffloadingManager(num_blocks=8, cache_policy="sae")
    ctx_a = ReqContext(req_id="A")
    ctx_b = ReqContext(req_id="B")
    # Prime the cache with a first store from request A.
    out = mgr.prepare_store([key(1), key(2)], ctx_a)
    assert out is not None and out.keys_to_store == [key(1), key(2)]
    mgr.complete_store([key(1), key(2)], ctx_a, success=True)
    mgr.touch([key(1), key(2)], ctx_a)
    policy = mgr._policy
    assert isinstance(policy, SAECachePolicy)
    # A second prepare_store on request B whose input includes an already-
    # stored key (key(1)) plus a new key (key(3)). The internal
    # get(key(1)) hit must not leak into request B's session decision.
    out = mgr.prepare_store([key(1), key(3)], ctx_b)
    assert out is not None
    # key(3) opened a fresh session, not merged into request A's session.
    assert policy.key_to_session[key(3)] != policy.key_to_session[key(1)]


def test_manager_prepare_store_merges_within_same_request_after_lookup():
    """Positive counterpart to the cross-request test: within the same
    request, a lookup-hit followed by a prepare_store on more keys
    should merge into the hit's session (that's the whole point of the
    merge machinery)."""
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    mgr = CPUOffloadingManager(num_blocks=8, cache_policy="sae")
    ctx = ReqContext(req_id="X")
    # First: store some blocks so there's a session to merge into.
    out = mgr.prepare_store([key(1), key(2)], ctx)
    assert out is not None
    mgr.complete_store([key(1), key(2)], ctx, success=True)
    # Simulate the scheduler's lookup pass on request X — this records
    # the merge pointer.
    mgr.lookup(key(1), ctx)
    # Now prepare_store for more keys in the same request. key(1) is
    # already-stored (part of the prefix), key(3) is new. The lookup
    # made request X eligible to merge.
    out = mgr.prepare_store([key(1), key(3)], ctx)
    assert out is not None
    policy = mgr._policy
    assert isinstance(policy, SAECachePolicy)
    # key(3) merged into key(1)'s session.
    assert policy.key_to_session[key(3)] == policy.key_to_session[key(1)]


def test_cpu_offloading_manager_accepts_sae_policy_and_kwargs():
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    mgr = CPUOffloadingManager(
        num_blocks=4,
        cache_policy="sae",
        policy_kwargs={"decay_interval": 42, "decay_factor": 0.5},
    )
    assert isinstance(mgr._policy, SAECachePolicy)
    assert mgr._policy.decay_interval == 42
    assert mgr._policy.decay_factor == 0.5


def test_cpu_offloading_manager_defaults_still_work_for_lru():
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

    mgr = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    assert mgr._policy is not None
    assert isinstance(mgr._policy, LRUCachePolicy)
