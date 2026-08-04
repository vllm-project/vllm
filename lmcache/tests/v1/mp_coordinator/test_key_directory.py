# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator key directory (I1): event application
semantics (seq dedup, gap detection, incarnation fencing), lookup, and
instance cleanup."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.key_directory import ApplyResult, KeyDirectory


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: CacheEventType = CacheEventType.STORE,
    tier: Tier = Tier.L1,
    backend: str = "dram",
    keys: list[ObjectKey] | None = None,
    size_bytes: int = 1024,
    ts: float = 0.0,
) -> CacheEventBatch:
    entries = [
        CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=size_bytes)
        for k in (keys or [_key(0xAA)])
    ]
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        entries=entries,
        ts=ts,
    )


# -- Store / lookup ----------------------------------------------------------


def test_store_then_lookup():
    directory = KeyDirectory()
    assert directory.apply_batch(_batch(keys=[_key(1)])) == ApplyResult.APPLIED

    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    p = placements[0]
    assert p.instance_id == "node-a"
    assert p.incarnation == 1
    assert p.tier == Tier.L1
    assert p.backend == "dram"
    assert p.size_bytes == 1024


def test_lookup_unknown_key_is_empty():
    directory = KeyDirectory()
    assert directory.lookup([_key(9)]) == [[]]


def test_lookup_preserves_request_order():
    directory = KeyDirectory()
    directory.apply_batch(_batch(keys=[_key(2)]))
    results = directory.lookup([_key(1), _key(2)])
    assert results[0] == []
    assert len(results[1]) == 1


def test_restore_updates_size_without_duplicating_placement():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    directory.apply_batch(_batch(seq=2, keys=[_key(1)], size_bytes=200))

    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1
    assert placements[0].size_bytes == 200


def test_same_key_on_two_instances():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-a", "node-b"]


def test_same_key_on_two_tiers_of_one_instance():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, tier=Tier.L1, backend="dram", keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [
        (Tier.L1, "dram"),
        (Tier.L2, "fs"),
    ]


# -- Delete ------------------------------------------------------------------


def test_delete_drops_placement_and_empty_record():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )

    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]
    stats = directory.stats()
    assert stats.num_keys == 0
    assert stats.num_placements == 0
    assert stats.instances["node-a"].num_keys == 0


def test_removal_of_one_tier_keeps_the_other():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, tier=Tier.L1, keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    directory.apply_batch(
        _batch(seq=3, event_type=CacheEventType.DELETE, tier=Tier.L1, keys=[_key(1)])
    )

    [placements] = directory.lookup([_key(1)])
    assert [(p.tier, p.backend) for p in placements] == [(Tier.L2, "fs")]
    # The instance still holds the key on L2, so its key count is intact.
    assert directory.stats().instances["node-a"].num_keys == 1


def test_removal_of_unknown_key_is_noop():
    directory = KeyDirectory()
    outcome = directory.apply_batch(
        _batch(event_type=CacheEventType.DELETE, keys=[_key(7)])
    )
    assert outcome == ApplyResult.APPLIED
    assert directory.stats().num_keys == 0


# -- Access ------------------------------------------------------------------


def test_access_does_not_create_records():
    directory = KeyDirectory()
    outcome = directory.apply_batch(
        _batch(event_type=CacheEventType.ACCESS, keys=[_key(1)])
    )
    assert outcome == ApplyResult.APPLIED
    assert directory.stats().num_keys == 0


def test_access_batch_allows_empty_backend():
    """ACCESS carries no placement identity, so ``backend`` may be empty;
    applying it refreshes recency without touching placements."""
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    outcome = directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.ACCESS, keys=[_key(1)], backend="")
    )
    assert outcome == ApplyResult.APPLIED
    [placements] = directory.lookup([_key(1)])
    assert len(placements) == 1  # placement identity untouched


def test_placement_bearing_batches_require_backend():
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.STORE, keys=[_key(1)], backend="")
    with pytest.raises(ValueError):
        _batch(event_type=CacheEventType.DELETE, keys=[_key(1)], backend="")


# -- Seq handling ------------------------------------------------------------


def test_duplicate_seq_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=100))
    outcome = directory.apply_batch(_batch(seq=1, keys=[_key(1)], size_bytes=999))

    assert outcome == ApplyResult.DUPLICATE
    [placements] = directory.lookup([_key(1)])
    assert placements[0].size_bytes == 100


def test_replayed_older_seq_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    directory.apply_batch(
        _batch(seq=2, event_type=CacheEventType.DELETE, keys=[_key(1)])
    )
    outcome = directory.apply_batch(_batch(seq=1, keys=[_key(1)]))

    assert outcome == ApplyResult.DUPLICATE
    assert directory.lookup([_key(1)]) == [[]]


def test_seq_gap_sets_resync_flag_but_applies():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(_batch(seq=5, keys=[_key(2)]))

    assert outcome == ApplyResult.APPLIED
    info = directory.stats().instances["node-a"]
    assert info.gap_detected is True
    assert info.last_seq == 5
    assert len(directory.lookup([_key(2)])[0]) == 1


def test_contiguous_seqs_do_not_flag_gap():
    directory = KeyDirectory()
    directory.apply_batch(_batch(seq=1, keys=[_key(1)]))
    directory.apply_batch(_batch(seq=2, keys=[_key(2)]))
    assert directory.stats().instances["node-a"].gap_detected is False


# -- Incarnation fencing -----------------------------------------------------


def test_new_incarnation_fences_old_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(incarnation=1, seq=1, keys=[_key(1), _key(2)]))
    outcome = directory.apply_batch(_batch(incarnation=2, seq=1, keys=[_key(3)]))

    assert outcome == ApplyResult.APPLIED
    assert directory.lookup([_key(1)]) == [[]]
    assert directory.lookup([_key(2)]) == [[]]
    [placements] = directory.lookup([_key(3)])
    assert placements[0].incarnation == 2
    info = directory.stats().instances["node-a"]
    assert info.incarnation == 2
    assert info.last_seq == 1


def test_fence_spares_other_instances_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))
    directory.apply_batch(_batch(instance_id="node-a", incarnation=2, keys=[_key(9)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]


def test_stale_incarnation_batch_is_dropped():
    directory = KeyDirectory()
    directory.apply_batch(_batch(incarnation=2, seq=1, keys=[_key(1)]))
    outcome = directory.apply_batch(_batch(incarnation=1, seq=99, keys=[_key(2)]))

    assert outcome == ApplyResult.STALE_INCARNATION
    assert directory.lookup([_key(2)]) == [[]]


# -- drop_instance -----------------------------------------------------------


def test_drop_instance_removes_all_its_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", keys=[_key(1), _key(2)]))
    directory.apply_batch(_batch(instance_id="node-b", keys=[_key(1)]))

    removed = directory.drop_instance("node-a")

    assert removed == 2
    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]
    assert directory.lookup([_key(2)]) == [[]]
    assert "node-a" not in directory.stats().instances


def test_drop_unknown_instance_returns_zero():
    directory = KeyDirectory()
    assert directory.drop_instance("ghost") == 0


# -- Intrinsic invariants ------------------------------------------------------


def test_tier_all_is_unconstructible():
    with pytest.raises(ValueError, match="concrete tier"):
        _batch(tier=Tier.ALL)


def test_seq_below_one_is_unconstructible():
    with pytest.raises(ValueError, match="seq"):
        _batch(seq=0)


def test_negative_size_is_unconstructible():
    with pytest.raises(ValueError, match="size_bytes"):
        CacheEventEntry(key=_key(1).to_encoded_object_key(), size_bytes=-1)


# -- Stats -------------------------------------------------------------------


def test_stats_counts_keys_and_placements():
    directory = KeyDirectory()
    directory.apply_batch(_batch(instance_id="node-a", seq=1, keys=[_key(1), _key(2)]))
    directory.apply_batch(
        _batch(instance_id="node-b", seq=1, tier=Tier.L2, backend="fs", keys=[_key(1)])
    )

    stats = directory.stats()
    assert stats.num_keys == 2
    assert stats.num_placements == 3
    assert stats.instances["node-a"].num_keys == 2
    assert stats.instances["node-b"].num_keys == 1
