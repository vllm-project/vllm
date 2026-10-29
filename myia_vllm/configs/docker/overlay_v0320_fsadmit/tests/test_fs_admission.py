# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests: myia fs-admission overlay (jsboige/vllm#198, Phase 1a).

Frequency-gated cascades to STORAGE (fs/SSD) secondary tiers:
  - MYIA_FS_ADMIT_MIN_HITS=0 (off)  -> exact upstream write-through;
  - MYIA_FS_ADMIT_MIN_HITS=2 (default) -> a chunk reaches the fs tier only
    once it has been found (lookup HIT) in the primary tier twice: first at
    the threshold crossing (cascade submitted from lookup()), then at any
    later re-store (persistent per-key counter read in complete_store()).

A refused store only costs a recompute later — never wrong output. The
CPU-medium tiers and the primary (RAM) tier are never filtered.

On an unpatched image (tests/run_tests.sh stock) the filter-on tests fail:
upstream write-through cascades the first store, which is exactly the
behaviour the overlay removes.
"""

from collections.abc import Iterable
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering import manager as tiering_manager_module
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="test")
_NUM_PRIMARY = 5


def _keys(ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in ids]


class FakeStorageTierManager(SecondaryTierManager):
    """Test-only STORAGE-medium secondary tier (same shape as the example
    tier): in-memory dict, synchronous completion, records every job."""

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks: dict[OffloadKey, bool] = {}
        self.store_jobs: list[TransferJob] = []
        self.load_jobs: list[TransferJob] = []
        self.completed_jobs: list[JobResult] = []

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.HIT if key in self.chunks else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        self.store_jobs.append(job_metadata)
        for key in job_metadata.keys:
            self.chunks[key] = True
        self.completed_jobs.append(
            JobResult(job_id=job_metadata.job_id, success=True, transfer_time=0.0)
        )

    def submit_load(self, job_metadata: TransferJob) -> None:
        self.load_jobs.append(job_metadata)
        ok = all(key in self.chunks for key in job_metadata.keys)
        self.completed_jobs.append(
            JobResult(job_id=job_metadata.job_id, success=ok, transfer_time=0.0)
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = self.completed_jobs
        self.completed_jobs = []
        return results

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def drain_jobs(self) -> None:
        return


def _make_manager():
    region = MagicMock()
    view = memoryview(torch.zeros((_NUM_PRIMARY, 16), dtype=torch.int8).numpy())
    region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=_NUM_PRIMARY, mmap_region=region
    )
    cpu_tier = ExampleSecondaryTierManager(
        offloading_spec=MagicMock(), primary_kv_view=view, tier_type="example"
    )
    fs_tier = FakeStorageTierManager(
        offloading_spec=MagicMock(), primary_kv_view=view, tier_type="fake_fs"
    )
    manager = TieringOffloadingManager(
        primary_tier=primary, secondary_tiers=[cpu_tier, fs_tier]
    )
    manager.on_new_request(_CTX)
    return manager, primary, cpu_tier, fs_tier


def _end_step(manager):
    manager.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    list(manager.take_events())


def _store(manager, keys, ctx=_CTX):
    assert manager.prepare_store(keys, ctx) is not None
    manager.complete_store(keys, ctx, success=True)


# --- filter off: exact upstream write-through -------------------------------


def test_filter_off_restores_write_through(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "0")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    assert manager._fs_admit_min_hits == 0

    keys = _keys([1, 2])
    _store(manager, keys)

    # First store, zero hits: still cascaded to every tier (rollback mode).
    assert len(fs_tier.store_jobs) == 1
    assert set(fs_tier.store_jobs[0].keys) == set(keys)
    assert len(cpu_tier.chunks) == 2
    assert all(fs_tier.lookup(k, _CTX) is LookupResult.HIT for k in keys)

    # Hit counting is fully disabled.
    for key in keys:
        assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert manager._fs_admit_hits == {}


# --- filter on (default N=2) ------------------------------------------------


def test_first_sighting_is_not_admitted(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    manager, primary, cpu_tier, fs_tier = _make_manager()

    keys = _keys([1, 2])
    _store(manager, keys)

    # The storing walk saw no primary HIT: nothing reaches the fs tier...
    assert fs_tier.store_jobs == []
    assert fs_tier.chunks == {}
    # ...but the RAM store and non-STORAGE cascades are untouched.
    assert all(primary.lookup(k, _CTX) is LookupResult.HIT for k in keys)
    assert len(cpu_tier.chunks) == 2
    # The filter is scoped to STORAGE medium only.
    assert cpu_tier.medium is Medium.CPU
    assert fs_tier.medium is Medium.STORAGE
    assert manager._fs_admit_hits == {}


def test_threshold_crossing_cascades_on_nth_hit(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    key = _keys([1])[0]
    _store(manager, [key])

    # First re-sighting: count 1, below the threshold.
    assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert fs_tier.store_jobs == []

    # Second re-sighting: threshold crossed, cascade submitted from lookup().
    assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert len(fs_tier.store_jobs) == 1
    assert list(fs_tier.store_jobs[0].keys) == [key]
    assert key in manager._fs_admitted_keys

    # Further hits do not re-submit (single crossing).
    assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert len(fs_tier.store_jobs) == 1


def test_restock_after_ram_eviction_is_admitted_by_counter(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    key = _keys([1])[0]
    _store(manager, [key])

    # Block the crossing write the way a back-pressure detector would: the
    # threshold is reached, nothing is submitted, the key is NOT marked
    # admitted — only the persistent counter knows it qualified.
    blocked = MagicMock()
    blocked.should_store.return_value = False
    fs_tier._bp_detector = blocked
    assert manager.lookup(key, _CTX) is LookupResult.HIT  # count 1
    assert manager.lookup(key, _CTX) is LookupResult.HIT  # count 2: crossing blocked
    assert fs_tier.store_jobs == []
    assert key not in manager._fs_admitted_keys
    fs_tier._bp_detector = None

    # Drain cascades so every primary chunk is idle and evictable, then
    # evict `key` from the primary tier by filling it (capacity 5) and prune
    # it from both secondary tiers (the external pruner's job in production
    # for the fs tier; the CPU tier is cleared only to isolate this test).
    _end_step(manager)
    _end_step(manager)
    _store(manager, _keys(range(10, 15)))
    _end_step(manager)
    _end_step(manager)
    fs_tier.chunks.clear()
    cpu_tier.chunks.clear()
    assert manager.lookup(key, _CTX) is LookupResult.MISS

    # Recompute + re-store: the persistent counter (2 >= 2) admits the key
    # even though this store follows no lookup of its own.
    _store(manager, [key])
    assert len(fs_tier.store_jobs) == 1
    assert list(fs_tier.store_jobs[0].keys) == [key]


def test_fs_promotion_then_hits_reach_disk_again(monkeypatch):
    """A chunk that lives only on the fs tier: the promotion itself is not a
    primary-tier sighting, but once the chunk is back in RAM its hits count,
    and the crossing re-submits (the tier's idempotent store dedups)."""
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    key = _keys([42])[0]
    fs_tier.chunks[key] = True  # e.g. survived a restart on disk

    assert manager.lookup(key, _CTX) is LookupResult.HIT_PENDING  # promotion
    _end_step(manager)  # flush + complete the promotion
    _end_step(manager)

    assert manager.lookup(key, _CTX) is LookupResult.HIT  # count 1
    assert fs_tier.store_jobs == []
    assert manager.lookup(key, _CTX) is LookupResult.HIT  # count 2, crossing
    assert len(fs_tier.store_jobs) == 1
    assert list(fs_tier.store_jobs[0].keys) == [key]


# --- GDN latest-state rule: documented stub ---------------------------------


def test_gdn_latest_flag_is_a_noop_stub(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    monkeypatch.setenv("MYIA_FS_ADMIT_GDN_LATEST", "1")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    assert manager._fs_admit_gdn_latest is True

    # The stub changes nothing: same admission behaviour as without it.
    key = _keys([1])[0]
    _store(manager, [key])
    assert fs_tier.store_jobs == []  # first sighting still filtered
    assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert manager.lookup(key, _CTX) is LookupResult.HIT
    assert len(fs_tier.store_jobs) == 1  # crossing still admits
    # The hook is consulted and never excludes (see _gdn_latest_allows).
    assert manager._gdn_latest_allows(key, _CTX) is True


# --- bookkeeping bounds -----------------------------------------------------


def test_tracker_overflow_clears_both_structures(monkeypatch):
    monkeypatch.setenv("MYIA_FS_ADMIT_MIN_HITS", "2")
    manager, primary, cpu_tier, fs_tier = _make_manager()
    key = _keys([1])[0]
    _store(manager, [key])
    manager._fs_admitted_keys.add(_keys([2])[0])

    monkeypatch.setattr(tiering_manager_module, "_FS_ADMIT_TRACKER_MAX", 0)
    manager.lookup(key, _CTX)  # count 1 -> overflow check clears

    assert manager._fs_admit_hits == {}
    assert manager._fs_admitted_keys == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
