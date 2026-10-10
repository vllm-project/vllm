# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.config.ec_transfer import ECRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler

_N = 16
_BS = 64


class _Pos:
    def __init__(self, offset, length):
        self.offset = offset
        self.length = length


class _Feature:
    def __init__(self, mm_hash, length=1, identifier=None):
        self.mm_hash = mm_hash
        self.identifier = identifier if identifier is not None else mm_hash
        self.mm_position = _Pos(0, length)


class _Request:
    _counter = 0

    def __init__(self, features, request_id=None):
        self.mm_features = features
        if request_id is None:
            _Request._counter += 1
            self.request_id = f"req_{_Request._counter}"
        else:
            self.request_id = request_id


class _WorkerOutput:
    """Stand-in for ECConnectorOutput carrying a completion report.

    `saves` are mm_hashes; `loads` are transfer ids. One entry per reporting
    rank, so a list holding the same id twice is two ranks reporting it.
    """

    def __init__(self, *, saves=None, loads=None):
        self.ec_connector_worker_meta = ECCPUWorkerMetadata(
            completed_saves=list(saves or []),
            completed_loads=list(loads or []),
        )


def _make_scheduler(
    monkeypatch,
    num_blocks=_N,
    *,
    ec_role: ECRole = "ec_both",
    tensor_parallel_size=1,
) -> ECCPUScheduler:
    region = ECSharedRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=num_blocks,
        block_size_bytes=_BS,
    )
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)

    return ECCPUScheduler(
        create_ec_vllm_config(
            ec_role=ec_role,
            tensor_parallel_size=tensor_parallel_size,
        )
    )


def _load_ids(meta) -> list[int]:
    """Transfer ids the scheduler dispatched in this step's metadata."""
    return [transfer_id for transfer_id, _ in meta.loads.values()]


def _load_blocks(meta, mm_hash: str) -> list[int]:
    """Block ids the scheduler dispatched for one load."""
    return meta.loads[mm_hash][1]


def _seed_cached(s: ECCPUScheduler, mm_hash: str, n_blocks: int):
    """Pre-populate a ready cache entry backed by real blocks."""
    s._cache.alloc(mm_hash, n_blocks)
    s._cache.mark_ready(mm_hash)


def test_offload_reuse_cycle(monkeypatch):
    s = _make_scheduler(monkeypatch)
    # Two blocks, so the reload also pins down that every allocated block id
    # comes back in meta.loads, not just the first.
    req = _Request([_Feature("h1", length=2)])

    # Step A: first sight — allocate and save.
    assert s.has_cache_item("h1") is False
    assert s.ensure_cache_available(req, 0) is True
    s.update_state_after_alloc(req, 0)
    meta_a = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta_a.saves
    assert meta_a.loads == {}
    # Not ready until the worker reports the save memcpy complete.
    assert s.has_cache_item("h1") is False

    # Worker reports the save copy finished → entry becomes ready.
    s.update_connector_output(_WorkerOutput(saves=["h1"]))
    assert s.has_cache_item("h1") is True

    # Step C: reload now works — entry is ready at update_state_after_alloc time.
    s.update_state_after_alloc(req, 0)
    meta_c = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta_c.loads
    assert _load_blocks(meta_c, "h1") == meta_a.saves["h1"]

    s.shutdown()


def test_has_cache_item_false_when_not_consumer(monkeypatch):
    s = _make_scheduler(monkeypatch, ec_role="ec_producer")
    assert s.has_cache_item("anything") is False
    s.shutdown()


def test_connector_keys_on_identifier_not_mm_hash(monkeypatch):
    """The connector must key the encoder cache on feature.identifier (what
    has_cache_item is called with), NOT feature.mm_hash."""
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("PROC_KEY", length=1, identifier="ENC_KEY")])
    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "ENC_KEY" in meta.saves
    assert "PROC_KEY" not in meta.saves
    # Not ready until the completion report.
    assert s.has_cache_item("ENC_KEY") is False
    # After the report it becomes ready — keyed on identifier.
    s.update_connector_output(_WorkerOutput(saves=["ENC_KEY"]))
    assert s.has_cache_item("ENC_KEY") is True
    assert s.has_cache_item("PROC_KEY") is False
    s.shutdown()


def test_cpu_region_fifo_eviction(monkeypatch):
    # Region holds exactly 2 one-block encodings.
    s = _make_scheduler(monkeypatch, num_blocks=2)

    # Fill with A and B, marking each ready via a completion report.
    for h in ("A", "B"):
        s.update_state_after_alloc(_Request([_Feature(h, length=1)]), 0)
        s.build_connector_meta(scheduler_output=None)
        s.update_connector_output(_WorkerOutput(saves=[h]))
    assert s.has_cache_item("A") is True
    assert s.has_cache_item("B") is True

    # Allocate C: must evict A (oldest) to make room.
    s.update_state_after_alloc(_Request([_Feature("C", length=1)]), 0)
    s.build_connector_meta(scheduler_output=None)
    assert s.has_cache_item("A") is False  # evicted
    assert s.has_cache_item("B") is True


def test_multiple_mm_items_per_request(monkeypatch):
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("h1", length=1), _Feature("h2", length=1)])
    s.update_state_after_alloc(req, 0)
    s.update_state_after_alloc(req, 1)
    meta = s.build_connector_meta(scheduler_output=None)
    assert set(meta.saves) == {"h1", "h2"}

    # Worker reports both saves complete.
    s.update_connector_output(_WorkerOutput(saves=["h1", "h2"]))
    assert s.has_cache_item("h1") is True
    assert s.has_cache_item("h2") is True

    # Now reload both.
    s.update_state_after_alloc(req, 0)
    s.update_state_after_alloc(req, 1)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert set(meta2.loads) == {"h1", "h2"}
    s.shutdown()


def test_loads_only_serves_hashes_touched_this_step(monkeypatch):
    s = _make_scheduler(monkeypatch)
    _seed_cached(s, "a", n_blocks=2)
    _seed_cached(s, "b", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "a" in meta.loads
    assert "b" not in meta.loads
    s.shutdown()


def test_repeated_reload_same_step_loads_once(monkeypatch):
    """The same mm_hash requested twice in one step must appear in
    meta.loads exactly once."""
    s = _make_scheduler(monkeypatch)
    _seed_cached(s, "a", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert list(meta.loads.keys()).count("a") == 1
    s.shutdown()


def test_load_not_emitted_for_uncached_entry(monkeypatch):
    """A feature that was never saved must not appear in meta.loads."""
    s = _make_scheduler(monkeypatch)
    s.update_state_after_alloc(_Request([_Feature("missing")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert meta.loads == {}
    s.shutdown()


def test_load_pin_protects_blocks_until_completion(monkeypatch):
    """Loaded blocks stay pinned until the worker reports the load memcpy
    complete. Observable: a save that needs eviction cannot reclaim a
    still-pinned loaded entry, but can once the completion report arrives."""
    # 2 blocks, both occupied by "a" (ready).
    s = _make_scheduler(monkeypatch, num_blocks=2)
    _seed_cached(s, "a", n_blocks=2)

    # Load "a" — pins it.
    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "a" in meta.loads
    (transfer_id,) = _load_ids(meta)

    # Try to save "b" (needs 2 blocks, but "a" is pinned) — must fail.
    s.update_state_after_alloc(_Request([_Feature("b", length=2)]), 0)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert "b" not in meta2.saves  # can't evict pinned "a"

    # The one participant reports the load complete → "a" unpins. Now "b" can
    # evict it.
    s.update_connector_output(_WorkerOutput(loads=[transfer_id]))
    s.update_state_after_alloc(_Request([_Feature("b", length=2)]), 0)
    meta3 = s.build_connector_meta(scheduler_output=None)
    assert "b" in meta3.saves
    s.shutdown()


def _can_evict(s: ECCPUScheduler, mm_hash: str, n_blocks: int) -> bool:
    """Whether a new save of `mm_hash` can claim `n_blocks`, i.e. whether the
    blocks currently held by other entries are reclaimable."""
    s.update_state_after_alloc(_Request([_Feature(mm_hash, length=n_blocks)]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    return mm_hash in meta.saves


def test_load_needs_every_participant_before_unpin(monkeypatch):
    """With two participating ranks, one report must not release the pin.

    This is the multi-rank case: each rank copies the same CPU blocks into its
    own GPU memory, so the blocks stay in use until the last rank is done.
    """
    s = _make_scheduler(monkeypatch, num_blocks=2, tensor_parallel_size=2)
    _seed_cached(s, "a", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    (transfer_id,) = _load_ids(s.build_connector_meta(scheduler_output=None))

    # First rank reports: still in use by the second, so "a" must hold.
    s.update_connector_output(_WorkerOutput(loads=[transfer_id]))
    assert _can_evict(s, "b", 2) is False
    assert s.has_pending_push_work() is True

    # Second rank reports: now it is genuinely free.
    s.update_connector_output(_WorkerOutput(loads=[transfer_id]))
    assert s.has_pending_push_work() is False
    assert _can_evict(s, "c", 2) is True
    s.shutdown()


def test_late_report_does_not_release_a_later_load_of_same_hash(monkeypatch):
    """A straggling report from one dispatch must not release the pin taken by
    a subsequent dispatch of the same mm_hash.

    This is what the transfer id is for: mm_hash identifies a cache entry, not
    a transfer, so it cannot tell the two dispatches apart.
    """
    s = _make_scheduler(monkeypatch, num_blocks=2, tensor_parallel_size=2)
    _seed_cached(s, "a", n_blocks=2)

    # First dispatch, fully reported by both ranks → pin released.
    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    (first_id,) = _load_ids(s.build_connector_meta(scheduler_output=None))
    s.update_connector_output(_WorkerOutput(loads=[first_id, first_id]))
    assert s.has_pending_push_work() is False

    # Second dispatch of the same hash takes a fresh pin.
    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    (second_id,) = _load_ids(s.build_connector_meta(scheduler_output=None))
    assert second_id != first_id

    # A replayed report for the first dispatch must be ignored entirely.
    s.update_connector_output(_WorkerOutput(loads=[first_id, first_id]))
    assert s.has_pending_push_work() is True
    assert _can_evict(s, "b", 2) is False
    s.shutdown()


def test_region_full_skips_save_and_never_blocks(monkeypatch):
    """When the region is fully occupied by pinned entries, new saves are
    silently skipped and ensure_cache_available never blocks."""
    s = _make_scheduler(monkeypatch, num_blocks=1)
    _seed_cached(s, "pinned", n_blocks=1)
    s._cache.pin("pinned")

    req = _Request([_Feature("new", length=1)])
    assert s.ensure_cache_available(req, 0) is True

    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "new" not in meta.saves
    assert s.has_cache_item("new") is False

    s._cache.unpin("pinned")
    s.shutdown()


def test_producer_only_never_emits_loads(monkeypatch):
    """A producer-only scheduler must never populate meta.loads, even when
    entries are ready."""
    s = _make_scheduler(monkeypatch, ec_role="ec_producer")
    req = _Request([_Feature("h1", length=1)])

    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta.saves
    assert meta.loads == {}

    s.update_connector_output(_WorkerOutput(saves=["h1"]))

    # Even if we re-encounter the same feature, no load.
    s.update_state_after_alloc(req, 0)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert meta2.loads == {}
    s.shutdown()


def test_consumer_only_never_emits_saves(monkeypatch):
    """A consumer-only scheduler must never populate meta.saves."""
    s = _make_scheduler(monkeypatch, ec_role="ec_consumer")
    _seed_cached(s, "a", n_blocks=1)

    req = _Request([_Feature("a", length=1)])
    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert meta.saves == {}
    assert "a" in meta.loads
    s.shutdown()


def test_consumer_only_has_cache_item(monkeypatch):
    """Consumer-only: has_cache_item works normally for ready entries."""
    s = _make_scheduler(monkeypatch, ec_role="ec_consumer")
    assert s.has_cache_item("a") is False
    _seed_cached(s, "a", n_blocks=1)
    assert s.has_cache_item("a") is True
    s.shutdown()


def test_save_not_emitted_for_already_cached_entry(monkeypatch):
    """An entry already in the cache (from a prior save) must not trigger
    a second allocation or appear in meta.saves again."""
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("h1", length=1)])

    # First encounter → save.
    s.update_state_after_alloc(req, 0)
    meta1 = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta1.saves

    # Second encounter (same step or later) → no save.
    s.update_state_after_alloc(req, 0)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert "h1" not in meta2.saves
    s.shutdown()


# ── completion reporting (update_connector_output) ──────────────────────────


def test_update_connector_output_ignores_foreign_meta(monkeypatch):
    """A worker output whose payload is not an ECCPUWorkerMetadata is a
    no-op — no cache mutation, no crash."""
    s = _make_scheduler(monkeypatch)
    _seed_cached(s, "a", n_blocks=1)

    class _Foreign:
        ec_connector_worker_meta = object()

    s.update_connector_output(_Foreign())  # must not raise
    assert s.has_cache_item("a") is True
    s.shutdown()


def test_update_connector_output_ignores_unknown_report(monkeypatch):
    """Reports for a never-seen save hash or a transfer id this scheduler never
    dispatched are dropped rather than mutating unrelated state."""
    s = _make_scheduler(monkeypatch)
    s.update_connector_output(_WorkerOutput(saves=["gone"], loads=[4242]))
    assert s.has_cache_item("gone") is False
    s.shutdown()


# ── has_pending_push_work ───────────────────────────────────────────────────


def test_has_pending_push_work_tracks_inflight_save(monkeypatch):
    """A dispatched-but-unconfirmed save keeps push work pending until the
    completion report marks it ready."""
    s = _make_scheduler(monkeypatch)
    assert s.has_pending_push_work() is False

    s.update_state_after_alloc(_Request([_Feature("h1", length=1)]), 0)
    s.build_connector_meta(scheduler_output=None)
    assert s.has_pending_push_work() is True  # save not yet confirmed

    s.update_connector_output(_WorkerOutput(saves=["h1"]))
    assert s.has_pending_push_work() is False
    s.shutdown()


def test_has_pending_push_work_tracks_inflight_load(monkeypatch):
    """A dispatched-but-unconfirmed load keeps push work pending until the
    unpin completion report arrives."""
    s = _make_scheduler(monkeypatch)
    _seed_cached(s, "a", n_blocks=1)
    assert s.has_pending_push_work() is False  # ready + unpinned

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert s.has_pending_push_work() is True  # pinned, awaiting unpin

    s.update_connector_output(_WorkerOutput(loads=_load_ids(meta)))
    assert s.has_pending_push_work() is False
    s.shutdown()


# ── shutdown ────────────────────────────────────────────────────────────────


def test_shutdown_disables_roles_and_cleans_region(monkeypatch):
    """After shutdown the scheduler serves no items and the region is
    cleaned up."""
    s = _make_scheduler(monkeypatch, num_blocks=2)
    _seed_cached(s, "a", n_blocks=2)
    assert s.has_cache_item("a") is True

    s.shutdown()
    assert s.has_cache_item("a") is False  # consumer role disabled
