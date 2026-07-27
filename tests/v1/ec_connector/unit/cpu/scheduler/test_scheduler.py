# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
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
    """Stand-in for ECConnectorOutput carrying a completion report."""

    def __init__(self, *, saves=None, loads=None):
        self.ec_connector_worker_meta = ECCPUWorkerMetadata(
            completed_saves=list(saves or []),
            completed_loads=list(loads or []),
        )


def _make_scheduler(
    monkeypatch,
    num_blocks=_N,
    *,
    is_producer=True,
    is_consumer=True,
) -> ECCPUScheduler:
    region = ECSharedRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=num_blocks,
        block_size_bytes=_BS,
    )
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)

    _is_prod = is_producer
    _is_cons = is_consumer

    class _EC:
        is_ec_producer = _is_prod
        is_ec_consumer = _is_cons

    class _Cfg:
        ec_transfer_config = _EC()

    return ECCPUScheduler(_Cfg())


def _seed_cached(s: ECCPUScheduler, mm_hash: str, n_blocks: int):
    """Pre-populate a ready cache entry backed by real blocks."""
    s._cache.alloc(mm_hash, n_blocks)
    s._cache.mark_ready(mm_hash)


def test_offload_reuse_cycle(monkeypatch):
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("h1", length=1)])

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
    assert meta_c.loads["h1"] == meta_a.saves["h1"]

    s.shutdown()


def test_has_cache_item_false_when_not_consumer(monkeypatch):
    s = _make_scheduler(monkeypatch, is_consumer=False)
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


def test_load_returns_correct_block_ids(monkeypatch):
    """meta.loads must contain the same block IDs that meta.saves
    allocated — verified through the public API only."""
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("a", length=2)])

    s.update_state_after_alloc(req, 0)
    meta_save = s.build_connector_meta(scheduler_output=None)
    assert "a" in meta_save.saves

    s.update_connector_output(_WorkerOutput(saves=["a"]))

    # Reload.
    s.update_state_after_alloc(req, 0)
    meta_load = s.build_connector_meta(scheduler_output=None)
    assert meta_load.loads["a"] == meta_save.saves["a"]
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

    # Try to save "b" (needs 2 blocks, but "a" is pinned) — must fail.
    s.update_state_after_alloc(_Request([_Feature("b", length=2)]), 0)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert "b" not in meta2.saves  # can't evict pinned "a"

    # Worker reports the load complete → "a" unpins. Now "b" can evict it.
    s.update_connector_output(_WorkerOutput(loads=["a"]))
    s.update_state_after_alloc(_Request([_Feature("b", length=2)]), 0)
    meta3 = s.build_connector_meta(scheduler_output=None)
    assert "b" in meta3.saves
    s.shutdown()


def test_eviction_skips_entry_pinned_by_pending_load(monkeypatch):
    """A loaded entry still awaiting its unpin completion must survive
    eviction attempts from new saves."""
    # 2 blocks: A occupies both, ready.
    s = _make_scheduler(monkeypatch, num_blocks=2)
    _seed_cached(s, "A", n_blocks=2)

    # Load A → pinned.
    s.update_state_after_alloc(_Request([_Feature("A")]), 0)
    meta_load = s.build_connector_meta(scheduler_output=None)
    assert "A" in meta_load.loads

    # Try to save C (needs 2 blocks) — can't evict pinned A.
    s.update_state_after_alloc(_Request([_Feature("C", length=2)]), 0)
    meta_save = s.build_connector_meta(scheduler_output=None)
    assert "C" not in meta_save.saves
    assert s.has_cache_item("A") is True  # survived
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
    s = _make_scheduler(monkeypatch, is_consumer=False)
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
    s = _make_scheduler(monkeypatch, is_producer=False)
    _seed_cached(s, "a", n_blocks=1)

    req = _Request([_Feature("a", length=1)])
    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert meta.saves == {}
    assert "a" in meta.loads
    s.shutdown()


def test_consumer_only_has_cache_item(monkeypatch):
    """Consumer-only: has_cache_item works normally for ready entries."""
    s = _make_scheduler(monkeypatch, is_producer=False)
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


def test_update_connector_output_ignores_unknown_hash(monkeypatch):
    """Completion reports for evicted / never-seen hashes are dropped."""
    s = _make_scheduler(monkeypatch)
    # Neither hash exists in the cache.
    s.update_connector_output(_WorkerOutput(saves=["gone"], loads=["also_gone"]))
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
    s.build_connector_meta(scheduler_output=None)
    assert s.has_pending_push_work() is True  # pinned, awaiting unpin

    s.update_connector_output(_WorkerOutput(loads=["a"]))
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
