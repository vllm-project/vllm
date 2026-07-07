# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
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
    def __init__(self, features):
        self.mm_features = features


def _make_scheduler(monkeypatch) -> ECCPUScheduler:
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=_N,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = True
        is_ec_consumer = True

    class _Cfg:
        ec_transfer_config = _EC()

    return ECCPUScheduler(_Cfg())


def test_offload_reuse_cycle(monkeypatch):
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("h1", length=1)])

    # Step A: first sight — allocate, save, AND promote in the same step.
    assert s.has_cache_item("h1") is False
    assert s.ensure_cache_available(req, 0) is True
    s.update_state_after_alloc(req, 0)
    meta_a = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta_a.saves
    assert meta_a.loads == {}
    assert s.has_cache_item("h1") is True  # promoted same step

    # Step B: reuse — cache hit re-serves the same blocks as a load.
    s.update_state_after_alloc(req, 0)
    meta_b = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta_b.loads
    assert meta_b.loads["h1"] == meta_a.saves["h1"]

    s.shutdown()


def test_has_cache_item_false_when_not_consumer(monkeypatch):
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=_N,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = True
        is_ec_consumer = False

    class _Cfg:
        ec_transfer_config = _EC()

    s = ECCPUScheduler(_Cfg())
    assert s.has_cache_item("anything") is False
    s.shutdown()


def test_connector_keys_on_identifier_not_mm_hash(monkeypatch):
    """The connector must key the encoder cache on feature.identifier (what
    has_cache_item is called with), NOT feature.mm_hash. They differ under LoRA.
    """
    s = _make_scheduler(monkeypatch)
    # identifier = encoder-output key (what the scheduler passes to
    # has_cache_item); mm_hash = processor key (differs under
    # enable_tower_connector_lora).
    req = _Request([_Feature("PROC_KEY", length=1, identifier="ENC_KEY")])
    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "ENC_KEY" in meta.saves  # saved under the identifier
    assert "PROC_KEY" not in meta.saves
    # matches scheduler's has_cache_item(identifier)
    assert s.has_cache_item("ENC_KEY") is True
    assert s.has_cache_item("PROC_KEY") is False
    s.shutdown()


def test_cpu_region_fifo_eviction(monkeypatch):
    # Region holds exactly 2 one-block encodings.
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=2, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=2,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = True
        is_ec_consumer = True

    class _Cfg:
        ec_transfer_config = _EC()

    s = ECCPUScheduler(_Cfg())

    for h in ("A", "B", "C"):  # 3 into a 2-slot region
        s.update_state_after_alloc(_Request([_Feature(h, length=1)]), 0)
        s.build_connector_meta(scheduler_output=None)
    # A (oldest) evicted to make room for C; B and C remain.
    assert s.has_cache_item("A") is False
    assert s.has_cache_item("B") is True
    assert s.has_cache_item("C") is True
    s.shutdown()


def test_multiple_mm_items_per_request(monkeypatch):
    s = _make_scheduler(monkeypatch)
    req = _Request([_Feature("h1", length=1), _Feature("h2", length=1)])
    s.update_state_after_alloc(req, 0)
    s.update_state_after_alloc(req, 1)
    meta = s.build_connector_meta(scheduler_output=None)
    assert set(meta.saves) == {"h1", "h2"}
    assert s.has_cache_item("h1") and s.has_cache_item("h2")
    # Reuse both on the next step.
    s.update_state_after_alloc(req, 0)
    s.update_state_after_alloc(req, 1)
    meta2 = s.build_connector_meta(scheduler_output=None)
    assert set(meta2.loads) == {"h1", "h2"}
    s.shutdown()


def _seed_cached(s: ECCPUScheduler, mm_hash: str, n_blocks: int) -> list[int]:
    """Pre-populate a locally-cached encoding backed by real region blocks."""
    blocks = s._memory_context.region.alloc(n_blocks)
    s._local_encodings[mm_hash] = None
    s._blocks[mm_hash] = blocks
    return blocks


def test_reload_pins_this_step_and_unpins_after_build(monkeypatch):
    # A cache hit pins the blocks for the step (so eviction cannot reclaim an
    # in-flight reload) and releases them once the load has been served.
    s = _make_scheduler(monkeypatch)
    region = s._memory_context.region
    blocks = _seed_cached(s, "a", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert "a" in s._pending_reload
    assert region.try_free(blocks) is False  # pinned this step

    meta = s.build_connector_meta(scheduler_output=None)
    assert meta.loads == {"a": blocks}
    assert region.try_free(blocks) is True  # unpinned after serving the load
    assert s._pending_reload == set()
    s.shutdown()


def test_build_loads_serves_only_hashes_touched_this_step(monkeypatch):
    # Two cached items, only one referenced this step: the load set must be
    # exactly the touched hash, not every locally cached hash.
    s = _make_scheduler(monkeypatch)
    blocks_a = _seed_cached(s, "a", n_blocks=2)
    _seed_cached(s, "b", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert meta.loads == {"a": blocks_a}
    s.shutdown()


def test_repeated_reload_pins_once_and_single_build_releases(monkeypatch):
    # Two update_state_after_alloc calls for the same mm_hash in one step must
    # pin exactly once, so a single build unpin fully releases the blocks.
    # Guards the `if mm_hash not in self._pending_reload` check.
    s = _make_scheduler(monkeypatch)
    region = s._memory_context.region
    blocks = _seed_cached(s, "a", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert s._pending_reload == {"a"}
    assert region.try_free(blocks) is False  # still pinned

    s.build_connector_meta(scheduler_output=None)
    # If the guard were missing the ref count would be 2 and this would fail.
    assert region.try_free(blocks) is True
    s.shutdown()


def test_update_state_after_alloc_marks_no_reload_for_uncached(monkeypatch):
    # An mm_hash with no local encoding is never pinned/marked for reload
    # (the consumer side is a no-op; the producer side schedules a save).
    s = _make_scheduler(monkeypatch)
    s.update_state_after_alloc(_Request([_Feature("missing")]), 0)
    assert s._pending_reload == set()
    s.shutdown()


def test_shutdown_unpins_pending_reload_blocks(monkeypatch):
    # A step that pinned blocks but never reached build (e.g. teardown) must
    # not leak pins: shutdown releases them before the region is torn down.
    s = _make_scheduler(monkeypatch)
    region = s._memory_context.region
    blocks = _seed_cached(s, "a", n_blocks=2)

    s.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert region.try_free(blocks) is False  # pinned
    s.shutdown()
    assert s._pending_reload == set()
    assert region._ref_count == {}  # every pin was released


def test_fifo_eviction_skips_pinned_entry(monkeypatch):
    # A pending reload pin protects its blocks from FIFO eviction: a new save
    # into a full region reclaims the oldest UNPINNED entry, not the pinned one.
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=2, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=2,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = True
        is_ec_consumer = True

    class _Cfg:
        ec_transfer_config = _EC()

    s = ECCPUScheduler(_Cfg())

    # Fill the 2-block region with A (oldest) and B.
    for h in ("A", "B"):
        s.update_state_after_alloc(_Request([_Feature(h, length=1)]), 0)
        s.build_connector_meta(scheduler_output=None)
    assert s.has_cache_item("A") and s.has_cache_item("B")

    # Pin A via a pending reload, WITHOUT building (so the pin persists into the
    # save step below). Saves run before loads in build_connector_meta, so the
    # eviction loop sees A pinned.
    s.update_state_after_alloc(_Request([_Feature("A", length=1)]), 0)
    assert "A" in s._pending_reload

    # Offload C: region is full, so eviction must skip pinned A and reclaim B.
    s.update_state_after_alloc(_Request([_Feature("C", length=1)]), 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "C" in meta.saves
    assert meta.loads == {"A": s._blocks["A"]}
    assert s.has_cache_item("A")  # pinned, survived
    assert s.has_cache_item("C")  # newly saved
    assert s.has_cache_item("B") is False  # evicted
    s.shutdown()


def test_region_full_skips_save_and_never_blocks(monkeypatch):
    # CPU offload is best-effort and never defers scheduling: ensure_cache_
    # available always returns True, and when the region cannot make room the
    # save is silently skipped (the encoding is not promoted to the cache).
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=1, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=1,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = True
        is_ec_consumer = True

    class _Cfg:
        ec_transfer_config = _EC()

    s = ECCPUScheduler(_Cfg())

    # Occupy and pin the only block so eviction cannot reclaim it.
    blk = region.alloc(1)
    region.pin(blk)
    s._local_encodings["pinned"] = None
    s._blocks["pinned"] = blk

    req = _Request([_Feature("new", length=1)])
    assert s.ensure_cache_available(req, 0) is True  # never blocks

    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "new" not in meta.saves  # save skipped, region full
    assert s.has_cache_item("new") is False  # not promoted

    region.unpin(blk)
    s.shutdown()
