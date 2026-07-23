# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler

_N, _BS, _HID, _ES = 16, 64, 32, 2


class _Pos:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


class _Feature:
    def __init__(self, mm_hash, length=1):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        self.mm_position = _Pos(0, length)


class _Request:
    def __init__(self, features, params=None, req_id="r1"):
        self.mm_features = features
        self.request_id = req_id
        self.ec_transfer_params = params


class _FakeResults:
    def __init__(self):
        self.completed = set()
        self.tombstoned = set()
        self.quarantined = set()
        self.cancelled = set()
        self.settled = []


class _FakeSession:
    """Records started transfers; lets the test flip them to completed."""

    def __init__(self):
        self.started = []
        self._results = _FakeResults()

    def start_xfer(self, mm_hash, block_indices, deadline):
        self.started.append(mm_hash)

    def poll(self, messages, now):
        pass

    def take_results(self):
        r, self._results = self._results, _FakeResults()
        return r

    def close(self):
        pass


class _FakeTransport:
    """Stand-in for ZmqClientTransport so _poll_step runs without ZMQ."""

    def poll(self):
        return {}

    def poll_dead(self):
        return []

    def close(self):
        pass


def _consumer_sched(monkeypatch):
    def _region(cfg):
        return ECSharedRegion(
            engine_id="eng-" + str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
        )

    monkeypatch.setattr(sched_mod, "create_ec_shared_region", _region)

    class _EC:
        is_ec_producer = False
        is_ec_consumer = True
        engine_id = "e"
        ec_enable_nixl = False  # build gate-off, then flip fields on

    class _Model:
        model = "test-model"
        dtype = torch.float16
        hf_config = None

        def get_inputs_embeds_size(self):
            return _HID

    class _Cfg:
        ec_transfer_config = _EC()
        model_config = _Model()
        max_concurrent_batches = 1

    s = ECCPUScheduler(_Cfg())
    # Turn on NIXL consumer state without constructing real transports.
    s._nixl_enabled = True
    s._transport = _FakeTransport()
    s._data = None
    s._compat_hash = "c"
    # _setup_nixl normally computes these from model_config; set them
    # directly since this helper builds gate-off then flips fields on.
    s._hidden_dim = _HID
    s._element_size = _ES
    return s


def _params(mm_hash, length):
    return {
        mm_hash: {
            "peer_host": "h",
            "peer_port": 1,
            "size_bytes": length * _HID * _ES,
        }
    }


def test_new_remote_read_defers_then_completes(monkeypatch):
    s = _consumer_sched(monkeypatch)
    fake = _FakeSession()

    # Route _start_xfer to our fake session instead of real ZMQ/NIXL, but
    # still reserve a real not-ready cache entry so mark_ready works.
    def _fake_start(mm_hash, info, size):
        entry = s._cache.alloc(mm_hash, 1)
        assert entry is not None
        fake.started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", _fake_start)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    # Step 1: unseen remote item -> read started, request deferred.
    assert s.ensure_cache_available(req, 0) is False
    assert "h1" in s._in_flight
    assert "h1" in fake.started
    s.build_connector_meta(scheduler_output=None)  # re-arms _first_in_batch

    # Session reports completion this step.
    fake._results.completed.add("h1")
    s._sessions[("h", 1)] = fake

    # Step 2: poll drains completion -> entry marked ready, request admitted.
    assert s.ensure_cache_available(req, 0) is True
    assert "h1" not in s._in_flight
    assert "h1" in s._step_completed
    entry = s._cache.get("h1")
    assert entry is not None and entry.ready
    # The scheduler now allocates encoder input; the ready entry loads through
    # the same local path as a natively cached encoding.
    s.update_state_after_alloc(req, 0)
    meta = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta.loads
    assert "h1" not in s._step_completed  # cleared by promote

    # Step 3: still cached -> admitted directly, no new transfer.
    assert s.ensure_cache_available(req, 0) is True
    s.shutdown()


def test_no_params_is_ready(monkeypatch):
    s = _consumer_sched(monkeypatch)
    assert s.ensure_cache_available(_Request([_Feature("x")], params=None), 0) is True
    s.shutdown()


def test_tombstoned_read_discards_and_blocks_retry(monkeypatch):
    s = _consumer_sched(monkeypatch)
    fake = _FakeSession()

    def _fake_start(mm_hash, info, size):
        entry = s._cache.alloc(mm_hash, 1)
        assert entry is not None
        fake.started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", _fake_start)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    # Step 1: start the read.
    assert s.ensure_cache_available(req, 0) is False
    s.build_connector_meta(scheduler_output=None)

    # Producer rejected the read -> tombstoned.
    fake._results.tombstoned.add("h1")
    s._sessions[("h", 1)] = fake

    # Step 2: poll discards the entry and records a tombstone; the tombstone
    # is consumed by admit so the request proceeds to local compute.
    assert s.ensure_cache_available(req, 0) is True
    assert s._cache.get("h1") is None
    assert "h1" not in s._in_flight
    s.build_connector_meta(scheduler_output=None)

    # Step 3: tombstone was consumed, so a fresh transfer starts again.
    assert s.ensure_cache_available(req, 0) is False
    assert "h1" in s._in_flight
    s.shutdown()


def test_size_mismatch_skips_transfer(monkeypatch):
    s = _consumer_sched(monkeypatch)
    # Advertised size disagrees with pos.length * hidden_dim * element_size.
    bad = {"h1": {"peer_host": "h", "peer_port": 1, "size_bytes": 999}}
    req = _Request([_Feature("h1", 1)], params=bad)
    assert s.ensure_cache_available(req, 0) is True
    assert "h1" not in s._in_flight
    assert s._cache.get("h1") is None
    s.shutdown()


def test_orphan_not_ready_entry_falls_back_no_realloc(monkeypatch):
    # Post-quarantine, post-tombstone-consumed orphan: a not-ready cache
    # entry exists for the mm_hash, but the hash is tracked in none of
    # _in_flight / _step_completed / _tombstones. Its blocks are held by a
    # quarantined/settling DMA and must not be re-allocated.
    s = _consumer_sched(monkeypatch)
    entry = s._cache.alloc("h1", 1)
    assert entry is not None and not entry.ready
    assert "h1" not in s._in_flight
    assert "h1" not in s._step_completed
    assert "h1" not in s._tombstones

    # Spy on alloc: record calls, delegate to the real allocator (which
    # asserts on a duplicate key). A re-alloc for "h1" is the bug.
    real_alloc = s._cache.alloc
    calls: list[str] = []

    def _spy_alloc(key, n):
        calls.append(key)
        return real_alloc(key, n)

    monkeypatch.setattr(s._cache, "alloc", _spy_alloc)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    # Must not raise; request falls back to local compute (admitted True),
    # the orphan entry is untouched, and alloc is never called again.
    assert s.ensure_cache_available(req, 0) is True
    assert calls == []
    assert s._cache.get("h1") is entry
    assert not entry.ready
    s.shutdown()


def test_alloc_failure_falls_back_to_local(monkeypatch):
    s = _consumer_sched(monkeypatch)

    # Force the cache to reject the allocation.
    monkeypatch.setattr(s._cache, "alloc", lambda key, n: None)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    # _start_xfer returns False -> request admitted for local recompute.
    assert s.ensure_cache_available(req, 0) is True
    assert "h1" not in s._in_flight
    s.shutdown()


def test_already_computed_feature_admits_without_transfer(monkeypatch):
    # A feature entirely within num_computed_tokens is skipped by admit; no
    # cache lookup/alloc or transfer should be triggered for it.
    s = _consumer_sched(monkeypatch)
    calls: list[str] = []
    real_alloc = s._cache.alloc

    def _spy_alloc(key, n):
        calls.append(key)
        return real_alloc(key, n)

    monkeypatch.setattr(s._cache, "alloc", _spy_alloc)
    # Feature spans [0, 4); with num_computed_tokens=4 it is fully computed.
    req = _Request([_Feature("h1", 4)], params=_params("h1", 4))

    assert s.ensure_cache_available(req, 4) is True
    assert "h1" not in s._in_flight
    assert calls == []
    s.shutdown()


def test_local_ready_entry_admits_without_transfer(monkeypatch):
    # A READY entry already present in the cache is a local hit: admit
    # returns True and no new transfer is started for that mm_hash.
    s = _consumer_sched(monkeypatch)
    entry = s._cache.alloc("h1", 1)
    assert entry is not None
    s._cache.mark_ready("h1")
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    assert s.ensure_cache_available(req, 0) is True
    assert "h1" not in s._in_flight
    s.shutdown()


def test_in_flight_hash_defers_without_second_transfer(monkeypatch):
    # A mm_hash already tracked in _in_flight must defer the request rather
    # than starting a duplicate transfer.
    s = _consumer_sched(monkeypatch)
    started: list[str] = []

    def _spy_start_xfer(mm_hash, info, size):
        started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", _spy_start_xfer)
    s._in_flight.add("h1")
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    assert s.ensure_cache_available(req, 0) is False
    assert started == []
    s.shutdown()
