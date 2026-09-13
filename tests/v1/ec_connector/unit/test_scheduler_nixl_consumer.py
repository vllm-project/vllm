# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import uuid

import pytest
import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)

_N, _BS, _HID, _ES = 16, 64, 32, 2


class _Pos:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


class _Feature:
    def __init__(self, mm_hash, length=1, data=None):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        self.mm_position = _Pos(0, length)
        self.data = data
        self.modality = "image"


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
        self.retryable = set()
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

    # ec_enable_nixl defaults to False, so this builds gate-off; the fields
    # below flip on the NIXL consumer state directly.
    s = ECCPUScheduler(create_ec_vllm_config(ec_role="ec_consumer"))
    # Turn on NIXL consumer state without constructing real transports.
    s._nixl_enabled = True
    s._transport = _FakeTransport()
    s._data = None
    s._compat_hash = "c"
    # _setup_nixl normally computes these from model_config; set them
    # directly since this helper builds gate-off then flips fields on.
    s._hidden_dim = _HID
    s._element_size = _ES
    s._metadata_resolver._cache["image"] = {"image_grid_thw"}
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


@pytest.mark.parametrize("params", [None, {"x": 123}])
def test_no_remote_announcement_is_ready(monkeypatch, params):
    s = _consumer_sched(monkeypatch)
    assert s.ensure_cache_available(_Request([_Feature("x")], params=params), 0)
    assert s.take_unavailable_requests() == set()
    s.shutdown()


@pytest.mark.parametrize("payload", [None, "metadata", "pixels", "embeds", "shm"])
def test_failed_read_falls_back_only_with_local_input(monkeypatch, payload):
    """A remote miss is fatal only when local model input is unavailable."""
    s = _consumer_sched(monkeypatch)
    fake = _FakeSession()

    def _fake_start(mm_hash, info, size):
        entry = s._cache.alloc(mm_hash, 1)
        assert entry is not None
        fake.started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", _fake_start)
    fields = {
        "metadata": {"image_grid_thw": torch.tensor([1, 2, 2])},
        "pixels": {"pixel_values": torch.ones(1, 3)},
        "embeds": {"image_embeds": torch.ones(1, _HID)},
        "shm": {"address": 123, "monotonic_id": 1},
    }
    data = (
        MultiModalKwargsItem(
            {
                key: MultiModalFieldElem(value, MultiModalSharedField(batch_size=1))
                for key, value in fields[payload].items()
            }
        )
        if payload is not None
        else None
    )
    req = _Request([_Feature("h1", 1, data)], params=_params("h1", 1))

    # Step 1: start the read.
    assert s.ensure_cache_available(req, 0) is False
    s.build_connector_meta(scheduler_output=None)

    fake._results.tombstoned.add("h1")
    s._sessions[("h", 1)] = fake

    can_fallback = payload in ("pixels", "embeds", "shm")
    assert s.ensure_cache_available(req, 0) is can_fallback
    assert s._cache.get("h1") is None
    assert "h1" not in s._in_flight
    assert s.take_unavailable_requests() == (set() if can_fallback else {"r1"})
    # Drained by the read, so the scheduler cannot abort it twice.
    assert s.take_unavailable_requests() == set()
    if can_fallback:
        # A later scheduling step must not retry the failed remote source.
        s.build_connector_meta(scheduler_output=None)
        assert s.ensure_cache_available(req, 0)
        assert fake.started == ["h1"]
    s.shutdown()


@pytest.mark.parametrize("retry", [False, True], ids=["in-flight", "retryable"])
def test_remote_wait_budget_survives_retries_and_long_steps(monkeypatch, retry):
    """Neither retries nor delayed scheduling may restart the request budget."""
    s = _consumer_sched(monkeypatch)
    fake = _FakeSession()
    s._sessions[("h", 1)] = fake

    def start(mm_hash, info, size):
        assert s._cache.alloc(mm_hash, 1) is not None
        fake.started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", start)
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    req = _Request([_Feature("h1")], params=_params("h1", 1))
    assert not s.ensure_cache_available(req, 0)
    for elapsed in (20, 40, 121):
        now[0] = 1000.0 + elapsed
        if retry:
            fake._results.retryable.add("h1")
        s.build_connector_meta(scheduler_output=None)
        assert not s.ensure_cache_available(req, 0)
        assert s.take_unavailable_requests() == ({"r1"} if elapsed > 60 else set())
    assert len(fake.started) == (3 if retry else 1)
    s.shutdown()


def test_retryable_read_re_requests_without_admitting(monkeypatch):
    """A not-ready NACK defers the request and re-requests the read.

    Unlike a tombstone it never admits the request, because an admitted request
    whose media was rewritten away upstream has no embedding and no way to
    recompute one.
    """
    s = _consumer_sched(monkeypatch)
    fake = _FakeSession()

    def _fake_start(mm_hash, info, size):
        entry = s._cache.alloc(mm_hash, 1)
        assert entry is not None
        fake.started.append(mm_hash)
        return True

    monkeypatch.setattr(s, "_start_xfer", _fake_start)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    assert s.ensure_cache_available(req, 0) is False
    s.build_connector_meta(scheduler_output=None)

    # Producer's save has not landed: retryable, not tombstoned.
    fake._results.retryable.add("h1")
    s._sessions[("h", 1)] = fake

    # The entry is released and the read re-issued in the same step, so the
    # request stays deferred rather than being admitted unfetched.
    assert s.ensure_cache_available(req, 0) is False
    assert "h1" in s._in_flight
    assert "h1" not in s._tombstones
    assert fake.started.count("h1") == 2
    s.shutdown()


@pytest.mark.parametrize(
    "size_fields",
    [{"size_bytes": 999}, {"size_bytes": None}, {"size_bytes": "big"}, {}],
    ids=["mismatch", "null-size", "text-size", "no-size"],
)
def test_invalid_remote_size_fails_the_request(monkeypatch, size_fields):
    """Invalid remote sizes fail the request without raising in the scheduler."""
    s = _consumer_sched(monkeypatch)
    announced = {"peer_host": "h", "peer_port": 1, **size_fields}
    req = _Request([_Feature("h1", 1)], params={"h1": announced})
    assert s.ensure_cache_available(req, 0) is False
    assert s.take_unavailable_requests() == {"r1"}
    assert "h1" not in s._in_flight
    assert s._cache.get("h1") is None
    s.shutdown()


def test_deferral_budget_is_per_request(monkeypatch):
    """Shared hashes have independent wait budgets and cancellation cleanup."""
    s = _consumer_sched(monkeypatch)
    monkeypatch.setattr(s._cache, "alloc", lambda key, n: None)
    params = _params("h1", 1)
    old = _Request([_Feature("h1", 1)], params=params, req_id="old")
    new = _Request([_Feature("h1", 1)], params=params, req_id="new")

    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert s.ensure_cache_available(old, 0) is False
    now[0] += sched_mod._ADMIT_DEFER_TIMEOUT_S - 1
    assert s.ensure_cache_available(new, 0) is False
    assert s.take_unavailable_requests() == set()

    now[0] += 2
    # `old` is now past its budget; `new` is not.
    assert s.ensure_cache_available(new, 0) is False
    assert s.take_unavailable_requests() == set()
    assert s.request_finished(new) == (False, None)
    assert set(s._deferred_since) == {("old", "h1")}
    assert s.ensure_cache_available(old, 0) is False
    assert s.take_unavailable_requests() == {"old"}
    assert "h1" not in s._in_flight
    s.shutdown()


def test_metadata_only_entry_admits_without_transfer(monkeypatch):
    # A producer that reported placeholder metadata but had no cache entry
    # to transfer omits peer_host/peer_port/size_bytes entirely; the consumer
    # must fall back to local compute rather than treating that as a size
    # mismatch.
    s = _consumer_sched(monkeypatch)
    metadata_only = {"h1": {"image_grid_thw": [1, 2, 3]}}
    req = _Request([_Feature("h1", 1)], params=metadata_only)
    assert s.ensure_cache_available(req, 0) is True
    assert "h1" not in s._in_flight
    assert s._cache.get("h1") is None
    s.shutdown()


def test_orphan_not_ready_entry_defers_no_realloc(monkeypatch):
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

    # Must not raise; the request is deferred while the DMA settles, the
    # orphan entry is untouched, and alloc is never called again.
    assert s.ensure_cache_available(req, 0) is False
    assert calls == []
    assert s._cache.get("h1") is entry
    assert not entry.ready
    assert s.take_unavailable_requests() == set()
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
