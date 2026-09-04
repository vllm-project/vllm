# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler

_N, _BS = 16, 64


class _Pos:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


class _Feature:
    def __init__(self, mm_hash, length=1):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        self.mm_position = _Pos(0, length)
        self.data = None
        self.modality = "image"


class _Request:
    def __init__(self, features, req_id="r1"):
        self.mm_features = features
        self.request_id = req_id


def _sched_gate_off(monkeypatch):
    def _region(cfg):
        return ECSharedRegion(
            engine_id="eng-" + str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
        )

    monkeypatch.setattr(sched_mod, "create_ec_shared_region", _region)
    return ECCPUScheduler(create_ec_vllm_config(ec_role="ec_both"))


def test_request_finished_gate_off_returns_none(monkeypatch):
    s = _sched_gate_off(monkeypatch)
    assert s.request_finished(_Request([_Feature("h1")])) == (False, None)
    s.shutdown()


def test_request_finished_producer_emits_params(monkeypatch):
    s = _sched_gate_off(monkeypatch)
    # Simulate NIXL-enabled producer bookkeeping without building real NIXL.
    s._nixl_enabled = True
    s._peer_host, s._peer_port = "1.2.3.4", 5601
    # _setup_nixl normally computes these from model_config; set them
    # directly since this test builds gate-off then flips fields on.
    s._hidden_dim, s._element_size = 32, 2
    # feature length=2, hidden_dim=32, element_size=2 -> 128 bytes -> 2 blocks.
    entry = s._cache.alloc("h1", 2)
    assert entry is not None
    s._cache.mark_ready("h1")

    delay, params = s.request_finished(_Request([_Feature("h1", length=2)]))
    assert delay is False
    assert params == {
        "h1": {
            "metadata": {},
            "peer_host": "1.2.3.4",
            "peer_port": 5601,
            "size_bytes": 2 * 32 * 2,
        }
    }
    s.shutdown()


def test_request_finished_announces_not_ready_entry(monkeypatch):
    """A save whose GPU->mmap copy hasn't been confirmed complete yet is
    still announced: the entry can't be evicted before it's ready, and a read
    arriving too early is NACKed NACK_NOT_READY for the consumer to retry —
    protecting it is not this method's job."""
    s = _sched_gate_off(monkeypatch)
    s._nixl_enabled = True
    s._peer_host, s._peer_port = "1.2.3.4", 5601
    s._hidden_dim, s._element_size = 32, 2
    s._cache.alloc("h1", 2)  # allocated but not marked ready

    delay, params = s.request_finished(_Request([_Feature("h1", length=2)]))
    assert delay is False
    assert params == {
        "h1": {
            "metadata": {},
            "peer_host": "1.2.3.4",
            "peer_port": 5601,
            "size_bytes": 2 * 32 * 2,
        }
    }
    s.shutdown()


def test_request_finished_skips_unallocated_entry(monkeypatch):
    s = _sched_gate_off(monkeypatch)
    s._nixl_enabled = True
    s._peer_host, s._peer_port = "1.2.3.4", 5601
    s._hidden_dim, s._element_size = 32, 2
    # No alloc() for "h1" at all — e.g. the cache was full at save time.

    delay, params = s.request_finished(_Request([_Feature("h1", length=2)]))
    assert delay is False
    # The item's placeholder metadata is still reported even though there's
    # no cache entry to transfer (empty "metadata": no fields, no transfer).
    assert params == {"h1": {"metadata": {}}}
    s.shutdown()


# ── announced encodings are held until read ──────────────────────────────────


class _FakeProducerSession:
    """Stands in for ProducerSession so build_connector_meta can run."""

    def __init__(self):
        self.served: list[str] = []

    def poll_step(self):
        pass

    def take_served(self):
        served, self.served = self.served, []
        return served


def _announcing_sched(monkeypatch, lease=30.0):
    s = _sched_gate_off(monkeypatch)
    s._nixl_enabled = True
    s._peer_host, s._peer_port = "1.2.3.4", 5601
    s._hidden_dim, s._element_size = 32, 2
    s._announce_lease_s = lease
    s._producer_session = _FakeProducerSession()
    return s


def _announce(s, mm_hash, length=2):
    return s.request_finished(_Request([_Feature(mm_hash, length=length)]))


def test_announced_encoding_is_not_evictable(monkeypatch):
    """Announcing publishes an address the consumer uses on a later step.

    By then the orchestrator has rewritten the media off the request, so an
    eviction inside that window leaves the consumer nothing to fall back on.
    """
    s = _announcing_sched(monkeypatch)
    entry = s._cache.alloc("h1", 2)
    assert entry is not None
    s._cache.mark_ready("h1")
    assert entry.evictable

    _delay, params = _announce(s, "h1")
    assert "peer_host" in params["h1"]
    assert not entry.evictable
    s.shutdown()


def test_hold_is_released_once_the_read_lands(monkeypatch):
    s = _announcing_sched(monkeypatch)
    entry = s._cache.alloc("h1", 2)
    s._cache.mark_ready("h1")
    _announce(s, "h1")
    assert not entry.evictable

    s._producer_session.served.append("h1")
    s.build_connector_meta(scheduler_output=None)
    assert entry.evictable
    s.shutdown()


def test_hold_lapses_when_no_consumer_ever_reads(monkeypatch):
    """A consumer that never asks must not pin the pool forever."""
    s = _announcing_sched(monkeypatch, lease=0.0)
    entry = s._cache.alloc("h1", 2)
    s._cache.mark_ready("h1")
    _announce(s, "h1")
    assert not entry.evictable

    import time as _time

    _time.sleep(0.01)
    s.build_connector_meta(scheduler_output=None)
    assert entry.evictable
    s.shutdown()


def test_hold_is_taken_when_a_late_save_lands(monkeypatch):
    """An entry announced mid-save is pinned when it becomes evictable.

    A not-ready entry cannot be evicted, so the hold has to start at
    mark_ready rather than at the announcement.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
        ECCPUWorkerMetadata,
    )
    from vllm.v1.outputs import ECConnectorOutput

    s = _announcing_sched(monkeypatch)
    entry = s._cache.alloc("h1", 2)
    assert entry is not None and not entry.ready
    _announce(s, "h1")
    assert s._announce["h1"].pending == 1

    s.update_connector_output(
        ECConnectorOutput(
            ec_connector_worker_meta=ECCPUWorkerMetadata(completed_saves=["h1"])
        )
    )
    assert entry.ready
    assert not entry.evictable
    assert not s._announce["h1"].pending
    s.shutdown()


def test_each_announcement_keeps_its_own_hold(monkeypatch):
    """Two consumers told to read the same encoding each need it to survive.

    Releasing on the first read would let the entry be evicted while the
    second consumer, whose media has already been rewritten to a remote
    reference, has not started its read.
    """
    s = _announcing_sched(monkeypatch)
    entry = s._cache.alloc("h1", 2)
    s._cache.mark_ready("h1")
    _announce(s, "h1")
    _announce(s, "h1")

    s._producer_session.served.append("h1")
    s.build_connector_meta(scheduler_output=None)
    assert not entry.evictable

    s._producer_session.served.append("h1")
    s.build_connector_meta(scheduler_output=None)
    assert entry.evictable
    s.shutdown()
