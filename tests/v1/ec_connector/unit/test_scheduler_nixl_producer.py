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
