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
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=_HID,
        element_size=_ES,
        block_size_bytes=_BS,
        num_blocks=_N,
    )
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)

    class _EC:
        is_ec_producer = False
        is_ec_consumer = True
        engine_id = "e"
        ec_enable_nixl = False  # build gate-off, then flip fields on

    class _Cfg:
        ec_transfer_config = _EC()

    s = ECCPUScheduler(_Cfg())
    # Turn on NIXL consumer state without constructing real transports.
    s._nixl_enabled = True
    s._transport = _FakeTransport()
    s._data = None
    s._compat_hash = "c"
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

    # Route _start_xfer to our fake session instead of real ZMQ/NIXL.
    def _fake_start(mm_hash, info, size):
        s._blocks[mm_hash] = s._fifo_alloc(1)
        fake.started.append(mm_hash)

    monkeypatch.setattr(s, "_start_xfer", _fake_start)
    req = _Request([_Feature("h1", 1)], params=_params("h1", 1))

    # Step 1: unseen remote item -> read started, request deferred.
    assert s.ensure_cache_available(req, 0) is False
    assert "h1" in s._in_flight
    s.build_connector_meta(scheduler_output=None)  # re-arms _first_in_batch

    # Session reports completion this step.
    fake._results.completed.add("h1")
    s._sessions[("h", 1)] = fake

    # Step 2: poll drains completion; still deferred (awaiting promote).
    assert s.ensure_cache_available(req, 0) is False
    assert "h1" in s._step_completed
    meta = s.build_connector_meta(scheduler_output=None)
    assert "h1" in meta.loads
    assert "h1" in s._local_encodings

    # Step 3: now local -> ready.
    assert s.ensure_cache_available(req, 0) is True
    s.shutdown()


def test_no_params_is_ready(monkeypatch):
    s = _consumer_sched(monkeypatch)
    assert s.ensure_cache_available(_Request([_Feature("x")], params=None), 0) is True
    s.shutdown()
