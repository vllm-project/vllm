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

_N, _BS = 16, 64


class _Pos:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


class _Feature:
    def __init__(self, mm_hash, length=1):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        self.mm_position = _Pos(0, length)


class _Request:
    def __init__(self, features, req_id="r1"):
        self.mm_features = features
        self.request_id = req_id


def _sched_gate_off(monkeypatch):
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
        engine_id = "e"
        ec_enable_nixl = False

    class _Cfg:
        ec_transfer_config = _EC()

    return ECCPUScheduler(_Cfg())


def test_request_finished_gate_off_returns_none(monkeypatch):
    s = _sched_gate_off(monkeypatch)
    assert s.request_finished(_Request([_Feature("h1")])) == (False, None)
    s.shutdown()


def test_request_finished_producer_emits_params(monkeypatch):
    s = _sched_gate_off(monkeypatch)
    # Simulate NIXL-enabled producer bookkeeping without building real NIXL.
    s._nixl_enabled = True
    s._peer_host, s._peer_port = "1.2.3.4", 5601
    s._local_encodings["h1"] = None
    delay, params = s.request_finished(_Request([_Feature("h1", length=2)]))
    assert delay is False
    assert params == {
        "h1": {"peer_host": "1.2.3.4", "peer_port": 5601, "size_bytes": 2 * 32 * 2}
    }
    s.shutdown()
