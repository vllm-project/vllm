# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import ECSharedRegion

_N = 16
_BS = 64


class _Pos:
    def __init__(self, offset, length):
        self.offset = offset
        self.length = length


class _Feature:
    def __init__(self, mm_hash, length=1):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
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
