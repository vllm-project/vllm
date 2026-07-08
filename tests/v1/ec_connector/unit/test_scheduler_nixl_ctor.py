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


def _ctx():
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )
    return ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=_N,
    )


def _cfg(enable_nixl):
    class _EC:
        is_ec_producer = True
        is_ec_consumer = True
        engine_id = "eng-" + str(uuid.uuid4())
        ec_enable_nixl = enable_nixl

    class _Model:
        model = "test-model"

    class _Cfg:
        ec_transfer_config = _EC()
        model_config = _Model()

    return _Cfg()


def test_gate_off_builds_no_nixl(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(sched_mod, "setup_ec_region", lambda cfg: ctx)
    s = ECCPUScheduler(_cfg(enable_nixl=False))
    assert s._nixl_enabled is False
    assert getattr(s, "_data", None) is None
    assert getattr(s, "_producer_session", None) is None
    s.shutdown()
