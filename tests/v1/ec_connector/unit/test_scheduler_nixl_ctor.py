# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler

_N = 16
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        engine_id="eng-" + str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def _cfg(enable_nixl):
    class _EC:
        is_ec_producer = True
        is_ec_consumer = True
        engine_id = "eng-" + str(uuid.uuid4())
        ec_enable_nixl = enable_nixl

    class _Model:
        model = "test-model"
        dtype = torch.float16
        hf_config = None

        def get_inputs_embeds_size(self):
            return 32

    class _Cfg:
        ec_transfer_config = _EC()
        model_config = _Model()
        max_concurrent_batches = 1

    return _Cfg()


def test_gate_off_builds_no_nixl(monkeypatch):
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    s = ECCPUScheduler(_cfg(enable_nixl=False))
    assert s._nixl_enabled is False
    assert getattr(s, "_data", None) is None
    assert getattr(s, "_producer_session", None) is None
    s.shutdown()


def test_gate_on_wires_data_transport_and_producer_session(monkeypatch):
    # Port 0 lets the OS pick an ephemeral port so the real ZMQ ROUTER bind
    # in ProducerSession.start() cannot collide with another test/process.
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_PORT", "0")
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    s = ECCPUScheduler(_cfg(enable_nixl=True))
    try:
        assert s._nixl_enabled is True
        assert s._data is not None
        assert s._compat_hash is not None
        assert s._producer_session is not None
        assert s._peer_host == "127.0.0.1"
        assert s._transport is not None  # is_ec_consumer is also True
    finally:
        s.shutdown()
