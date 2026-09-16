# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import pytest

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.config.ec_transfer import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.nixl_utils import NixlWrapper

_N = 16
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        engine_id="eng-" + str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def test_ec_enable_nixl_absent_from_extra_config_by_default():
    cfg = ECTransferConfig()
    assert cfg.get_from_extra_config("ec_enable_nixl", False) is False


def test_ec_enable_nixl_read_from_extra_config():
    cfg = ECTransferConfig(
        ec_connector="ECCPUConnector",
        ec_role="ec_both",
        ec_connector_extra_config={"ec_enable_nixl": True},
    )
    assert cfg.get_from_extra_config("ec_enable_nixl", False) is True


def test_gate_off_builds_no_nixl(monkeypatch):
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    # ec_enable_nixl defaults to False.
    s = ECCPUScheduler(create_ec_vllm_config(ec_role="ec_both"))
    assert s._nixl_enabled is False
    assert getattr(s, "_data", None) is None
    assert getattr(s, "_producer_session", None) is None
    s.shutdown()


def test_string_false_in_extra_config_leaves_nixl_off(monkeypatch):
    """Extra config is not type coerced, and bool("false") is True."""
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    cfg = create_ec_vllm_config(ec_role="ec_both")
    cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = "false"
    s = ECCPUScheduler(cfg)
    assert s._nixl_enabled is False
    s.shutdown()


@pytest.mark.skipif(NixlWrapper is None, reason="Requires NIXL package")
def test_gate_on_wires_data_transport_and_producer_session(monkeypatch):
    # Port 0 lets the OS pick an ephemeral port so the real ZMQ ROUTER bind
    # in ProducerSession.start() cannot collide with another test/process.
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_PORT", "0")
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    cfg = create_ec_vllm_config(ec_role="ec_both")
    cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = True
    s = ECCPUScheduler(cfg)
    try:
        assert s._nixl_enabled is True
        assert s._data is not None
        assert s._compat_hash is not None
        assert s._producer_session is not None
        assert s._peer_host == "127.0.0.1"
        assert s._transport is not None  # is_ec_consumer is also True
    finally:
        s.shutdown()
