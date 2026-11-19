# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from pathlib import Path

import pytest

import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
import vllm.distributed.nixl_utils as nixl_utils
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.config.ec_transfer import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.nixl_utils import NixlWrapper
from vllm.utils.network_utils import get_open_port

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


@pytest.mark.cpu_test
def test_missing_nixl_releases_shared_region(monkeypatch):
    """A failed constructor leaves no scheduler for the caller to shut down."""
    # Import before patching to avoid caching the missing-NIXL values.
    from vllm.distributed.ec_transfer.ec_connector.cpu.data import nixl  # noqa: F401

    region = _region()
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)
    monkeypatch.setattr(nixl_utils, "NixlWrapper", None)
    monkeypatch.setattr(nixl_utils, "nixl_agent_config", None)
    cfg = create_ec_vllm_config(ec_role="ec_both")
    cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = True

    try:
        with pytest.raises(RuntimeError, match="ec_enable_nixl requires NIXL"):
            ECCPUScheduler(cfg)

        assert not Path(region._mmap_path).exists()
        assert region._fd is None
        assert region._mmap_obj is None
    finally:
        region.cleanup()


@pytest.mark.cpu_test
def test_control_transport_failure_releases_nixl_and_shared_region(monkeypatch):
    import vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq as zmq_mod
    import vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl as data_mod

    class RegisteredData:
        registered = True

        def deregister(self):
            self.registered = False

    data = RegisteredData()
    error = RuntimeError("control transport failed")

    def fail_control_transport():
        raise error

    region = _region()
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)
    monkeypatch.setattr(nixl_utils, "NixlWrapper", object())
    monkeypatch.setattr(nixl_utils, "nixl_agent_config", object())
    monkeypatch.setattr(data_mod, "NixlDataTransport", lambda **kwargs: data)
    monkeypatch.setattr(zmq_mod, "ZmqClientTransport", fail_control_transport)
    cfg = create_ec_vllm_config(ec_role="ec_consumer")
    cfg.model_config.hf_config = None
    cfg.model_config.get_inputs_embeds_size.return_value = 8
    cfg.model_config.model = "test-model"
    cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = True

    try:
        with pytest.raises(RuntimeError, match="control transport failed") as exc:
            ECCPUScheduler(cfg)

        assert exc.value is error
        assert not data.registered
        assert not Path(region._mmap_path).exists()
        assert region._fd is None
        assert region._mmap_obj is None
    finally:
        region.cleanup()


@pytest.mark.skipif(NixlWrapper is None, reason="Requires NIXL package")
@pytest.mark.parametrize("dp_index", [0, 1])
@pytest.mark.parametrize("port_mode", ["unset", "zero", "fixed"])
def test_gate_on_wires_data_transport_and_producer_session(
    monkeypatch, dp_index, port_mode
):
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.delenv("VLLM_EC_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_EC_SIDE_CHANNEL_PORT", raising=False)
    port = 0
    if port_mode == "fixed":
        port = get_open_port()
        monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_PORT", str(port - dp_index))
        monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_HOST", "localhost")
    elif port_mode == "zero":
        monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_PORT", "0")
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    cfg = create_ec_vllm_config(ec_role="ec_both")
    cfg.parallel_config.data_parallel_index = dp_index
    cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = True
    s = ECCPUScheduler(cfg)
    try:
        assert s._nixl_enabled is True
        assert s._data is not None
        assert s._compat_hash is not None
        assert s._producer_session is not None
        assert s._peer_host == ("localhost" if port else "127.0.0.1")
        assert s._peer_port is not None and 0 < s._peer_port <= 65535
        if port:
            assert s._peer_port == port
        else:
            assert s._peer_port != dp_index  # Port 0 must not get the DP offset.
        assert s._transport is not None  # is_ec_consumer is also True
    finally:
        s.shutdown()


@pytest.mark.skipif(NixlWrapper is None, reason="Requires NIXL package")
def test_producers_allocate_distinct_ports_and_agent_names(monkeypatch):
    """DP replicas retain their configured identity without peer collisions."""
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.delenv("VLLM_EC_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_EC_SIDE_CHANNEL_PORT", raising=False)
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: _region())
    engine_id = f"ec-test-{uuid.uuid4()}"
    schedulers = []
    expected_names = []
    try:
        for engine, dp_index in [(engine_id, 0), (engine_id, 1), (None, 0)]:
            cfg = create_ec_vllm_config(ec_role="ec_producer")
            if engine is not None:
                cfg.ec_transfer_config.engine_id = engine
            cfg.parallel_config.data_parallel_index = dp_index
            cfg.ec_transfer_config.ec_connector_extra_config["ec_enable_nixl"] = True
            expected_names.append(f"{cfg.ec_transfer_config.engine_id}_dp{dp_index}")
            schedulers.append(ECCPUScheduler(cfg))
        ports = [s._peer_port for s in schedulers]
        assert all(port is not None and 0 < port <= 65535 for port in ports)
        assert len(set(ports)) == len(schedulers)
        for index, scheduler in enumerate(schedulers):
            peer = schedulers[(index + 1) % len(schedulers)]._data
            name = peer.add_remote_peer(
                scheduler._data.get_agent_metadata(),
                scheduler._data.get_mem_descriptor(),
            )
            if isinstance(name, bytes):
                name = name.decode()
            assert name == expected_names[index]
    finally:
        for scheduler in reversed(schedulers):
            scheduler.shutdown()
