# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import json
import os
import sys
import types
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    MemoryRegion,
    RangeRequest,
    TransferEngineKind,
    TransferOp,
    TransferState,
    format_peer_address,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.factory import (
    create_transfer_backend,
    resolve_transfer_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent import (
    DEFAULT_TRANSFER_TIMEOUT_S,
    MAX_MERGED_LENGTH,
    TentTransferBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    TENT_METRICS_BASE_PORT,
    build_tent_config,
)


class _StatusEnum(enum.Enum):
    COMPLETED = 0
    FAILED = 1
    CANCELED = 2
    INVALID = 3
    PENDING = 4
    TIMEOUT = 5


class _OpCode(enum.Enum):
    WRITE = 1
    READ = 2


class _Status:
    def __init__(self, state: _StatusEnum) -> None:
        self.state = state


class _Request:
    def __init__(
        self,
        opcode: _OpCode,
        source: int,
        target_id: int,
        target_offset: int,
        length: int,
    ) -> None:
        self.opcode = opcode
        self.source = source
        self.target_id = target_id
        self.target_offset = target_offset
        self.length = length


class _Control:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.configs: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.available = True
        self.rpc_port = 16000
        self.segment_name: str | None = None
        self.open_error: Exception | None = None
        self.submit_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.allocate_returns = 1
        self.wait_statuses: list[list[_StatusEnum]] = []
        self.batches = 0

    def reset(self) -> None:
        self.__init__()


_CONTROL = _Control()


class _FakeEngine:
    def __init__(self, config_path: str) -> None:
        with open(config_path) as config_file:
            self.config = json.load(config_file)
        self.config_path = config_path
        _CONTROL.configs.append(self.config)
        _CONTROL.paths.append(config_path)
        self.segments: dict[str, int] = {}

    def available(self) -> bool:
        return _CONTROL.available

    def get_rpc_server_port(self) -> int:
        return _CONTROL.rpc_port

    def get_segment_name(self) -> str:
        if _CONTROL.segment_name is not None:
            return _CONTROL.segment_name
        return format_peer_address(self.config["rpc_server_hostname"], _CONTROL.rpc_port)

    def register_local_memory_batch(self, addresses, lengths) -> None:
        _CONTROL.calls.append(("register", list(addresses), list(lengths)))

    def unregister_local_memory_batch(self, addresses, lengths) -> None:
        _CONTROL.calls.append(("unregister", list(addresses), list(lengths)))

    def open_segment(self, segment_name: str) -> int:
        if _CONTROL.open_error is not None:
            raise _CONTROL.open_error
        handle = len(self.segments) + 1
        self.segments[segment_name] = handle
        _CONTROL.calls.append(("open", segment_name, handle))
        return handle

    def close_segment(self, segment: int) -> None:
        _CONTROL.calls.append(("close", segment))

    def allocate_transfer_batch(self, size: int) -> int:
        if _CONTROL.allocate_returns == 0:
            _CONTROL.calls.append(("allocate", size, 0))
            return 0
        _CONTROL.batches += 1
        _CONTROL.calls.append(("allocate", size, _CONTROL.batches))
        return _CONTROL.batches

    def submit_transfer(self, batch_id: int, requests) -> None:
        if _CONTROL.submit_error is not None:
            raise _CONTROL.submit_error
        _CONTROL.calls.append(("submit", batch_id, list(requests)))

    def wait_transfer(self, batch_id: int, timeout_ms: int) -> list[_Status]:
        _CONTROL.calls.append(("wait", batch_id, timeout_ms))
        if _CONTROL.wait_error is not None:
            raise _CONTROL.wait_error
        states = _CONTROL.wait_statuses.pop(0) if _CONTROL.wait_statuses else []
        return [_Status(state) for state in states]

    def free_transfer_batch(self, batch_id: int) -> None:
        _CONTROL.calls.append(("free", batch_id))


def _install_fake_tent() -> types.ModuleType:
    _CONTROL.reset()
    module = types.ModuleType("tent")
    module.TransferEngine = _FakeEngine  # type: ignore[attr-defined]
    module.Request = _Request  # type: ignore[attr-defined]
    module.OpCode = _OpCode  # type: ignore[attr-defined]
    module.TransferStatusEnum = _StatusEnum  # type: ignore[attr-defined]
    sys.modules["tent"] = module
    return module


@pytest.fixture(autouse=True)
def _drop_fake_module():
    yield
    sys.modules.pop("tent", None)


def _build(
    *,
    hostname: str = "10.0.0.1",
    protocol: str = "tcp",
    device_name: str = "",
    transfer_timeout_s: float = DEFAULT_TRANSFER_TIMEOUT_S,
    overrides: dict[str, Any] | None = None,
) -> TentTransferBackend:
    return TentTransferBackend(
        hostname=hostname,
        protocol=protocol,
        device_name=device_name,
        metrics_port=TENT_METRICS_BASE_PORT,
        transfer_timeout_s=transfer_timeout_s,
        overrides=overrides,
    )


def _ranges(count: int, length: int = 4096) -> list[RangeRequest]:
    """Ranges that cannot merge: both sides advance by twice their length."""
    return [
        RangeRequest(local_addr=index * 2 * length, remote_addr=index * 2 * length, length=length)
        for index in range(count)
    ]


def _calls(kind: str) -> list[tuple[Any, ...]]:
    return [call for call in _CONTROL.calls if call[0] == kind]


def test_backend_removes_the_config_file_after_construction():
    _install_fake_tent()

    backend = _build()

    assert backend.rpc_port == 16000
    assert not os.path.exists(backend._engine.config_path)


def test_backend_rejects_an_unavailable_engine():
    _install_fake_tent()
    _CONTROL.available = False

    with pytest.raises(RuntimeError, match="not available"):
        _build()


def test_backend_rejects_a_segment_name_peers_cannot_reach():
    _install_fake_tent()
    _CONTROL.segment_name = "peer-alias:16000"

    with pytest.raises(RuntimeError, match="advertises segment"):
        _build()


def test_backend_advertises_the_hostname_from_the_config():
    _install_fake_tent()

    backend = _build(overrides={"rpc_server_hostname": "localhost"})

    assert backend._engine.config["rpc_server_hostname"] == "localhost"
    assert backend.rpc_port == 16000


def test_backend_rejects_a_negative_timeout():
    _install_fake_tent()

    with pytest.raises(RuntimeError, match="must not be negative"):
        _build(transfer_timeout_s=-1)


def test_backend_reports_a_missing_tent_module(monkeypatch):
    sys.modules.pop("tent", None)
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent."
        "importlib.util.find_spec",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="requires the 'tent' Python module"):
        _build()


def test_transfer_reports_completion_for_every_range():
    _install_fake_tent()
    _CONTROL.wait_statuses.append([_StatusEnum.COMPLETED] * 3)
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(3))

    assert [event.state for event in events] == [TransferState.COMPLETED] * 3
    assert _calls("wait") == [("wait", 1, int(DEFAULT_TRANSFER_TIMEOUT_S * 1000))]


def test_transfer_maps_every_engine_state():
    _install_fake_tent()
    _CONTROL.wait_statuses.append(
        [
            _StatusEnum.COMPLETED,
            _StatusEnum.FAILED,
            _StatusEnum.INVALID,
            _StatusEnum.TIMEOUT,
            _StatusEnum.CANCELED,
            _StatusEnum.PENDING,
        ]
    )
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.READ, _ranges(6))

    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.FAILED,
        TransferState.FAILED,
        TransferState.FAILED,
        TransferState.CANCELED,
        TransferState.IN_FLIGHT,
    ]
    assert events[1].error == "FAILED"
    assert events[5].error == "PENDING"


def test_transfer_merges_contiguous_ranges():
    _install_fake_tent()
    _CONTROL.wait_statuses.append([_StatusEnum.COMPLETED])
    backend = _build()
    length = 4096

    events = backend.transfer(
        "10.0.0.2:16000",
        TransferOp.WRITE,
        [
            RangeRequest(index * length, index * length, length)
            for index in range(3)
        ],
    )

    submitted = _calls("submit")[0][2]
    assert [request.length for request in submitted] == [3 * length]
    assert [event.state for event in events] == [TransferState.COMPLETED] * 3


def test_transfer_stops_merging_at_the_limit():
    _install_fake_tent()
    _CONTROL.wait_statuses.append([_StatusEnum.COMPLETED] * 2)
    backend = _build()
    half = MAX_MERGED_LENGTH // 2

    backend.transfer(
        "10.0.0.2:16000",
        TransferOp.WRITE,
        [
            RangeRequest(index * half, index * half, half)
            for index in range(3)
        ],
    )

    submitted = _calls("submit")[0][2]
    assert [request.length for request in submitted] == [MAX_MERGED_LENGTH, half]


def test_transfer_reports_failed_ranges_when_the_segment_cannot_open():
    _install_fake_tent()
    _CONTROL.open_error = RuntimeError("peer is gone")
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "peer is gone" in events[0].error
    assert _calls("allocate") == []


def test_transfer_frees_the_batch_when_submission_raises():
    _install_fake_tent()
    _CONTROL.submit_error = RuntimeError("transport down")
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "transport down" in events[0].error
    assert ("free", 1) in _CONTROL.calls


def test_transfer_reports_failed_ranges_when_no_batch_is_available():
    _install_fake_tent()
    _CONTROL.allocate_returns = 0
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(1))

    assert [event.state for event in events] == [TransferState.FAILED]
    assert "no free transfer batch" in events[0].error
    assert _calls("free") == []


def test_transfer_keeps_ranges_in_flight_when_the_wait_raises():
    _install_fake_tent()
    _CONTROL.wait_error = RuntimeError("wait did not settle")
    backend = _build()

    events = backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.IN_FLIGHT] * 2
    assert "did not settle" in events[0].error
    assert _calls("close") != []
    assert ("free", 1) in _CONTROL.calls


def test_transfer_closes_the_segment_after_a_failed_task():
    _install_fake_tent()
    _CONTROL.wait_statuses.append([_StatusEnum.FAILED])
    backend = _build()

    backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(1))

    assert _calls("close") == [("close", 1)]


def test_transfer_reuses_the_peer_segment():
    _install_fake_tent()
    _CONTROL.wait_statuses.extend(
        [[_StatusEnum.COMPLETED], [_StatusEnum.COMPLETED]]
    )
    backend = _build()

    backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(1))
    backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(1))

    assert len(_calls("open")) == 1


def test_transfer_ignores_an_empty_range_list():
    _install_fake_tent()
    backend = _build()

    assert backend.transfer("10.0.0.2:16000", TransferOp.WRITE, []) == []
    assert _CONTROL.calls == []


def test_register_and_unregister_regions_use_batch_calls():
    _install_fake_tent()
    backend = _build()
    regions = [MemoryRegion(0x1000, 4096), MemoryRegion(0x3000, 8192)]

    backend.register_regions(regions)
    backend.unregister_regions(regions)

    assert _calls("register") == [("register", [0x1000, 0x3000], [4096, 8192])]
    assert _calls("unregister") == [("unregister", [0x1000, 0x3000], [4096, 8192])]


def test_empty_regions_skip_the_engine():
    _install_fake_tent()
    backend = _build()

    backend.register_regions([])
    backend.unregister_regions([])

    assert _CONTROL.calls == []


def test_shutdown_closes_open_segments():
    _install_fake_tent()
    _CONTROL.wait_statuses.append([_StatusEnum.COMPLETED])
    backend = _build()
    backend.transfer("10.0.0.2:16000", TransferOp.WRITE, _ranges(1))

    backend.shutdown()

    assert _calls("close") == [("close", 1)]


def test_peer_address_brackets_ipv6():
    assert format_peer_address("2001:db8::1", 9100) == "[2001:db8::1]:9100"
    assert format_peer_address("10.0.0.1", 9100) == "10.0.0.1:9100"


@pytest.mark.parametrize(
    ("protocol", "tcp", "rdma"),
    [("tcp", True, False), ("rdma", False, True)],
)
def test_tent_config_enables_only_the_selected_transport(protocol, tcp, rdma):
    config = build_tent_config(
        hostname="10.0.0.1",
        protocol=protocol,
        device_name="",
        metrics_port=TENT_METRICS_BASE_PORT,
    )

    assert config["transports"]["tcp"]["enable"] is tcp
    assert config["transports"]["rdma"]["enable"] is rdma
    assert config["metadata_type"] == "p2p"
    assert config["rpc_server_hostname"] == "10.0.0.1"


def test_tent_config_lists_rdma_devices_and_merges_overrides():
    config = build_tent_config(
        hostname="10.0.0.1",
        protocol="rdma",
        device_name="mlx5_0,mlx5_1",
        metrics_port=9101,
        overrides={"metrics": {"enabled": False}},
    )

    assert config["topology"]["rdma_whitelist"] == ["mlx5_0", "mlx5_1"]
    assert config["metrics"]["enabled"] is False
    assert config["metrics"]["http_port"] == 9101


def test_tent_config_turns_metrics_off_without_a_port():
    config = build_tent_config(
        hostname="10.0.0.1",
        protocol="tcp",
        device_name="",
        metrics_port=0,
    )

    assert config["metrics"]["enabled"] is False
    assert config["metrics"]["http_port"] == 0


def test_tent_config_rejects_a_protocol_the_engine_cannot_use():
    with pytest.raises(RuntimeError, match="got protocol 'nvlink'"):
        build_tent_config(
            hostname="10.0.0.1",
            protocol="nvlink",
            device_name="",
            metrics_port=TENT_METRICS_BASE_PORT,
        )


def test_resolve_transfer_engine_defaults_to_te():
    assert resolve_transfer_engine({}, environ={}) is TransferEngineKind.TE


def test_resolve_transfer_engine_honours_the_config():
    kind = resolve_transfer_engine({"transfer_engine": "tent"}, environ={})

    assert kind is TransferEngineKind.TENT


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_resolve_transfer_engine_follows_mooncake_env(value):
    kind = resolve_transfer_engine({}, environ={"MC_USE_TENT": value})

    assert kind is TransferEngineKind.TENT


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_resolve_transfer_engine_ignores_a_disabled_env(value):
    kind = resolve_transfer_engine({}, environ={"MC_USE_TENT": value})

    assert kind is TransferEngineKind.TE


def test_resolve_transfer_engine_rejects_a_contradicting_environment():
    with pytest.raises(RuntimeError, match="MC_USE_TENT or MC_USE_TEV1"):
        resolve_transfer_engine({"transfer_engine": "te"}, environ={"MC_USE_TEV1": "1"})


def test_resolve_transfer_engine_rejects_an_unknown_value():
    with pytest.raises(RuntimeError, match="transfer_engine must be one of"):
        resolve_transfer_engine({"transfer_engine": "nixl"}, environ={})


def test_create_backend_rejects_te_without_mooncake():
    with pytest.raises(RuntimeError, match="Mooncake is not available"):
        create_transfer_backend(
            TransferEngineKind.TE,
            engine_cls=None,
            hostname="10.0.0.1",
            protocol="tcp",
            device_name="",
            device_id=0,
        )


def test_create_backend_uses_the_rank_specific_metrics_port():
    _install_fake_tent()

    backend = create_transfer_backend(
        TransferEngineKind.TENT,
        engine_cls=None,
        hostname="10.0.0.1",
        protocol="tcp",
        device_name="",
        device_id=3,
    )

    assert backend._engine.config["metrics"]["http_port"] == TENT_METRICS_BASE_PORT + 3


def test_generated_config_disables_transports_the_connector_did_not_choose():
    config = build_tent_config(
        hostname="10.0.0.1",
        protocol="rdma",
        device_name="",
        metrics_port=TENT_METRICS_BASE_PORT,
    )

    transports = config["transports"]
    assert transports["rdma"] == {"enable": True}
    assert transports["tcp"] == {"enable": False}
    for name in ("gds", "hp_tcp", "io_uring", "mnnvl", "nvlink", "shm"):
        assert transports[name] == {"enable": False}


def test_generated_config_trims_device_names():
    config = build_tent_config(
        hostname="10.0.0.1",
        protocol="rdma",
        device_name="mlx5_0, mlx5_1",
        metrics_port=TENT_METRICS_BASE_PORT,
    )

    assert config["topology"]["rdma_whitelist"] == ["mlx5_0", "mlx5_1"]
