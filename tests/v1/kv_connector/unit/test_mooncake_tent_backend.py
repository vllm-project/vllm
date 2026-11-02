# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import json
import os
import sys
import types
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
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
    MAX_MERGED_LENGTH,
    TentTransferBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    TENT_METRICS_BASE_PORT,
    build_tent_config,
)


class _StatusEnum(enum.Enum):
    INITIAL = 0
    PENDING = 1
    INVALID = 2
    CANCELED = 3
    COMPLETED = 4
    TIMEOUT = 5
    FAILED = 6


class _OpCode(enum.Enum):
    READ = 0
    WRITE = 1


class _Status:
    def __init__(self, state: _StatusEnum) -> None:
        self.state = state


class _TentError(Exception):
    """Base of the exceptions the tent module exports."""


class _TentNotImplemented(_TentError):
    """Mirrors `tent.NotImplementedError`, which is not Python's builtin."""


class _Request:
    def __init__(self, opcode, source, target_id, target_offset, length) -> None:
        self.opcode = opcode
        self.source = source
        self.target_id = target_id
        self.target_offset = target_offset
        self.length = length


class _Control:
    """Settings the fake module reads while a backend starts."""

    def __init__(self) -> None:
        self.segment_name = "10.0.0.1:9100"
        self.rpc_port = 9100
        self.available = True
        self.polls: list[list[_StatusEnum]] = []
        self.overall: list[_StatusEnum] | None = None
        self.cancel_error: BaseException | None = None
        self.submit_error: BaseException | None = None
        self.allocate_fails = False


_CONTROL = _Control()


class _FakeEngine:
    def __init__(self, config_path: str) -> None:
        with open(config_path) as handle:
            self.config = json.load(handle)
        self.rpc_port = _CONTROL.rpc_port
        self.segment_name = _CONTROL.segment_name
        self.calls: list[tuple] = []
        self.batches: dict[int, list[list[_StatusEnum]]] = {}
        self._next_batch = 0
        self._next_segment = 100

    def available(self) -> bool:
        return _CONTROL.available

    def get_rpc_server_port(self) -> int:
        return self.rpc_port

    def get_segment_name(self) -> str:
        return self.segment_name

    def register_local_memory_batch(self, addresses, lengths) -> None:
        self.calls.append(("register", list(addresses), list(lengths)))

    def unregister_local_memory_batch(self, addresses, lengths) -> None:
        self.calls.append(("unregister", list(addresses), list(lengths)))

    def allocate_transfer_batch(self, size: int) -> int:
        if _CONTROL.allocate_fails:
            self.calls.append(("allocate-failed", size))
            return 0
        self._next_batch += 1
        self.batches[self._next_batch] = [list(script) for script in _CONTROL.polls]
        self.calls.append(("allocate", self._next_batch, size))
        return self._next_batch

    def free_transfer_batch(self, batch_id: int) -> None:
        self.calls.append(("free", batch_id))

    def open_segment(self, segment_name: str) -> int:
        self._next_segment += 1
        self.calls.append(("open", segment_name, self._next_segment))
        return self._next_segment

    def close_segment(self, segment: int) -> None:
        self.calls.append(("close", segment))

    def submit_transfer(self, batch_id: int, requests) -> None:
        if _CONTROL.submit_error is not None:
            raise _CONTROL.submit_error
        self.calls.append(("submit", batch_id, list(requests)))

    def cancel_transfer(self, batch_id: int, index: int) -> None:
        if _CONTROL.cancel_error is not None:
            raise _CONTROL.cancel_error
        self.calls.append(("cancel", batch_id, index))

    def get_transfer_status_list(self, batch_id: int) -> list[_Status]:
        script = self.batches[batch_id]
        states = script.pop(0) if len(script) > 1 else script[0]
        return [_Status(state) for state in states]

    def get_transfer_status_overall(self, batch_id: int) -> _Status:
        """Reports the batch as a whole and lets the script run ahead."""
        if _CONTROL.overall:
            overall_script = _CONTROL.overall
            head = (
                overall_script.pop(0) if len(overall_script) > 1 else overall_script[0]
            )
            return _Status(head)
        steps = self.batches[batch_id]
        states = steps[0]
        settled = {
            _StatusEnum.COMPLETED,
            _StatusEnum.FAILED,
            _StatusEnum.TIMEOUT,
            _StatusEnum.INVALID,
            _StatusEnum.CANCELED,
        }
        if all(state in settled for state in states):
            worst = next(
                (state for state in states if state is not _StatusEnum.COMPLETED),
                _StatusEnum.COMPLETED,
            )
            return _Status(worst)
        if len(steps) > 1:
            steps.pop(0)
        return _Status(_StatusEnum.PENDING)


def _install_fake_tent(**settings: Any) -> types.ModuleType:
    global _CONTROL
    _CONTROL = _Control()
    for name, value in settings.items():
        setattr(_CONTROL, name, value)
    module = types.ModuleType("tent")
    module.__dict__.update(
        {
            "TransferEngine": _FakeEngine,
            "TransferStatusEnum": _StatusEnum,
            "OpCode": _OpCode,
            "Request": _Request,
            "TentException": _TentError,
            "NotImplementedError": _TentNotImplemented,
        }
    )
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
    overrides: dict[str, Any] | None = None,
) -> TentTransferBackend:
    return TentTransferBackend(
        hostname=hostname,
        protocol=protocol,
        device_name=device_name,
        metrics_port=TENT_METRICS_BASE_PORT,
        overrides=overrides,
    )


def _ranges(count: int) -> list[RangeRequest]:
    return [
        RangeRequest(local_addr=0x1000 + index, remote_addr=0x2000 + index, length=64)
        for index in range(count)
    ]


def test_backend_reports_its_kind_and_port():
    _install_fake_tent()
    backend = _build()

    assert backend.kind is TransferEngineKind.TENT
    assert backend.rpc_port == 9100


def test_backend_removes_the_config_file_after_construction(monkeypatch):
    _install_fake_tent()
    seen: dict[str, Any] = {}

    class _RecordingEngine(_FakeEngine):
        def __init__(self, config_path: str) -> None:
            super().__init__(config_path)
            seen["existed_during_init"] = os.path.exists(config_path)
            seen["path"] = config_path

    monkeypatch.setattr(sys.modules["tent"], "TransferEngine", _RecordingEngine)
    _build()

    assert seen["existed_during_init"] is True
    assert not os.path.exists(seen["path"])


def test_backend_rejects_an_unavailable_engine():
    _install_fake_tent(available=False)

    with pytest.raises(RuntimeError, match="not available over 'tcp'"):
        _build()


def test_backend_rejects_a_segment_name_peers_cannot_reach():
    _install_fake_tent(segment_name="other:1234")

    with pytest.raises(RuntimeError, match="advertises segment 'other:1234'"):
        _build()


def test_transfer_reports_completion_for_every_range():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED, _StatusEnum.COMPLETED]])
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.COMPLETED,
    ]
    engine = backend._engine
    assert ("free", 1) in engine.calls
    submitted = [call for call in engine.calls if call[0] == "submit"][0]
    assert [request.length for request in submitted[2]] == [64, 64]


def test_transfer_reports_each_range_outcome_separately():
    _install_fake_tent(
        polls=[[_StatusEnum.COMPLETED, _StatusEnum.FAILED, _StatusEnum.TIMEOUT]]
    )
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.READ, _ranges(3))

    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.FAILED,
        TransferState.FAILED,
    ]
    assert events[0].error is None
    assert "FAILED" in events[1].error


def test_transfer_waits_while_ranges_are_pending():
    _install_fake_tent(
        polls=[
            [_StatusEnum.PENDING, _StatusEnum.INITIAL],
            [_StatusEnum.COMPLETED, _StatusEnum.PENDING],
            [_StatusEnum.COMPLETED, _StatusEnum.COMPLETED],
        ]
    )
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.COMPLETED,
    ]


def test_transfer_keeps_range_order_when_a_later_task_settles_first():
    _install_fake_tent(
        polls=[
            [_StatusEnum.PENDING, _StatusEnum.FAILED],
            [_StatusEnum.COMPLETED, _StatusEnum.FAILED],
        ],
        # The aggregated status reports the failure while the first task runs,
        # which is when a reader that appended events as they settled would
        # hand the caller the two outcomes in the wrong order.
        overall=[_StatusEnum.FAILED, _StatusEnum.FAILED],
    )
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.FAILED,
    ]


def test_transfer_reuses_the_peer_segment():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED]])
    backend = _build()

    backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(1))
    backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(1))

    opens = [call for call in backend._engine.calls if call[0] == "open"]
    assert len(opens) == 1


def test_contiguous_ranges_travel_as_one_task():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED]])
    backend = _build()
    ranges = [
        RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=64),
        RangeRequest(local_addr=0x1040, remote_addr=0x2040, length=64),
        RangeRequest(local_addr=0x1080, remote_addr=0x2080, length=64),
    ]

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, ranges)

    assert [event.state for event in events] == [TransferState.COMPLETED] * 3
    submitted = [call for call in backend._engine.calls if call[0] == "submit"][0]
    assert len(submitted[2]) == 1
    assert submitted[2][0].length == 192


def test_ranges_stay_split_when_a_side_is_not_contiguous():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED, _StatusEnum.COMPLETED]])
    backend = _build()
    ranges = [
        RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=64),
        RangeRequest(local_addr=0x1040, remote_addr=0x2400, length=64),
    ]

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, ranges)

    assert [event.state for event in events] == [TransferState.COMPLETED] * 2
    submitted = [call for call in backend._engine.calls if call[0] == "submit"][0]
    assert [request.length for request in submitted[2]] == [64, 64]


def test_register_and_unregister_regions_use_batch_calls():
    _install_fake_tent()
    backend = _build()
    regions = [MemoryRegion(base_addr=0x1000, length=4096)]

    backend.register_regions(regions)
    backend.unregister_regions(regions)

    assert backend._engine.calls == [
        ("register", [0x1000], [4096]),
        ("unregister", [0x1000], [4096]),
    ]


def test_empty_regions_skip_the_engine():
    _install_fake_tent()
    backend = _build()

    backend.register_regions([])
    backend.unregister_regions([])

    assert backend._engine.calls == []


def test_failed_submit_frees_the_batch_and_forgets_the_submission():
    _install_fake_tent(submit_error=RuntimeError("submit refused"))
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(1))

    assert events[0].state is TransferState.FAILED
    assert "submit refused" in events[0].error
    assert backend._batches == {}
    assert backend._engine.calls[-1] == ("free", 1)


def test_unallocatable_batch_reports_the_task_count():
    _install_fake_tent(allocate_fails=True)
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "could not allocate a batch of 2" in events[0].error
    assert backend._batches == {}
    assert ("free", 1) not in backend._engine.calls


def test_shutdown_cancels_outstanding_ranges_and_closes_segments():
    _install_fake_tent(polls=[[_StatusEnum.PENDING], [_StatusEnum.CANCELED]])
    backend = _build()
    backend._submit("10.0.0.2:9100", TransferOp.WRITE, _ranges(1))

    backend.shutdown()

    calls = backend._engine.calls
    assert ("cancel", 1, 0) in calls
    assert ("free", 1) in calls
    assert any(call[0] == "close" for call in calls)


def test_shutdown_reports_a_batch_that_never_settles(monkeypatch):
    _install_fake_tent(polls=[[_StatusEnum.PENDING], [_StatusEnum.PENDING]])
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent."
        "SHUTDOWN_TIMEOUT_S",
        0.05,
    )
    backend = _build()
    backend._submit("10.0.0.2:9100", TransferOp.WRITE, _ranges(1))
    _CONTROL.cancel_error = sys.modules["tent"].NotImplementedError("no cancel")

    with pytest.raises(RuntimeError, match="still holds 1 batches"):
        backend.shutdown()


def test_transfer_after_shutdown_reports_failed_ranges():
    _install_fake_tent()
    backend = _build()
    backend.shutdown()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "shutting down" in events[0].error


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


def test_transfer_reports_failed_ranges_when_the_submission_raises():
    _install_fake_tent(submit_error=RuntimeError("transport down"))
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "transport down" in events[0].error
    assert backend._batches == {}
    assert ("free", 1) in backend._engine.calls


def test_transfer_reports_failed_ranges_when_the_status_query_raises():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED]])
    backend = _build()

    def _explode(batch_id: int) -> None:
        raise RuntimeError("status unavailable")

    backend._engine.get_transfer_status_overall = _explode

    events = backend.transfer("10.0.0.2:9100", TransferOp.WRITE, _ranges(2))

    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "status unavailable" in events[0].error
    # Without a verdict the batch cannot be freed, so shutdown has to settle it.
    assert list(backend._batches) == [1]
    with pytest.raises(RuntimeError, match="could not confirm that 1 batches settled"):
        backend.shutdown()


def test_a_canceled_range_keeps_the_canceled_state():
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED, _StatusEnum.CANCELED]])
    backend = _build()

    events = backend.transfer("10.0.0.2:9100", TransferOp.READ, _ranges(2))

    assert events[0].state is TransferState.COMPLETED
    assert events[1].state is TransferState.CANCELED


def test_contiguous_ranges_merge_up_to_the_merge_limit():
    """A merge stops at the limit, so the range that would overflow starts a task."""
    half = MAX_MERGED_LENGTH // 2
    _install_fake_tent(polls=[[_StatusEnum.COMPLETED, _StatusEnum.COMPLETED]])
    backend = _build()

    events = backend.transfer(
        "10.0.0.2:9100",
        TransferOp.WRITE,
        [
            RangeRequest(local_addr=0, remote_addr=0, length=half),
            RangeRequest(local_addr=half, remote_addr=half, length=half),
            RangeRequest(local_addr=2 * half, remote_addr=2 * half, length=half),
        ],
    )

    submitted = [call for call in backend._engine.calls if call[0] == "submit"][0]
    assert [request.length for request in submitted[2]] == [
        MAX_MERGED_LENGTH,
        half,
    ]
    assert len(events) == 3


def test_backend_advertises_the_hostname_from_the_config(monkeypatch):
    _install_fake_tent(segment_name="peer-alias:9100")
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent."
        "socket.getaddrinfo",
        lambda host, port: [],
    )

    backend = _build(overrides={"rpc_server_hostname": "peer-alias"})

    assert backend.rpc_port == 9100


def test_peer_address_brackets_ipv6():
    assert format_peer_address("2001:db8::1", 9100) == "[2001:db8::1]:9100"
    assert format_peer_address("10.0.0.1", 9100) == "10.0.0.1:9100"
