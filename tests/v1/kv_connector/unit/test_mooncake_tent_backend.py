# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import threading
import types
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    MemoryRegion,
    RangeRequest,
    TransferOp,
    TransferState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.factory import (
    create_transfer_backend,
    resolve_transfer_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent import (
    TentTransferBackend,
)


class OpCode(IntEnum):
    READ = 0
    WRITE = 1


class Status(IntEnum):
    INITIAL = 0
    PENDING = 1
    INVALID = 2
    CANCELED = 3
    COMPLETED = 4
    TIMEOUT = 5
    FAILED = 6


class Segment:
    def __init__(self, peer: str) -> None:
        self.peer = peer


class Engine:
    instances: list["Engine"] = []
    next_available = True

    def __init__(self, config_path: str) -> None:
        self.config = json.loads(Path(config_path).read_text())
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[tuple[int, int]] = []
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.requests: list[object] = []
        self.batch_size = 0
        self.freed = 0
        self.overall = Status.PENDING
        self.statuses: list[int] = []
        self.submit_error: Exception | None = None
        self._available = Engine.next_available
        Engine.instances.append(self)

    def available(self) -> bool:
        return self._available

    def get_rpc_server_port(self) -> int:
        return 15012

    def get_segment_name(self) -> str:
        return f"127.0.0.1:{self.get_rpc_server_port()}"

    def register_local_memory_batch(self, addrs, sizes) -> None:
        self.registered.append((len(addrs), len(sizes)))

    def unregister_local_memory_batch(self, addrs, sizes=None) -> None:
        self.unregistered.append((len(addrs), None if sizes is None else len(sizes)))

    def open_segment(self, peer: str) -> Segment:
        self.opened.append(peer)
        return Segment(peer)

    def close_segment(self, segment: Segment) -> None:
        self.closed.append(segment.peer)

    def allocate_transfer_batch(self, count: int) -> int:
        self.batch_size = count
        return 7

    def submit_transfer(self, batch: int, requests) -> None:
        if self.submit_error is not None:
            raise self.submit_error
        self.requests = list(requests)

    def get_transfer_status_overall(self, batch: int):
        return types.SimpleNamespace(state=self.overall, bytes=0)

    def get_transfer_status_list(self, batch: int):
        return [
            types.SimpleNamespace(state=Status(int(state)), bytes=0)
            for state in self.statuses
        ]

    def free_transfer_batch(self, batch: int) -> None:
        self.freed = batch


class Request:
    def __init__(self, opcode, source, segment, offset, length) -> None:
        self.opcode = opcode
        self.source = source
        self.segment = segment
        self.offset = offset
        self.length = length


class MissingStatusEngine(Engine):
    get_transfer_status_list = None


class ArrayEngine(Engine):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.array_calls: list[tuple[int, str, object]] = []

    def transfer_sync_arrays(self, src, dst, lengths, target_id, opcode):
        self.array_calls.append((len(src), target_id.peer, opcode))
        return np.asarray(self.statuses, dtype=np.uint8)


class TimedArrayEngine(Engine):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.array_calls: list[tuple[int, str, object, int]] = []

    def transfer_sync_arrays(
        self, src, dst, lengths, target_id, opcode, timeout_ms: int = 0
    ):
        self.array_calls.append((len(src), target_id.peer, opcode, timeout_ms))
        return np.asarray(self.statuses, dtype=np.uint8)


class WaitEngine(Engine):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.waited: tuple[int, int] | None = None

    def wait_transfer(self, batch, timeout_ms):
        self.waited = (batch, timeout_ms)
        return self.get_transfer_status_list(batch)


class UntimedArrayAndWaitEngine(ArrayEngine, WaitEngine):
    pass


def fake_tent_module(engine_cls: type[Engine] = Engine) -> Any:
    module: Any = types.ModuleType("tent")
    module.TransferEngine = engine_cls
    module.Request = Request
    module.OpCode = OpCode
    module.TransferStatusEnum = Status
    return module


@pytest.fixture
def engine(monkeypatch) -> None:
    Engine.instances = []
    Engine.next_available = True
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module())


def make_backend(timeout: float = 5.0) -> TentTransferBackend:
    return TentTransferBackend(
        hostname="127.0.0.1",
        protocol="tcp",
        device_name="",
        transfer_timeout_s=timeout,
    )


def two_ranges() -> list[RangeRequest]:
    return [
        RangeRequest(local_addr=1000, remote_addr=2000, length=16),
        RangeRequest(local_addr=1016, remote_addr=2016, length=16),
    ]


def test_missing_status_binding_is_rejected(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(MissingStatusEngine))
    with pytest.raises(RuntimeError, match="get_transfer_status_list"):
        make_backend()


def test_unavailable_engine_is_rejected(engine) -> None:
    Engine.next_available = False
    with pytest.raises(RuntimeError, match="not available"):
        make_backend()


def test_transfer_reports_completed_ranges(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.COMPLETED
    backend._engine.statuses = [Status.COMPLETED] * 2
    events = backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [TransferState.COMPLETED] * 2
    assert backend._engine.batch_size == 2
    assert len(backend._engine.requests) == 2
    assert backend._engine.freed == 7


def test_transfer_reports_failed_ranges(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.FAILED
    backend._engine.statuses = [Status.COMPLETED, Status.FAILED]
    events = backend.transfer("peer", TransferOp.READ, two_ranges())
    assert events[0].state is TransferState.COMPLETED
    assert events[1].state is TransferState.FAILED
    assert "FAILED" in (events[1].error or "")


def test_empty_ranges_skip_the_engine(engine) -> None:
    backend = make_backend()
    assert backend.transfer("peer", TransferOp.WRITE, []) == []
    result = backend.transfer_sync_arrays(
        "peer",
        TransferOp.WRITE,
        np.empty(0, dtype=np.uint64),
        np.empty(0, dtype=np.uint64),
        np.empty(0, dtype=np.uint64),
    )
    assert result.state is TransferState.COMPLETED
    assert result.num_ranges == 0
    assert backend._engine.batch_size == 0


def test_transfer_sync_arrays_reports_overall_result(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.COMPLETED
    backend._engine.statuses = [Status.COMPLETED] * 2
    result = backend.transfer_sync_arrays(
        "peer",
        TransferOp.WRITE,
        np.asarray([1000, 1016], dtype=np.uint64),
        np.asarray([2000, 2016], dtype=np.uint64),
        np.asarray([16, 16], dtype=np.uint64),
    )
    assert result.state is TransferState.COMPLETED
    assert result.num_ranges == 2


def test_transfer_sync_arrays_reports_failures(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.FAILED
    backend._engine.statuses = [Status.TIMEOUT, Status.FAILED]
    result = backend.transfer_sync_arrays(
        "peer",
        TransferOp.WRITE,
        np.asarray([1000, 1016], dtype=np.uint64),
        np.asarray([2000, 2016], dtype=np.uint64),
        np.asarray([16, 16], dtype=np.uint64),
    )
    assert result.state is TransferState.FAILED
    assert "TIMEOUT" in (result.error or "")
    assert "2 of 2" in (result.error or "")


def test_transfer_sync_arrays_rejects_mismatched_lengths(engine) -> None:
    backend = make_backend()
    with pytest.raises(ValueError, match="one local address"):
        backend.transfer_sync_arrays(
            "peer",
            TransferOp.WRITE,
            np.asarray([1000, 1016], dtype=np.uint64),
            np.asarray([2000], dtype=np.uint64),
            np.asarray([16, 16], dtype=np.uint64),
        )


def test_transfer_sync_arrays_rejects_invalid_columns(engine) -> None:
    backend = make_backend()
    valid = np.asarray([1000, 1016], dtype=np.uint64)
    with pytest.raises(ValueError, match="contiguous 1D native uint64"):
        backend.transfer_sync_arrays(
            "peer",
            TransferOp.WRITE,
            np.asarray([[1000, 1016]], dtype=np.uint64),
            valid,
            valid,
        )
    with pytest.raises(ValueError, match="contiguous 1D native uint64"):
        backend.transfer_sync_arrays(
            "peer",
            TransferOp.WRITE,
            np.asarray([1000, 1016], dtype=np.int64),
            valid,
            valid,
        )
    strided = np.asarray([1000, 0, 1016, 0], dtype=np.uint64)[::2]
    with pytest.raises(ValueError, match="contiguous 1D native uint64"):
        backend.transfer_sync_arrays("peer", TransferOp.WRITE, strided, valid, valid)


def test_segment_is_reused(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.COMPLETED
    backend._engine.statuses = [Status.COMPLETED] * 2
    backend.transfer("peer", TransferOp.WRITE, two_ranges())
    backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert backend._engine.opened == ["peer"]


def test_segment_survives_submit_error(engine) -> None:
    backend = make_backend()
    backend._engine.submit_error = RuntimeError("submit failed")
    with pytest.raises(RuntimeError, match="submit failed"):
        backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert backend._engine.closed == []


def test_timed_arrays_binding_skips_batch_lifecycle(engine, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(TimedArrayEngine))
    backend = make_backend(timeout=5.0)
    backend._engine.statuses = [Status.COMPLETED, Status.FAILED]
    events = backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.FAILED,
    ]
    assert backend._engine.array_calls == [(2, "peer", OpCode.WRITE, 5000)]
    assert backend._engine.freed == 0


def test_untimed_arrays_binding_requires_zero_timeout(engine, monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "tent", fake_tent_module(UntimedArrayAndWaitEngine)
    )
    bounded = make_backend(timeout=5.0)
    bounded._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    events = bounded.transfer("peer", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [TransferState.COMPLETED] * 2
    assert bounded._engine.array_calls == []
    assert bounded._engine.waited == (7, 5000)

    unbounded = make_backend(timeout=0)
    unbounded._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    events = unbounded.transfer("peer", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [TransferState.COMPLETED] * 2
    assert unbounded._engine.array_calls == [(2, "peer", OpCode.WRITE)]
    assert unbounded._engine.freed == 0


def test_sub_millisecond_timeout_rounds_up_and_invalid_timeouts_fail(
    engine, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(WaitEngine))
    backend = make_backend(timeout=0.0004)
    backend._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert backend._engine.waited == (7, 1)

    for bad_timeout in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(RuntimeError, match="finite non-negative"):
            make_backend(timeout=bad_timeout)


def test_request_error_still_frees_allocated_batch(monkeypatch) -> None:
    module = fake_tent_module()

    def broken_request(*_args):
        raise ValueError("bad request")

    module.Request = broken_request
    monkeypatch.setitem(sys.modules, "tent", module)
    backend = make_backend()
    with pytest.raises(ValueError, match="bad request"):
        backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert backend._engine.freed == 7


def test_concurrent_transfers_to_same_peer_overlap(monkeypatch) -> None:
    barrier = threading.Barrier(2)

    class BarrierWaitEngine(WaitEngine):
        def wait_transfer(self, batch, timeout_ms):
            barrier.wait(timeout=2.0)
            return [types.SimpleNamespace(state=Status.COMPLETED, bytes=0)] * 2

    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(BarrierWaitEngine))
    backend = make_backend()
    results: list[list[TransferState]] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            events = backend.transfer("peer", TransferOp.WRITE, two_ranges())
            results.append([event.state for event in events])
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    assert errors == []
    assert results == [[TransferState.COMPLETED] * 2] * 2
    assert backend._engine.opened == ["peer"]


def test_shutdown_drains_inflight_before_closing_segments(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingWaitEngine(WaitEngine):
        def wait_transfer(self, batch, timeout_ms):
            entered.set()
            assert release.wait(timeout=2.0)
            return [types.SimpleNamespace(state=Status.COMPLETED, bytes=0)] * 2

    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(BlockingWaitEngine))
    backend = make_backend()
    transfer_thread = threading.Thread(
        target=backend.transfer, args=("peer", TransferOp.WRITE, two_ranges())
    )
    transfer_thread.start()
    assert entered.wait(timeout=2.0)

    shutdown_thread = threading.Thread(target=backend.shutdown)
    shutdown_thread.start()
    shutdown_thread.join(timeout=0.05)
    assert shutdown_thread.is_alive()
    assert backend._engine.closed == []

    release.set()
    transfer_thread.join(timeout=2.0)
    shutdown_thread.join(timeout=2.0)
    assert not shutdown_thread.is_alive()
    assert backend._engine.closed == ["peer"]


def test_shutdown_skips_closing_segments_when_drain_times_out(monkeypatch) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer import tent

    monkeypatch.setattr(tent, "SHUTDOWN_TIMEOUT_S", 0.02)
    entered = threading.Event()
    release = threading.Event()

    class BlockingWaitEngine(WaitEngine):
        def wait_transfer(self, batch, timeout_ms):
            entered.set()
            release.wait(timeout=2.0)
            return [types.SimpleNamespace(state=Status.COMPLETED, bytes=0)] * 2

    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(BlockingWaitEngine))
    backend = make_backend()
    transfer_thread = threading.Thread(
        target=backend.transfer, args=("peer", TransferOp.WRITE, two_ranges())
    )
    transfer_thread.start()
    assert entered.wait(timeout=2.0)

    try:
        backend.shutdown()
        assert backend._engine.closed == []
    finally:
        release.set()
        transfer_thread.join(timeout=2.0)


def test_wait_transfer_replaces_polling(engine, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(WaitEngine))
    backend = make_backend()
    backend._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    backend._engine.overall = Status.PENDING
    events = backend.transfer("peer", TransferOp.READ, two_ranges())
    assert [event.state for event in events] == [TransferState.COMPLETED] * 2
    assert backend._engine.waited == (7, 5000)


def test_transfers_are_rejected_after_shutdown(engine) -> None:
    backend = make_backend()
    backend.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        backend.transfer("peer", TransferOp.WRITE, two_ranges())


def test_nvlink_follows_the_rdma_protocol(engine) -> None:
    rdma = TentTransferBackend(
        hostname="127.0.0.1",
        protocol="rdma",
        device_name="",
        transfer_timeout_s=5.0,
    )
    assert "nvlink" not in rdma._engine.config["transports"]
    tcp = make_backend()
    assert tcp._engine.config["transports"]["nvlink"] == {"enable": False}
    sleeping = TentTransferBackend(
        hostname="127.0.0.1",
        protocol="rdma",
        device_name="",
        enable_sleep_mode=True,
    )
    assert sleeping._engine.config["transports"]["nvlink"] == {"enable": False}


def test_short_status_list_fails_every_range(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(TimedArrayEngine))
    backend = make_backend()
    backend._engine.statuses = [Status.COMPLETED]
    events = backend.transfer("peer", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [TransferState.FAILED] * 2
    assert "terminal states" in events[0].error


def test_register_and_unregister_delegate(engine) -> None:
    backend = make_backend()
    backend.register_regions([MemoryRegion(base_addr=1, length=64)] * 2)
    backend.unregister_regions([MemoryRegion(base_addr=1, length=64)])
    assert backend._engine.registered == [(2, 2)]
    assert backend._engine.unregistered == [(1, None)]


def test_timeout_stops_polling(engine) -> None:
    backend = make_backend(timeout=0.01)
    backend._engine.overall = Status.PENDING
    backend._engine.statuses = [Status.PENDING]
    events = backend.transfer(
        "peer",
        TransferOp.WRITE,
        [RangeRequest(local_addr=1, remote_addr=2, length=16)],
    )
    assert events[0].state is TransferState.FAILED
    assert "PENDING" in (events[0].error or "")


def test_shutdown_closes_open_segments(engine) -> None:
    backend = make_backend()
    backend._engine.overall = Status.COMPLETED
    backend._engine.statuses = [Status.COMPLETED]
    backend.transfer(
        "peer",
        TransferOp.WRITE,
        [RangeRequest(local_addr=1, remote_addr=2, length=16)],
    )
    backend.shutdown()
    assert backend._engine.closed == ["peer"]


def test_factory_defaults_to_te(engine) -> None:
    assert resolve_transfer_engine({}, environ={}) == "te"
    assert resolve_transfer_engine({"transfer_engine": "tent"}, environ={}) == "tent"
    assert resolve_transfer_engine({}, environ={"MC_USE_TENT": "1"}) == "tent"
    with pytest.raises(RuntimeError, match="transfer_engine"):
        resolve_transfer_engine({"transfer_engine": "te"}, environ={"MC_USE_TENT": "1"})
    with pytest.raises(RuntimeError, match="transfer_engine"):
        resolve_transfer_engine({"transfer_engine": "bogus"})


def test_factory_builds_tent_backend(engine) -> None:
    backend = create_transfer_backend(
        "tent",
        engine_cls=None,
        hostname="127.0.0.1",
        protocol="tcp",
        device_name="",
    )
    assert isinstance(backend, TentTransferBackend)
    assert backend.rpc_port == 15012
