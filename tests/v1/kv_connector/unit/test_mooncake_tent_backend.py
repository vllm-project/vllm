# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import types
from enum import IntEnum
from pathlib import Path

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


class Status(IntEnum):
    INITIAL = 0
    PENDING = 1
    INVALID = 2
    CANCELED = 3
    COMPLETED = 4
    TIMEOUT = 5
    FAILED = 6


class OpCode(IntEnum):
    READ = 0
    WRITE = 1


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
        self.batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []
        self.statuses: list[int] = []
        self.failure: Exception | None = None
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

    def unregister_local_memory_batch(self, addrs, sizes) -> None:
        self.unregistered.append((len(addrs), len(sizes)))

    def open_segment(self, peer: str) -> Segment:
        if self.failure is not None:
            raise self.failure
        self.opened.append(peer)
        return Segment(peer)

    def close_segment(self, segment: Segment) -> None:
        self.closed.append(segment.peer)

    def transfer_sync_arrays(self, src, dst, lengths, segment, opcode):
        self.batches.append((src, dst, lengths, segment.peer, int(opcode)))
        if self.failure is not None:
            raise self.failure
        return np.asarray(self.statuses, dtype=np.uint8)

    def allocate_transfer_batch(self, count: int) -> int:
        self.batch_size = count
        return 7

    def submit_transfer(self, batch: int, requests) -> None:
        self.submitted = list(requests)

    def wait_transfer(self, batch: int, timeout_ms: int = 0):
        return [
            types.SimpleNamespace(state=Status(int(state)), bytes=0)
            for state in self.statuses
        ]

    def free_transfer_batch(self, batch: int) -> None:
        self.freed = batch


class ListOnlyEngine(Engine):
    transfer_sync_arrays = None


class Request:
    def __init__(self, opcode, source, segment, offset, length) -> None:
        self.opcode = opcode
        self.source = source
        self.segment = segment
        self.offset = offset
        self.length = length


def fake_tent_module(engine_cls: type[Engine] = Engine) -> types.ModuleType:
    module = types.ModuleType("tent")
    module.TransferEngine = engine_cls
    module.Request = Request
    module.OpCode = OpCode
    module.TransferStatusEnum = Status
    return module


@pytest.fixture
def engine(monkeypatch) -> None:
    Engine.instances = []
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
        RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=512),
        RangeRequest(local_addr=0x1200, remote_addr=0x2200, length=512),
    ]


def test_transfer_maps_completed_ranges(engine) -> None:
    backend = make_backend()
    backend._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    events = backend.transfer("127.0.0.1:9000", TransferOp.WRITE, two_ranges())
    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.COMPLETED,
    ]
    src, dst, lengths, peer, opcode = backend._engine.batches[0]
    assert src.tolist() == [0x1000, 0x1200]
    assert dst.tolist() == [0x2000, 0x2200]
    assert lengths.tolist() == [512, 512]
    assert peer == "127.0.0.1:9000"
    assert opcode == int(OpCode.WRITE)


def test_transfer_reports_failed_range(engine) -> None:
    backend = make_backend()
    backend._engine.statuses = [Status.COMPLETED, Status.FAILED]
    events = backend.transfer("127.0.0.1:9000", TransferOp.READ, two_ranges())
    assert events[0].state is TransferState.COMPLETED
    assert events[1].state is TransferState.FAILED
    assert "range 1" in str(events[1].error)
    assert "FAILED" in str(events[1].error)


def test_transfer_sync_arrays_returns_completed(engine) -> None:
    backend = make_backend()
    backend._engine.statuses = [Status.COMPLETED, Status.COMPLETED]
    src = np.array([0x1000, 0x1200], dtype=np.uint64)
    dst = np.array([0x2000, 0x2200], dtype=np.uint64)
    lengths = np.array([512, 512], dtype=np.uint64)
    result = backend.transfer_sync_arrays(
        "127.0.0.1:9000", TransferOp.WRITE, src, dst, lengths
    )
    assert result.num_ranges == 2
    assert result.state is TransferState.COMPLETED
    assert result.error is None


def test_transfer_sync_arrays_counts_failures(engine) -> None:
    backend = make_backend()
    backend._engine.statuses = [Status.FAILED, Status.FAILED, Status.COMPLETED]
    src = np.array([1, 2, 3], dtype=np.uint64)
    result = backend.transfer_sync_arrays(
        "127.0.0.1:9000", TransferOp.WRITE, src, src, src
    )
    assert result.num_ranges == 3
    assert result.state is TransferState.FAILED
    assert "2 of 3 ranges failed" in str(result.error)


def test_transfer_sync_arrays_rejects_mismatched_length(engine) -> None:
    backend = make_backend()
    src = np.array([1, 2], dtype=np.uint64)
    dst = np.array([3], dtype=np.uint64)
    with pytest.raises(RuntimeError, match="per range"):
        backend.transfer_sync_arrays("127.0.0.1:9000", TransferOp.WRITE, src, dst, src)


def test_segment_is_reused_and_dropped_on_error(engine) -> None:
    backend = make_backend()
    engine = backend._engine
    engine.statuses = [Status.COMPLETED]
    single = [RangeRequest(local_addr=0x10, remote_addr=0x20, length=64)]
    backend.transfer("peer-a", TransferOp.WRITE, single)
    backend.transfer("peer-a", TransferOp.WRITE, single)
    assert engine.opened == ["peer-a"]
    engine.failure = RuntimeError("transport reset")
    with pytest.raises(RuntimeError, match="transport reset"):
        backend.transfer("peer-a", TransferOp.WRITE, single)
    assert engine.closed == ["peer-a"]
    engine.failure = None
    backend.transfer("peer-a", TransferOp.WRITE, single)
    assert engine.opened == ["peer-a", "peer-a"]


def test_register_and_unregister_delegate(engine) -> None:
    backend = make_backend()
    regions = [
        MemoryRegion(base_addr=0x1000, length=4096),
        MemoryRegion(base_addr=0x2000, length=8192),
    ]
    backend.register_regions(regions)
    backend.unregister_regions(regions)
    assert backend._engine.registered == [(2, 2)]
    assert backend._engine.unregistered == [(2, 2)]


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
    assert backend._engine.batches == []


def test_unavailable_engine_is_rejected(engine) -> None:
    Engine.next_available = False
    try:
        with pytest.raises(RuntimeError, match="not available"):
            make_backend()
    finally:
        Engine.next_available = True


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


def test_missing_wait_transfer_is_rejected(monkeypatch) -> None:
    class NoWaitEngine(Engine):
        wait_transfer = None

    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(NoWaitEngine))
    with pytest.raises(RuntimeError, match="wait_transfer"):
        make_backend()


def test_falls_back_to_list_submission(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tent", fake_tent_module(ListOnlyEngine))
    backend = make_backend()
    backend._engine.statuses = [int(Status.COMPLETED)] * 2
    result = backend.transfer_sync_arrays(
        "peer",
        TransferOp.WRITE,
        np.asarray([1, 3], dtype=np.uint64),
        np.asarray([2, 4], dtype=np.uint64),
        np.asarray([16, 16], dtype=np.uint64),
    )
    assert result.state is TransferState.COMPLETED
    assert result.num_ranges == 2
    assert backend._engine.batch_size == 2
    assert len(backend._engine.submitted) == 2
    assert backend._engine.freed == 7
