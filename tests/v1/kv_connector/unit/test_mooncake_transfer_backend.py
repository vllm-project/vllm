# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
    MemoryRegion,
    RangeRequest,
    TransferEvent,
    TransferOp,
    TransferState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.te import (
    TransferEngineBackend,
)


class FakeEngine:
    def __init__(
        self,
        *,
        write_status: int = 0,
        read_status: int = 0,
        register_status: int = 0,
        unregister_status: int = 0,
    ) -> None:
        self.write_status = write_status
        self.read_status = read_status
        self.register_status = register_status
        self.unregister_status = unregister_status
        self.calls: list[tuple] = []

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        self.calls.append(
            ("initialize", local_hostname, metadata_server, protocol, device_name)
        )
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(self, session, src_ptrs, dst_ptrs, lengths) -> int:
        self.calls.append(
            ("write", session, list(src_ptrs), list(dst_ptrs), list(lengths))
        )
        return self.write_status

    def batch_transfer_sync_read(self, session, src_ptrs, dst_ptrs, lengths) -> int:
        self.calls.append(
            ("read", session, list(src_ptrs), list(dst_ptrs), list(lengths))
        )
        return self.read_status

    def batch_register_memory(self, base_addrs, lengths) -> int:
        self.calls.append(("register", list(base_addrs), list(lengths)))
        return self.register_status

    def batch_unregister_memory(self, base_addrs) -> int:
        self.calls.append(("unregister", list(base_addrs)))
        return self.unregister_status


def make_backend(engine: FakeEngine) -> TransferEngineBackend:
    return TransferEngineBackend(
        engine_cls=lambda: engine,
        hostname="10.0.0.1",
        protocol="rdma",
        device_name="mlx5_0",
    )


def test_backend_initializes_engine_with_p2p_handshake():
    engine = FakeEngine()
    backend = make_backend(engine)

    assert engine.calls[0] == (
        "initialize",
        "10.0.0.1",
        "P2PHANDSHAKE",
        "rdma",
        "mlx5_0",
    )
    assert backend.rpc_port == 12345


def test_transfer_reports_completion_per_range():
    engine = FakeEngine()
    backend = make_backend(engine)

    events = backend.transfer(
        "10.0.0.2:1234",
        TransferOp.WRITE,
        [
            RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=256),
            RangeRequest(local_addr=0x1100, remote_addr=0x2100, length=256),
        ],
    )

    assert engine.calls[-1] == (
        "write",
        "10.0.0.2:1234",
        [0x1000, 0x1100],
        [0x2000, 0x2100],
        [256, 256],
    )
    assert [event.state for event in events] == [
        TransferState.COMPLETED,
        TransferState.COMPLETED,
    ]
    assert [event.error for event in events] == [None, None]


def test_transfer_reports_engine_failure_for_every_range():
    engine = FakeEngine(write_status=7)
    backend = make_backend(engine)

    events = backend.transfer(
        "10.0.0.2:1234",
        TransferOp.WRITE,
        [RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=64)],
    )

    assert len(events) == 1
    assert events[0].state is TransferState.FAILED
    assert events[0].error == "Mooncake transfer engine returned 7"


def test_transfer_reads_when_op_is_read():
    engine = FakeEngine(read_status=5)
    backend = make_backend(engine)

    events = backend.transfer(
        "10.0.0.2:1234",
        TransferOp.READ,
        [RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=8)],
    )

    assert engine.calls[-1][0] == "read"
    assert events[0].state is TransferState.FAILED


def test_transfer_returns_nothing_for_an_empty_request():
    engine = FakeEngine()
    backend = make_backend(engine)

    assert backend.transfer("10.0.0.2:1234", TransferOp.WRITE, []) == []
    assert engine.calls[-1][0] == "initialize"


def test_register_regions_forwards_addresses_and_lengths():
    engine = FakeEngine()
    backend = make_backend(engine)

    backend.register_regions([MemoryRegion(base_addr=0x1000, length=4096)])
    backend.unregister_regions([MemoryRegion(base_addr=0x1000, length=4096)])

    assert engine.calls[-2:] == [
        ("register", [0x1000], [4096]),
        ("unregister", [0x1000]),
    ]


def test_register_regions_raises_when_engine_rejects_memory():
    engine = FakeEngine(register_status=2)
    backend = make_backend(engine)

    with pytest.raises(RuntimeError, match="registration failed"):
        backend.register_regions([MemoryRegion(base_addr=0x1000, length=4096)])


def test_unregister_regions_raises_when_engine_rejects_memory():
    engine = FakeEngine(unregister_status=2)
    backend = make_backend(engine)

    with pytest.raises(RuntimeError, match="unregistration failed"):
        backend.unregister_regions([MemoryRegion(base_addr=0x1000, length=4096)])


def test_register_regions_skips_empty_input():
    engine = FakeEngine()
    backend = make_backend(engine)

    backend.register_regions([])
    backend.unregister_regions([])

    assert engine.calls[-1][0] == "initialize"


def test_backend_rejects_failed_initialize():
    class FailingEngine(FakeEngine):
        def initialize(self, local_hostname, metadata_server, protocol, device_name):
            return 1

    with pytest.raises(RuntimeError, match="initialization failed"):
        make_backend(FailingEngine())


def test_concurrent_transfers_keep_their_own_events():
    engine = FakeEngine()
    backend = make_backend(engine)
    results: dict[str, list[TransferEvent]] = {}
    submitted = threading.Barrier(2)

    def submit_and_collect(name: str, length: int) -> None:
        submitted.wait()
        results[name] = backend.transfer(
            "10.0.0.2:1234",
            TransferOp.WRITE,
            [RangeRequest(local_addr=0x1000, remote_addr=0x2000, length=length)],
        )

    threads = [
        threading.Thread(target=submit_and_collect, args=("first", 16)),
        threading.Thread(target=submit_and_collect, args=("second", 32)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [event.state for event in results["first"]] == [TransferState.COMPLETED]
    assert [event.state for event in results["second"]] == [TransferState.COMPLETED]
    lengths = sorted(call[4][0] for call in engine.calls if call[0] == "write")
    assert lengths == [16, 32]
