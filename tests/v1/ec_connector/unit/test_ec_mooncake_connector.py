# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral contract for the refactored Mooncake encoder-cache connector.

The suite covers public compatibility, configuration, control and data planes,
memory ownership, the three role-specific lifecycle managers, Scheduler
metadata, Worker orchestration, and failure or cancellation races.
"""

from __future__ import annotations

import copy
import ctypes
import gc
import importlib
import socket
import sys
import threading
import time
import weakref
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from multiprocessing.reduction import ForkingPickler
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
import zmq

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector import mooncake_ec_connector
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.mooncake import (
    control,
    memory,
    metadata,
    producer,
    state,
    transfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import MooncakeECConfig
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
    ControlCompletion,
    EventInbox,
    ShardTopology,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ContiguousAllocator,
    ProducerMemoryPool,
    ResidentPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    CancellationOutcome,
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    InvalidSchedulerTransferTransition,
    SchedulerTransferState,
    SchedulerTransferTable,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    _LEASE_TTL_SECONDS,
    ECMooncakeWorker,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput

pytest_plugins = ("tests.v1.ec_connector.unit.test_ec_example_connector",)


class CopyingFakeTransferEngine:
    """Model Mooncake registration rules while copying bytes in-process.

    Attributes:
        registered: Base addresses of currently registered ranges.
        regions: Registered byte lengths keyed by base address.
        register_calls: Address batches passed to memory registration.
        unregister_calls: Addresses passed to single-range unregistration.
        batch_unregister_calls: Address batches passed to unregistration.
        transfer_calls: Byte lengths recorded for each transfer batch.
        transfer_batches: Complete source and destination transfer arguments.
        initialize_calls: Arguments used to initialize the fake engine.
    """

    def __init__(self, *args, **kwargs):
        self.registered: set[int] = set()
        self.regions: dict[int, int] = {}
        self.register_calls: list[list[int]] = []
        self.unregister_calls: list[int] = []
        self.batch_unregister_calls: list[list[int]] = []
        self.transfer_calls: list[list[int]] = []
        self.transfer_batches: list[tuple[str, list[int], list[int], list[int]]] = []
        self.initialize_calls: list[tuple[str, str, str, str]] = []

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        self.initialize_calls.append(
            (local_hostname, metadata_server, protocol, device_name)
        )
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        sources = [int(address) for address in buffers]
        destinations = [int(address) for address in peer_buffer_addresses]
        sizes = [int(length) for length in lengths]
        self.transfer_calls.append(sizes)
        self.transfer_batches.append(
            (str(target_hostname), sources, destinations, sizes)
        )
        for src, dst, nbytes in zip(sources, destinations, sizes):
            ctypes.memmove(int(dst), int(src), int(nbytes))
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        addresses = [int(addr) for addr in buffer_addresses]
        lengths = [int(length) for length in capacities]
        # A real Transfer Engine refuses overlapping memory regions, so model
        # that here: registering a range that intersects a live one fails.
        regions = dict(self.regions)
        for address, length in zip(addresses, lengths):
            for other, other_length in regions.items():
                if address < other + other_length and other < address + length:
                    return 1
            regions[address] = length
        self.register_calls.append(addresses)
        self.registered.update(addresses)
        self.regions = regions
        return 0

    def unregister_memory(self, buffer_address) -> int:
        address = int(buffer_address)
        self.unregister_calls.append(address)
        self.registered.discard(address)
        self.regions.pop(address, None)
        return 0

    def batch_unregister_memory(self, buffer_addresses) -> int:
        addresses = [int(addr) for addr in buffer_addresses]
        self.batch_unregister_calls.append(addresses)
        self.registered.difference_update(addresses)
        for address in addresses:
            self.regions.pop(address, None)
        return 0


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _wait_for_worker_io(
    connector: ECMooncakeConnector, timeout: float = 5.0
) -> ECMooncakeWorkerMetadata:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        meta = connector.build_connector_worker_meta()
        assert isinstance(meta, ECMooncakeWorkerMetadata)
        if not meta.pending_loads and not meta.pending_saves:
            return meta
        time.sleep(0.01)
    raise TimeoutError("EC Mooncake worker I/O did not finish")


class TestECMooncakeControlPlane:
    """Validate ZMQ client reuse, shard discovery, events, and server RPCs."""

    def test_worker_get_ip_failure_does_not_construct_client(
        self, mock_vllm_config_producer
    ):
        with (
            patch_ec_mooncake_deps(),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake.worker.get_ip",
                side_effect=RuntimeError("no address"),
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "worker.ControlClient"
            ) as client_cls,
            pytest.raises(RuntimeError, match="no address"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, ECConnectorRole.WORKER)

        client_cls.assert_not_called()

    def test_worker_constructor_failure_closes_client(self, mock_vllm_config_producer):
        with (
            patch_ec_mooncake_deps(),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "worker.ControlClient"
            ) as client_cls,
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "worker.ThreadPoolExecutor",
                side_effect=RuntimeError("executor failed"),
            ),
            pytest.raises(RuntimeError, match="executor failed"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, ECConnectorRole.WORKER)

        client_cls.return_value.close.assert_called_once_with()

    def test_client_reuses_socket_and_discards_failed_exchange(self):
        context = MagicMock()
        socket = context.socket.return_value
        socket.recv_json.side_effect = [
            {"ok": True, "result": {"ports": [19019]}},
            {"ok": True},
            {"ok": False, "error": "reservation rejected"},
            RuntimeError("timeout"),
        ]

        with patch.object(control.zmq, "Context", return_value=context):
            client = ControlClient(17)
            assert client.request("tcp://consumer:19019", {"op": "peers"}) == {
                "ports": [19019]
            }
            assert client.request("tcp://consumer:19019", {"op": "event_port"}) is None
            with pytest.raises(RuntimeError, match="reservation rejected"):
                client.request(
                    "tcp://consumer:19019",
                    {"op": "status", "transfer_id": "transfer"},
                )
            with pytest.raises(RuntimeError, match="timeout"):
                client.request("tcp://consumer:19019", {"op": "event_port"})
            client.close()
            client.close()

        context.socket.assert_called_once_with(zmq.REQ)
        assert socket.setsockopt.call_args_list == [
            call(zmq.RCVTIMEO, 17),
            call(zmq.SNDTIMEO, 17),
            call(zmq.LINGER, 0),
        ]
        socket.connect.assert_called_once_with("tcp://consumer:19019")
        socket.close.assert_called_once_with(linger=0)
        context.destroy.assert_called_once_with(linger=0)

    def test_client_uses_one_socket_per_thread(self):
        context = MagicMock()
        sockets = [MagicMock(), MagicMock()]
        for index, control_socket in enumerate(sockets):
            control_socket.recv_json.return_value = {"ok": True, "result": index}
        context.socket.side_effect = sockets
        barrier = threading.Barrier(2)
        results: list[int] = []

        with patch.object(control.zmq, "Context", return_value=context):
            client = ControlClient(20)

            def request() -> None:
                barrier.wait()
                results.append(client.request("tcp://consumer:19019", {"op": "peers"}))

            threads = [threading.Thread(target=request) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            client.close()

        assert sorted(results) == [0, 1]
        assert context.socket.call_args_list == [call(zmq.REQ), call(zmq.REQ)]
        for control_socket in sockets:
            control_socket.connect.assert_called_once_with("tcp://consumer:19019")

    def test_topology_retries_transient_discovery_failures(self):
        client = Mock(spec=ControlClient)
        client.request.side_effect = [
            {"ports": [19019, 19020]},
            RuntimeError("old consumer"),
            {"ports": [19029, 19030]},
        ]
        topology = ShardTopology(client)

        assert topology.shards("tcp://consumer:19019") == [
            "tcp://consumer:19019",
            "tcp://consumer:19020",
        ]
        assert topology.shards("tcp://consumer:19019") == [
            "tcp://consumer:19019",
            "tcp://consumer:19020",
        ]
        assert topology.shards("tcp://legacy:19019") == ["tcp://legacy:19019"]
        assert topology.shards("tcp://legacy:19019") == [
            "tcp://legacy:19029",
            "tcp://legacy:19030",
        ]
        assert topology.shards("tcp://legacy:19019") == [
            "tcp://legacy:19029",
            "tcp://legacy:19030",
        ]
        assert client.request.call_args_list == [
            call("tcp://consumer:19019", {"op": "peers"}),
            call("tcp://legacy:19019", {"op": "peers"}),
            call("tcp://legacy:19019", {"op": "peers"}),
        ]

    def test_event_inbox_retries_until_every_shard_is_connected(self):
        client = Mock(spec=ControlClient)
        client.request.side_effect = [
            RuntimeError("peers not ready"),
            {"ports": [19019, 19020]},
            20001,
            RuntimeError("event port not ready"),
            20001,
            20002,
        ]
        topology = ShardTopology(client)
        context = MagicMock()
        socket = context.socket.return_value
        event = {"transfer_id": "transfer", "ready": True}
        socket.recv_json.side_effect = [event, zmq.Again()]

        with patch.object(control.zmq, "Context", return_value=context) as create:
            inbox = EventInbox(client, topology)
            assert inbox.drain("tcp://consumer:19019") == []
            assert inbox.shard_count == 1
            create.assert_not_called()
            assert inbox.drain("tcp://consumer:19019") == []
            assert inbox.shard_count == 1
            create.assert_not_called()
            assert inbox.drain("tcp://consumer:19019") == [event]
            assert inbox.shard_count == 2
            inbox.close()
            inbox.close()

        assert client.request.call_args_list == [
            call("tcp://consumer:19019", {"op": "peers"}),
            call("tcp://consumer:19019", {"op": "peers"}),
            call("tcp://consumer:19019", {"op": "event_port"}),
            call("tcp://consumer:19020", {"op": "event_port"}),
            call("tcp://consumer:19019", {"op": "event_port"}),
            call("tcp://consumer:19020", {"op": "event_port"}),
        ]
        create.assert_called_once_with()
        assert socket.connect.call_args_list == [
            call("tcp://consumer:20001"),
            call("tcp://consumer:20002"),
        ]
        assert socket.recv_json.call_args_list == [
            call(flags=zmq.DONTWAIT),
            call(flags=zmq.DONTWAIT),
        ]
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    def test_server_preserves_wire_shapes_and_closes_twice(self):
        port = _find_free_port()
        completed: list[tuple[str, str]] = []
        cancelled: list[tuple[str, str, bool, bool]] = []

        def status(transfer_id: str):
            return {"transfer_id": transfer_id, "ready": False}

        def complete(transfer_id: str, reservation_id: str):
            completed.append((transfer_id, reservation_id))
            return ControlCompletion(True, became_ready=True)

        def cancel(
            transfer_id: str,
            reservation_id: str,
            abandon: bool,
            refresh: bool,
        ):
            cancelled.append((transfer_id, reservation_id, abandon, refresh))
            return True

        server = ConsumerControlServer(
            "127.0.0.1",
            port,
            reserve=lambda request: {"nbytes": request["nbytes"], "ready": False},
            status=status,
            complete=complete,
            cancel=cancel,
            reap=lambda: 0,
            peer_ports=[port, port + 1],
        )
        client = ControlClient(1000)
        server.start()
        try:
            addr = f"tcp://127.0.0.1:{port}"
            assert client.request(addr, {"op": "peers"}) == {"ports": [port, port + 1]}
            assert isinstance(client.request(addr, {"op": "event_port"}), int)
            assert client.request(
                addr, {"op": "status", "transfer_id": "transfer"}
            ) == {"transfer_id": "transfer", "ready": False}
            assert client.request(
                addr,
                {
                    "op": "reserve",
                    "transfer_id": "transfer",
                    "mm_hash": "hash",
                    "nbytes": 16,
                    "shape": [4],
                    "dtype": "float32",
                },
            ) == {"nbytes": 16, "ready": False}
            assert client.request(
                addr,
                {
                    "op": "complete",
                    "transfer_id": "transfer",
                    "reservation_id": "r0",
                },
            ) == {"completed": True, "became_ready": True}
            assert client.request(
                addr,
                {
                    "op": "complete_batch",
                    "items": [{"transfer_id": "transfer", "reservation_id": "r0"}],
                },
            ) == {"items": [{"completed": True, "became_ready": True}]}
            assert client.request(
                addr,
                {
                    "op": "cancel",
                    "transfer_id": "transfer",
                    "reservation_id": "r0",
                    "abandon": True,
                },
            ) == {"cancelled": True}
        finally:
            client.close()
            client.close()
            server.close()
            server.close()

        assert completed == [("transfer", "r0"), ("transfer", "r0")]
        assert cancelled == [("transfer", "r0", True, False)]


@pytest.fixture
def mock_vllm_config_producer():
    config = Mock(spec=VllmConfig)
    config.model_config = Mock(spec=ModelConfig)
    config.model_config.dtype = torch.float16
    config.model_config.hf_config = None
    config.model_config.get_inputs_embeds_size.return_value = 16
    config.parallel_config = Mock()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.data_parallel_index = 0
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = True
    config.ec_transfer_config.is_ec_consumer = False
    config.ec_transfer_config.ec_buffer_device = "cuda"
    config.ec_transfer_config.ec_buffer_size = 1e9
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
    }
    return config


@pytest.fixture
def mock_vllm_config_consumer():
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.data_parallel_index = 0
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = False
    config.ec_transfer_config.is_ec_consumer = True
    config.ec_transfer_config.ec_buffer_device = "cuda"
    config.ec_transfer_config.ec_buffer_size = 1e9
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
        "reservation_zmq_port": 19019,
    }
    return config


@contextmanager
def patch_ec_mooncake_deps():
    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake.transfer.TransferEngine",
            CopyingFakeTransferEngine,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake."
            "_availability._MOONCAKE_IMPORT_ERROR",
            None,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake.worker.get_ip",
            return_value="127.0.0.1",
        ),
    ):
        yield


class TestMooncakeTransfer:
    """Validate lazy engine setup and source registration ownership."""

    def test_initializes_engine_once_on_first_use(self):
        engine = CopyingFakeTransferEngine()
        with patch.object(
            transfer, "TransferEngine", return_value=engine
        ) as engine_cls:
            data_plane = MooncakeTransfer("host", "tcp")
            assert engine_cls.call_count == 0

            assert data_plane.local_session() == "host:12345"
            data_plane.ensure_ready()

            engine_cls.assert_called_once_with()
            assert engine.initialize_calls == [("host", "P2PHANDSHAKE", "tcp", "")]
            data_plane.close()

    def test_source_registration_is_refcounted_and_failed_release_keeps_owner(self):
        engine = CopyingFakeTransferEngine()
        data_plane = MooncakeTransfer("host", "tcp")
        source = torch.randn(4, 4)
        source_ref = weakref.ref(source)
        with patch.object(transfer, "TransferEngine", return_value=engine):
            first = data_plane.acquire_sources([source])
            second = data_plane.acquire_sources([source])
            assert engine.register_calls == [[source.data_ptr()]]

            assert data_plane.release_sources(first)
            assert engine.batch_unregister_calls == []
            with patch.object(
                engine, "batch_unregister_memory", return_value=1
            ) as unregister:
                assert not data_plane.release_sources(second)
                unregister.assert_called_once_with(second)

            del source
            gc.collect()
            assert source_ref() is not None
            with patch.object(
                engine, "batch_unregister_memory", return_value=0
            ) as unregister:
                data_plane.close()
                data_plane.close()
                unregister.assert_called_once_with(second)
            gc.collect()
            assert source_ref() is None

    def test_failed_destination_unregister_is_retried_on_close(self):
        engine = CopyingFakeTransferEngine()
        data_plane = MooncakeTransfer("host", "tcp")
        destination = torch.zeros(4, 4)
        destination_ref = weakref.ref(destination)
        with patch.object(transfer, "TransferEngine", return_value=engine):
            assert data_plane.register_memory(destination) == 0
            with patch.object(engine, "unregister_memory", return_value=2):
                assert not data_plane.unregister_memory(destination)
            del destination
            gc.collect()
            assert destination_ref() is not None

            with patch.object(
                engine, "batch_unregister_memory", return_value=0
            ) as unregister:
                data_plane.close()
                unregister.assert_called_once()
            gc.collect()
            assert destination_ref() is None

    def test_write_preserves_segments_and_reports_terminal_failure(self):
        engine = CopyingFakeTransferEngine()
        data_plane = MooncakeTransfer("host", "tcp")
        sources = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        destinations = [torch.zeros_like(source) for source in sources]
        source_addresses = [source.data_ptr() for source in sources]
        destination_addresses = [tensor.data_ptr() for tensor in destinations]
        lengths = [source.nbytes for source in sources]
        with patch.object(transfer, "TransferEngine", return_value=engine):
            data_plane.write("peer:1", source_addresses, destination_addresses, lengths)
            assert engine.transfer_batches == [
                ("peer:1", source_addresses, destination_addresses, lengths)
            ]
            assert all(
                torch.equal(source, destination)
                for source, destination in zip(sources, destinations)
            )

            with (
                patch.object(engine, "batch_transfer_sync_write", return_value=9),
                pytest.raises(RuntimeError, match="peer:2 failed with status 9"),
            ):
                data_plane.write(
                    "peer:2", source_addresses, destination_addresses, lengths
                )
            data_plane.close()

    def test_write_returns_only_after_sync_engine_call_finishes(self):
        engine = CopyingFakeTransferEngine()
        data_plane = MooncakeTransfer("host", "tcp")
        source = torch.ones(1, dtype=torch.uint8)
        entered = threading.Event()
        finish = threading.Event()

        def blocking_write(*args):
            entered.set()
            assert finish.wait(timeout=2)
            return 0

        with (
            patch.object(transfer, "TransferEngine", return_value=engine),
            patch.object(
                engine, "batch_transfer_sync_write", side_effect=blocking_write
            ),
        ):
            addresses = data_plane.acquire_sources([source])
            completed = threading.Event()

            def write():
                try:
                    data_plane.write("peer:1", addresses, [2], [source.nbytes])
                finally:
                    data_plane.release_sources(addresses)
                    completed.set()

            thread = threading.Thread(target=write)
            thread.start()
            assert entered.wait(timeout=2)
            assert not completed.is_set()
            assert engine.batch_unregister_calls == []
            finish.set()
            thread.join(timeout=2)
            assert completed.is_set()
            assert engine.batch_unregister_calls == [addresses]
            data_plane.close()


class TestECMooncakeFactory:
    """Validate factory registration and compatibility exports."""

    def test_factory_registers_connector(self):
        cls = ECConnectorFactory.get_connector_class(
            Mock(ec_connector="ECMooncakeConnector")
        )
        assert cls is ECMooncakeConnector
        assert (
            cls.__module__
            == "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector"
        )

    def test_public_exports_are_compatible_and_narrow(self):
        assert mooncake_ec_connector.__all__ == [
            "ECMooncakeConnector",
            "ECMooncakeConnectorMetadata",
            "ECMooncakeLoadSpec",
            "ECMooncakePushSpec",
            "ECMooncakeWorkerMetadata",
        ]


class TestContiguousAllocator:
    """Validate aligned allocation, reuse, and range coalescing."""

    def test_reuses_and_coalesces_contiguous_regions(self):
        allocator = ContiguousAllocator(1024, alignment=256)

        first = allocator.allocate(1)
        second = allocator.allocate(300)
        assert first == (0, 256)
        assert second == (256, 512)
        assert allocator.allocate(300) is None

        allocator.free(*first)
        allocator.free(*second)
        assert allocator.allocate(1024) == (0, 1024)

    def test_splits_until_exhausted(self):
        allocator = ContiguousAllocator(768, alignment=256)

        assert allocator.allocate(257) == (0, 512)
        assert allocator.allocate(1) == (512, 256)
        assert allocator.allocate(1) is None


class TestResidentPool:
    """Validate resident pin, lease, replacement, and LRU semantics."""

    def test_lru_skips_rejected_entry_and_replaces_without_losing_owner(self):
        pool = ResidentPool[str]()
        pool.insert("oldest", "first", 256)
        pool.insert("next", "second", 256)
        pool.retire("oldest")
        pool.retire("next")

        evicted = pool.evict_lru(lambda key, _: key != "oldest")

        assert evicted == "next"
        assert pool.get("oldest") == "first"
        assert pool.insert("oldest", "replacement", 128) == "first"
        assert pool.get("oldest") == "replacement"
        assert pool.used == 128

    def test_displaced_entry_waits_for_every_lease(self):
        pool = ResidentPool[str]()
        pool.insert("hash", "original", 256)
        first = pool.acquire("hash")
        second = pool.acquire("hash")
        assert first is not None and second is not None

        assert pool.insert("hash", "replacement", 256) is None
        assert pool.used == 512
        assert pool.release(first) is None
        assert pool.release(second) == "original"
        assert pool.used == 256
        assert pool.release(second) is None


class TestMooncakeMemoryPools:
    """Validate Producer staging and Consumer residency ownership."""

    class _Event:
        """Minimal CUDA-event substitute controlling deferred frees."""

        def __init__(self, complete: bool):
            self.complete = complete

        def record(self, stream):
            pass

        def query(self):
            return self.complete

    def test_consumer_replacement_waits_for_cached_owner(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        mooncake_transfer.unregister_memory.return_value = True
        pool = ConsumerMemoryPool(768, mooncake_transfer)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        first = pool.try_allocate(64, (16,), torch.float32)
        replacement = pool.try_allocate(64, (16,), torch.float32)
        assert first is not None and replacement is not None
        pool.publish("hash", first)
        held = pool.acquire_cached("hash", (16,), torch.float32)
        assert held is not None
        pool.publish("hash", replacement)

        third = pool.try_allocate(64, (16,), torch.float32)
        assert third is not None
        assert third.offset != first.offset

        pool.release_cached(held)
        reused = pool.try_allocate(64, (16,), torch.float32)
        assert reused is not None
        assert reused.offset == first.offset
        assert pool.take_resident("hash", (16,), "float32") is replacement.tensor

    def test_cached_consume_returns_newer_canonical_allocation(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        pool = ConsumerMemoryPool(768, mooncake_transfer)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        first = pool.try_allocate(64, (16,), torch.float32)
        replacement = pool.try_allocate(64, (16,), torch.float32)
        assert first is not None and replacement is not None
        pool.publish("hash", first)
        held = pool.acquire_cached("hash", (16,), torch.float32)
        assert held is not None
        pool.publish("hash", replacement)

        canonical = pool.publish("hash", held.value, held)

        assert canonical is replacement
        reused = pool.try_allocate(64, (16,), torch.float32)
        assert reused is not None
        assert reused.offset == first.offset

    def test_consumer_defers_retired_reuse_until_event_completes(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        pool = ConsumerMemoryPool(256, mooncake_transfer)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        allocation = pool.try_allocate(64, (16,), torch.float32)
        assert allocation is not None
        pool.publish("hash", allocation)
        event = self._Event(complete=False)

        with (
            patch.object(memory.torch, "Event", return_value=event),
            patch.object(memory.torch.accelerator, "current_stream"),
        ):
            pool.retire_stale({}, set())
            assert pool.reclaim_and_allocate(64, (16,), torch.float32) is None
            event.complete = True
            reused = pool.try_allocate(64, (16,), torch.float32)

        assert reused is not None
        assert reused.offset == allocation.offset
        assert pool.drain_reclaimed() == {"hash"}

    def test_consumer_registration_failure_disables_pool(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 1
        pool = ConsumerMemoryPool(256, mooncake_transfer)

        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)

        assert pool.tensor is None
        mooncake_transfer.register_memory.assert_called_once()

    def test_nonreceiving_consumer_never_registers_or_unregisters_pool(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        pool = ConsumerMemoryPool(256, mooncake_transfer)

        pool.prepare(torch.device("cpu"), receiving_rank=False, allow_host=True)
        pool.close()
        pool.close()

        assert pool.tensor is None
        mooncake_transfer.register_memory.assert_not_called()
        mooncake_transfer.unregister_memory.assert_not_called()

    def test_consumer_close_unregisters_once_and_releases_parent(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        mooncake_transfer.unregister_memory.return_value = True
        pool = ConsumerMemoryPool(256, mooncake_transfer)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        parent = pool.tensor

        pool.close()
        pool.close()

        mooncake_transfer.unregister_memory.assert_called_once_with(parent)
        assert pool.tensor is None

    def test_producer_reuses_staging_and_keeps_parent_for_later_close_phase(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        pool = ProducerMemoryPool(256, mooncake_transfer)
        source = torch.arange(16, dtype=torch.float32)

        first = pool.stage([source])
        assert first is not None
        assert torch.equal(first.tensors[0], source)
        pool.release(first)
        second = pool.stage([source])
        assert second is not None
        assert second.regions == first.regions
        pool.release(second)
        parent = pool.tensor

        pool.close()
        pool.close()

        assert pool.tensor is parent
        mooncake_transfer.unregister_memory.assert_not_called()

    def test_producer_falls_back_when_staging_pool_allocation_fails(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        pool = ProducerMemoryPool(256, mooncake_transfer)

        with patch.object(memory.torch, "empty", side_effect=torch.OutOfMemoryError):
            assert pool.stage([torch.ones(16)]) is None
            assert pool.stage([torch.ones(16)]) is None

        mooncake_transfer.register_memory.assert_not_called()


class TestMooncakeECConfig:
    """Validate normalization, defaults, immutability, and bounds."""

    def test_defaults_are_an_immutable_snapshot(self, mock_vllm_config_producer):
        config = MooncakeECConfig.from_vllm_config(
            mock_vllm_config_producer, ECConnectorRole.SCHEDULER
        )

        assert config == MooncakeECConfig(
            is_producer=True,
            is_consumer=False,
            protocol="tcp",
            buffer_device="cuda",
            reservation_port=None,
            reservation_addr=None,
            control_timeout_s=30,
            push_wait_timeout_s=60,
            transfer_workers=4,
            control_workers=8,
            producer_pool_size=1_000_000_000,
            consumer_pool_size=1_000_000_000,
            transfer_metrics_log_interval=10,
            consumer_metrics_log_interval=10,
        )

        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "mooncake_protocol"
        ] = "rdma"
        assert config.protocol == "tcp"
        with pytest.raises(FrozenInstanceError):
            config.protocol = "rdma"  # type: ignore[misc]

    def test_custom_values_are_normalized(self, mock_vllm_config_consumer):
        source = mock_vllm_config_consumer
        source.parallel_config.tensor_parallel_size = 2
        source.parallel_config.data_parallel_size = 3
        source.parallel_config.data_parallel_index = 1
        source.ec_transfer_config.ec_buffer_device = " cpu "
        source.ec_transfer_config.ec_buffer_size = 2048
        source.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": " tcp ",
            "reservation_zmq_port": "5000",
            "control_timeout_s": "1.5",
            "push_wait_timeout_s": "2.5",
            "transfer_max_workers": "3",
            "control_max_workers": "5",
            "producer_buffer_pool_size": "1024",
            "consumer_buffer_pool_size": "1536",
            "transfer_metrics_log_interval": "0",
            "consumer_metrics_log_interval": "7",
        }

        config = MooncakeECConfig.from_vllm_config(source, ECConnectorRole.WORKER)

        assert config.reservation_port == 5002
        assert config.reservation_addr == "tcp://127.0.0.1:5002"
        assert config.control_timeout_s == 1.5
        assert config.push_wait_timeout_s == 2.5
        assert config.transfer_workers == 3
        assert config.control_workers == 5
        assert config.producer_pool_size == 1024
        assert config.consumer_pool_size == 1536
        assert config.transfer_metrics_log_interval == 0
        assert config.consumer_metrics_log_interval == 7
        assert config.buffer_device == "cpu"
        assert config.protocol == "tcp"

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("control_timeout_s", 0),
            ("push_wait_timeout_s", -1),
            ("transfer_max_workers", 0),
            ("control_max_workers", -1),
            ("producer_buffer_pool_size", 0),
            ("consumer_buffer_pool_size", -1),
        ],
    )
    @pytest.mark.parametrize(
        "role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER]
    )
    def test_rejects_nonpositive_values(
        self, mock_vllm_config_producer, key, value, role
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_producer, role)

    @pytest.mark.parametrize(
        "role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER]
    )
    def test_rejects_nonpositive_registered_buffer(
        self, mock_vllm_config_producer, role
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = 0

        with pytest.raises(ValueError, match="ec_buffer_size > 0"):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_producer, role)

    @pytest.mark.parametrize("key", ["control_timeout_s", "push_wait_timeout_s"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
    def test_rejects_invalid_timeouts(self, mock_vllm_config_producer, key, value):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    @pytest.mark.parametrize(
        "key",
        [
            "reservation_zmq_port",
            "transfer_max_workers",
            "control_max_workers",
            "producer_buffer_pool_size",
            "consumer_buffer_pool_size",
        ],
    )
    @pytest.mark.parametrize("value", [1.5, True])
    def test_rejects_noninteger_values(self, mock_vllm_config_producer, key, value):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    @pytest.mark.parametrize("value", [1.5, True])
    def test_rejects_noninteger_registered_buffer(
        self, mock_vllm_config_producer, value
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = value

        with pytest.raises(ValueError, match="ec_buffer_size"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    def test_accepts_integral_float_integer_values(self, mock_vllm_config_producer):
        source = mock_vllm_config_producer
        source.ec_transfer_config.ec_buffer_size = 8.0
        source.ec_transfer_config.ec_connector_extra_config.update(
            {
                "reservation_zmq_port": 5000.0,
                "transfer_max_workers": 2.0,
                "control_max_workers": 3.0,
                "producer_buffer_pool_size": 4.0,
                "consumer_buffer_pool_size": 5.0,
            }
        )

        config = MooncakeECConfig.from_vllm_config(source, ECConnectorRole.WORKER)

        assert (
            config.reservation_port,
            config.transfer_workers,
            config.control_workers,
            config.producer_pool_size,
            config.consumer_pool_size,
        ) == (5000, 2, 3, 4, 5)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("mooncake_protocol", ""),
            ("mooncake_protocol", " "),
            ("mooncake_protocol", None),
            ("mooncake_protocol", 1),
            ("reservation_zmq_addr", ""),
            ("reservation_zmq_addr", " "),
            ("reservation_zmq_addr", None),
            ("reservation_zmq_addr", 1),
        ],
    )
    def test_rejects_invalid_strings(self, mock_vllm_config_producer, key, value):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )

    @pytest.mark.parametrize("buffer_device", [None, "", " \t"])
    def test_normalizes_default_buffer_device(
        self, mock_vllm_config_producer, buffer_device
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = buffer_device

        config = MooncakeECConfig.from_vllm_config(
            mock_vllm_config_producer, ECConnectorRole.WORKER
        )

        assert config.buffer_device == "cuda"

    def test_strips_buffer_device(self, mock_vllm_config_producer):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = " cuda "

        config = MooncakeECConfig.from_vllm_config(
            mock_vllm_config_producer, ECConnectorRole.WORKER
        )

        assert config.buffer_device == "cuda"

    @pytest.mark.parametrize("buffer_device", [1, True])
    def test_rejects_invalid_buffer_device(
        self, mock_vllm_config_producer, buffer_device
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = buffer_device

        with pytest.raises(ValueError, match="ec_buffer_device"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    @pytest.mark.parametrize(
        "key", ["transfer_metrics_log_interval", "consumer_metrics_log_interval"]
    )
    @pytest.mark.parametrize(
        "value", [-1, float("nan"), float("inf"), float("-inf"), True, None]
    )
    def test_rejects_invalid_metrics_intervals(
        self, mock_vllm_config_producer, key, value
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    def test_zero_disables_metrics(self, mock_vllm_config_producer):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config.update(
            {
                "transfer_metrics_log_interval": 0,
                "consumer_metrics_log_interval": 0,
            }
        )

        config = MooncakeECConfig.from_vllm_config(
            mock_vllm_config_producer, ECConnectorRole.WORKER
        )

        assert config.transfer_metrics_log_interval == 0
        assert config.consumer_metrics_log_interval == 0

    def test_submillisecond_control_timeout_uses_one_millisecond(
        self, mock_vllm_config_producer
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "control_timeout_s"
        ] = 0.0001
        with (
            patch_ec_mooncake_deps(),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "scheduler.ControlClient"
            ) as scheduler_client,
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "worker.ControlClient"
            ) as worker_client,
        ):
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            worker = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            scheduler.shutdown()
            worker.shutdown()

        scheduler_client.assert_called_once_with(1)
        worker_client.assert_called_once_with(1)

    @pytest.mark.parametrize("timeout", [1e308, sys.float_info.max])
    def test_rejects_control_timeout_too_large_for_zmq(
        self, mock_vllm_config_producer, timeout
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "control_timeout_s"
        ] = timeout

        with pytest.raises(ValueError, match="control_timeout_s"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )

    def test_rejects_pipeline_parallel_producer(self, mock_vllm_config_producer):
        mock_vllm_config_producer.parallel_config.pipeline_parallel_size = 2

        with pytest.raises(ValueError, match="pipeline parallelism"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )

    @pytest.mark.parametrize("port", [0, 65536])
    def test_rejects_out_of_range_base_port(self, mock_vllm_config_consumer, port):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "reservation_zmq_port"
        ] = port

        with pytest.raises(ValueError, match="1..65535"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )

    def test_rejects_topology_that_overflows_port_range(
        self, mock_vllm_config_consumer
    ):
        source = mock_vllm_config_consumer
        source.parallel_config.tensor_parallel_size = 4
        source.parallel_config.data_parallel_index = 1
        source.ec_transfer_config.ec_connector_extra_config["reservation_zmq_port"] = (
            65530
        )

        with pytest.raises(ValueError, match="ports must be in 1..65535"):
            MooncakeECConfig.from_vllm_config(source, ECConnectorRole.SCHEDULER)

    def test_consumer_role_requirements_differ_only_by_process_role(
        self, mock_vllm_config_consumer
    ):
        extra = {"reservation_zmq_addr": "tcp://consumer:19019"}
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = extra

        scheduler = MooncakeECConfig.from_vllm_config(
            mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
        )
        assert scheduler.reservation_addr == "tcp://consumer:19019"
        with pytest.raises(ValueError, match="workers require reservation_zmq_port"):
            MooncakeECConfig.from_vllm_config(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )


class TestECMooncakeConnectorValidation:
    """Validate role construction, optional dependencies, and topology rules."""

    @pytest.mark.parametrize(
        "role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER]
    )
    def test_requires_transfer_engine_symbol_for_each_role(
        self, mock_vllm_config_producer, monkeypatch, role
    ):
        from vllm.distributed.ec_transfer.ec_connector.mooncake import _availability

        fake_package = ModuleType("mooncake")
        fake_package.__path__ = []
        fake_engine = ModuleType("mooncake.engine")
        try:
            with monkeypatch.context() as context:
                context.setitem(sys.modules, "mooncake", fake_package)
                context.setitem(sys.modules, "mooncake.engine", fake_engine)
                importlib.reload(_availability)
                with pytest.raises(ImportError, match="mooncake-transfer-engine"):
                    ECMooncakeConnector(mock_vllm_config_producer, role)
        finally:
            importlib.reload(_availability)

    def test_rejects_sharded_producer(self, mock_vllm_config_producer):
        """One copy of each encoder output, so sharding only duplicates it."""
        mock_vllm_config_producer.parallel_config.tensor_parallel_size = 2
        with (
            patch_ec_mooncake_deps(),
            pytest.raises(ValueError, match="tensor_parallel_size"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, ECConnectorRole.WORKER)

    def test_accepts_sharded_consumer(self, mock_vllm_config_consumer):
        """Consumers shard: each rank gathers from its own encoder cache."""
        mock_vllm_config_consumer.parallel_config.tensor_parallel_size = 4
        mock_vllm_config_consumer.parallel_config.pipeline_parallel_size = 2
        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            connector.shutdown()

    def test_replicated_consumer_addresses_its_own_block(
        self, mock_vllm_config_consumer
    ):
        """Each replica owns a distinct block of control ports.

        Replicas run their own schedulers and control channels, so sharing a
        port would collide at bind time and cross-subscribe their event
        channels. The block is derived from `data_parallel_index` because a
        non-MoE replica is reconfigured to look like DP=1, which resets
        `data_parallel_rank`.
        """
        cfg = mock_vllm_config_consumer
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.data_parallel_size = 3
        cfg.parallel_config.data_parallel_index = 2
        # What a non-MoE replica actually looks like: reconfigured to DP=1, so
        # `data_parallel_rank` no longer identifies it but the index still does.
        cfg.parallel_config.data_parallel_rank = 0
        cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19500,
        }
        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(cfg, ECConnectorRole.SCHEDULER)
            try:
                # Replica 2 of a TP=2 consumer starts after two 2-port blocks.
                assert connector._scheduler is not None
                assert (
                    connector._scheduler._reservation_zmq_addr
                    == "tcp://127.0.0.1:19504"
                )
            finally:
                connector.shutdown()

    def test_rejects_replicated_producer(self, mock_vllm_config_producer):
        """A producer holds one copy of each output, so replicating it only
        duplicates the push."""
        mock_vllm_config_producer.parallel_config.data_parallel_size = 2
        with (
            patch_ec_mooncake_deps(),
            pytest.raises(ValueError, match="data_parallel_size=1"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, ECConnectorRole.SCHEDULER)

    def test_scheduler_hooks_route_exactly(self, mock_vllm_config_producer):
        scheduler = Mock()
        scheduler.take_unavailable_requests.return_value = {"unavailable"}
        scheduler.has_cache_item.return_value = True
        scheduler.ensure_cache_available.return_value = False
        scheduler.build_connector_meta.return_value = "metadata"
        scheduler.has_pending_push_work.return_value = True
        scheduler.request_finished.return_value = (True, {"result": 1})
        with patch.object(
            ECMooncakeScheduler,
            "from_vllm_config",
            return_value=scheduler,
        ) as from_vllm_config:
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            request = Mock()
            scheduler_output = Mock()
            connector_output = Mock()
            try:
                assert connector.take_unavailable_requests() == {"unavailable"}
                assert connector.has_cache_item("hash") is True
                assert connector.ensure_cache_available(request, 7, {"local"}) is False
                connector.update_state_after_alloc(request, 2)
                connector.update_state_after_free(request, 3)
                assert connector.build_connector_meta(scheduler_output) == "metadata"
                connector.update_connector_output(connector_output)
                assert connector.has_pending_push_work() is True
                assert connector.request_finished(request) == (True, {"result": 1})
            finally:
                connector.shutdown()

        from_vllm_config.assert_called_once_with(mock_vllm_config_producer)
        scheduler.take_unavailable_requests.assert_called_once_with()
        scheduler.has_cache_item.assert_called_once_with("hash")
        scheduler.ensure_cache_available.assert_called_once_with(request, 7, {"local"})
        scheduler.update_state_after_alloc.assert_called_once_with(request, 2)
        scheduler.update_state_after_free.assert_called_once_with(request, 3)
        scheduler.build_connector_meta.assert_called_once_with(scheduler_output)
        scheduler.update_connector_output.assert_called_once_with(connector_output)
        scheduler.has_pending_push_work.assert_called_once_with()
        scheduler.request_finished.assert_called_once_with(request)
        scheduler.close.assert_called_once_with()

    def test_worker_hooks_route_exactly(self, mock_vllm_config_producer):
        metadata = ECMooncakeConnectorMetadata()
        worker = Mock()
        worker.get_finished.return_value = ({"saved"}, {"loaded"})
        worker.build_connector_worker_meta.return_value = "worker-metadata"
        with patch.object(
            ECMooncakeWorker,
            "from_vllm_config",
            return_value=worker,
        ) as from_vllm_config:
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            connector.bind_connector_metadata(metadata)
            encoder_cache: dict[str, torch.Tensor] = {}
            try:
                connector.start_worker_services()
                connector.start_save_caches(encoder_cache=encoder_cache, marker=1)
                connector.start_load_caches(encoder_cache, marker=2)
                connector.save_caches(encoder_cache, "hash", marker=3)
                assert connector.get_finished({"finished"}) == (
                    {"saved"},
                    {"loaded"},
                )
                assert connector.build_connector_worker_meta() == "worker-metadata"
            finally:
                connector.shutdown()

        from_vllm_config.assert_called_once_with(mock_vllm_config_producer)
        worker.start_services.assert_called_once_with()
        worker.start_save_caches.assert_called_once_with(
            metadata, encoder_cache=encoder_cache, marker=1
        )
        worker.start_load_caches.assert_called_once_with(
            metadata, encoder_cache, marker=2
        )
        worker.save_caches.assert_called_once_with(encoder_cache, "hash", marker=3)
        worker.get_finished.assert_called_once_with({"finished"})
        worker.build_connector_worker_meta.assert_called_once_with()
        worker.close.assert_called_once_with()

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("start_save_caches", ()),
            ("start_load_caches", ({},)),
        ],
    )
    def test_worker_load_and_save_reject_wrong_metadata(
        self, mock_vllm_config_producer, method, args
    ):
        class OtherMetadata(ECConnectorMetadata):
            """Represent an incompatible connector metadata implementation."""

            pass

        worker = Mock()
        with patch.object(
            ECMooncakeWorker,
            "from_vllm_config",
            return_value=worker,
        ):
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            connector.bind_connector_metadata(OtherMetadata())
            try:
                with pytest.raises(AssertionError):
                    getattr(connector, method)(*args)
            finally:
                connector.shutdown()

        getattr(worker, method).assert_not_called()

    @pytest.mark.parametrize(
        ("method", "args", "kwargs"),
        [
            ("start_worker_services", (), {}),
            ("start_save_caches", (), {}),
            ("start_load_caches", ({},), {}),
            ("save_caches", ({}, "hash"), {}),
            ("get_finished", (set(),), {}),
            ("build_connector_worker_meta", (), {}),
        ],
    )
    def test_scheduler_rejects_worker_hooks(
        self, mock_vllm_config_producer, method, args, kwargs
    ):
        with patch.object(
            ECMooncakeScheduler,
            "from_vllm_config",
            return_value=Mock(),
        ):
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            try:
                with pytest.raises(AssertionError):
                    getattr(connector, method)(*args, **kwargs)
            finally:
                connector.shutdown()

    @pytest.mark.parametrize(
        ("method", "args", "kwargs"),
        [
            ("take_unavailable_requests", (), {}),
            ("has_cache_item", ("hash",), {}),
            ("ensure_cache_available", (Mock(), 0, set()), {}),
            ("update_state_after_alloc", (Mock(), 0), {}),
            ("update_state_after_free", (Mock(), 0), {}),
            ("build_connector_meta", (Mock(),), {}),
            ("update_connector_output", (Mock(),), {}),
            ("has_pending_push_work", (), {}),
            ("request_finished", (Mock(),), {}),
        ],
    )
    def test_worker_rejects_scheduler_hooks(
        self, mock_vllm_config_producer, method, args, kwargs
    ):
        with patch.object(
            ECMooncakeWorker,
            "from_vllm_config",
            return_value=Mock(),
        ):
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                with pytest.raises(AssertionError):
                    getattr(connector, method)(*args, **kwargs)
            finally:
                connector.shutdown()

    @pytest.mark.parametrize(
        ("role", "active", "inactive"),
        [
            (ECConnectorRole.SCHEDULER, "_scheduler", "_worker"),
            (ECConnectorRole.WORKER, "_worker", "_scheduler"),
        ],
    )
    def test_exactly_one_role_and_idempotent_shutdown(
        self, mock_vllm_config_producer, role, active, inactive
    ):
        scheduler = Mock()
        worker = Mock()
        with (
            patch.object(
                ECMooncakeScheduler,
                "from_vllm_config",
                return_value=scheduler,
            ),
            patch.object(
                ECMooncakeWorker,
                "from_vllm_config",
                return_value=worker,
            ),
        ):
            connector = ECMooncakeConnector(mock_vllm_config_producer, role)
            assert getattr(connector, active) is not None
            assert getattr(connector, inactive) is None
            assert set(connector.__dict__) - {
                "_connector_metadata",
                "_vllm_config",
                "_role",
                "_is_producer",
                "_is_consumer",
            } == {"_scheduler", "_worker", "_closed"}
            connector.shutdown()
            connector.shutdown()

        if role == ECConnectorRole.SCHEDULER:
            scheduler.close.assert_called_once_with()
            worker.close.assert_not_called()
        else:
            worker.close.assert_called_once_with()
            scheduler.close.assert_not_called()

    def test_rejects_unknown_role(self, mock_vllm_config_producer):
        invalid_role = Mock(name="invalid_role")
        with pytest.raises(ValueError, match="Unknown EC connector role"):
            ECMooncakeConnector(mock_vllm_config_producer, invalid_role)

    def test_del_is_best_effort(self):
        connector = object.__new__(ECMooncakeConnector)
        with patch.object(
            ECMooncakeConnector,
            "shutdown",
            side_effect=RuntimeError("shutdown failed"),
        ) as shutdown:
            connector.__del__()
        shutdown.assert_called_once_with()


class TestECMooncakeMetadata:
    """Validate metadata compatibility, pickling, and aggregation inputs."""

    def test_old_imports_reexport_packaged_metadata(self):
        assert ECMooncakeLoadSpec is metadata.ECMooncakeLoadSpec
        assert ECMooncakePushSpec is metadata.ECMooncakePushSpec
        assert ECMooncakeConnectorMetadata is metadata.ECMooncakeConnectorMetadata
        assert ECMooncakeWorkerMetadata is metadata.ECMooncakeWorkerMetadata

    @pytest.mark.parametrize(
        "metadata",
        [
            ECMooncakeConnectorMetadata(
                loads=[
                    ECMooncakeLoadSpec(
                        mm_hash="load",
                        num_token=2,
                        nbytes=8,
                        shape=(2, 4),
                        dtype="float16",
                        pushed=True,
                        transfer_id="transfer",
                        reservation_id="reservation",
                        local=True,
                    )
                ],
                pushes=[
                    ECMooncakePushSpec(
                        mm_hash="push",
                        nbytes=8,
                        shape=(2, 4),
                        dtype="float16",
                        consumer_zmq="tcp://127.0.0.1:1234",
                        transfer_id="transfer",
                        request_id="request",
                    )
                ],
            ),
            ECMooncakeWorkerMetadata(
                loaded={"loaded"},
                failed_loads={"failed"},
                reclaimed={"reclaimed"},
                pending_loads=True,
                pending_saves=True,
            ),
        ],
    )
    def test_metadata_pickle_round_trip(self, metadata):
        assert ForkingPickler.loads(ForkingPickler.dumps(metadata)) == metadata


class TestECMooncakeWorkerMetadataAggregation:
    """Validate cross-rank success intersection and failure union rules."""

    def test_an_item_one_rank_missed_is_not_loaded(self):
        """Each rank gathers from its own cache, so all of them must have it.

        Reporting it as loaded because one rank succeeded left the scheduler
        marking the hash ready while another rank raised on the cache miss.
        """
        rank0 = ECMooncakeWorkerMetadata(loaded={"a", "b"})
        rank1 = ECMooncakeWorkerMetadata(loaded={"a"}, failed_loads={"b"})

        merged = rank0.aggregate(rank1)

        assert merged.loaded == {"a"}
        assert merged.failed_loads == {"b"}

    def test_a_reclaim_on_any_rank_invalidates_residency(self):
        """The scheduler mirrors one pool, so the weakest rank decides."""
        merged = ECMooncakeWorkerMetadata(loaded={"a"}).aggregate(
            ECMooncakeWorkerMetadata(loaded={"a"}, reclaimed={"c"})
        )
        assert merged.reclaimed == {"c"}


class TestSchedulerTransferTable:
    """Validate Scheduler transfer transitions, indexes, and retention."""

    @staticmethod
    def pushed_spec(transfer_id: str, mm_hash: str = "hash") -> ECMooncakeLoadSpec:
        return ECMooncakeLoadSpec(
            mm_hash=mm_hash,
            num_token=0,
            nbytes=16,
            shape=(4,),
            dtype="float32",
            pushed=True,
            transfer_id=transfer_id,
            reservation_id=f"reservation-{transfer_id}",
        )

    def test_legal_load_and_resident_reload_use_authoritative_record(self):
        assert SchedulerTransferTable is state.SchedulerTransferTable
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, accepted = table.observe_ready(self.pushed_spec("transfer"), 10)

        assert accepted and record.state is SchedulerTransferState.AVAILABLE
        assert table.begin_load("hash", 7, "transfer", "request") is record
        assert table.take_loads_to_dispatch() == [record]
        assert record.spec is not None and record.spec.num_token == 7
        assert table.complete_load("hash")
        table.release_ready("hash", 1)
        assert record.state is SchedulerTransferState.RESIDENT
        assert table.begin_load("hash", 9) is record
        assert record.spec is not None and record.spec.local

    def test_illegal_transition_is_rejected(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("transfer"), 10)

        with pytest.raises(InvalidSchedulerTransferTransition):
            table.mark_unavailable("transfer", "late", 1)
        assert record.state is SchedulerTransferState.AVAILABLE

    def test_same_hash_index_preserves_transfer_order_and_identity(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        first, _ = table.observe_ready(self.pushed_spec("first"), 10)
        second, _ = table.observe_ready(self.pushed_spec("second"), 10)

        assert table.records_for_hash("hash", tuple(SchedulerTransferState)) == [
            first,
            second,
        ]
        assert table.begin_load("hash", 3) is first
        assert (
            table.first_for_hash("hash", (SchedulerTransferState.AVAILABLE,)) is second
        )
        with pytest.raises(ValueError):
            table.observe_ready(self.pushed_spec("first", "other-hash"), 10)

    def test_unavailable_notification_drains_once_and_rejects_late_ready(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record = table.wait_for_event("transfer", "request-r", "hash", 1)
        table.mark_unavailable("transfer", "timed out", 2)

        assert table.take_unavailable_requests() == {"request-r"}
        assert table.take_unavailable_requests() == set()
        table.wait_for_event("transfer", "request-r", "hash", 3)
        assert table.take_unavailable_requests() == set()
        table.wait_for_event("transfer", "request-n", "hash", 3)
        assert table.take_unavailable_requests() == {"request-n"}
        assert table.take_unavailable_requests() == set()
        table.wait_for_event("transfer", "request-n", "hash", 3)
        assert table.take_unavailable_requests() == set()
        same, accepted = table.observe_ready(self.pushed_spec("transfer"), 40)
        assert same is record and not accepted
        assert record.state is SchedulerTransferState.UNAVAILABLE

    def test_cancel_and_duplicate_completion_are_idempotent(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        cancelled = table.wait_for_event("cancelled", "request", "hash", 10)
        assert table.cancel("cancelled", 1)
        assert not table.cancel("cancelled", 2)
        _, accepted = table.observe_ready(self.pushed_spec("cancelled"), 40)
        assert not accepted and cancelled.state is SchedulerTransferState.CANCELLED

        record, _ = table.observe_ready(self.pushed_spec("completed", "other"), 10)
        table.begin_load("other", 4, "completed")
        assert table.complete_load("other")
        assert table.complete_load("other")
        assert record.state is SchedulerTransferState.READY

    def test_failed_record_expires_from_record_and_hash_index(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("failed"), 10)
        table.begin_load("hash", 4, "failed")

        assert table.fail_load("hash", "copy failed", 20)
        assert record.deadline == 50
        _, dropped = table.expire(51, terminal_limit=100)
        assert dropped == 1
        assert table.get("failed") is None
        assert table.records_for_hash("hash", tuple(SchedulerTransferState)) == []

    def test_reclaimed_resident_tombstone_expires(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("reclaimed"), 10)
        table.begin_load("hash", 4, "reclaimed")
        table.complete_load("hash")
        table.release_ready("hash", 20)

        table.reclaim("hash", 30)
        assert record.state is SchedulerTransferState.EXPIRED
        assert record.deadline == 60
        table.expire(61, terminal_limit=100)
        assert table.get("reclaimed") is None
        assert table.records_for_hash("hash", tuple(SchedulerTransferState)) == []

    def test_capacity_eviction_tombstone_expires(self):
        table = SchedulerTransferTable(resident_capacity=0, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("evicted"), 10)
        table.begin_load("hash", 4, "evicted")
        table.complete_load("hash")

        table.release_ready("hash", 20)
        assert record.state is SchedulerTransferState.EXPIRED
        assert record.deadline == 50
        table.expire(51, terminal_limit=100)
        assert table.get("evicted") is None
        assert table.records_for_hash("hash", tuple(SchedulerTransferState)) == []

    def test_terminal_record_limit_prunes_oldest_records(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        for transfer_id in ("first", "second", "third"):
            table.cancel(transfer_id, 1)

        _, dropped = table.expire(2, terminal_limit=1)
        assert dropped == 2
        assert table.get("first") is None
        assert table.get("second") is None
        assert table.get("third") is not None

    def test_zero_terminal_limit_prunes_every_record(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        table.cancel("first", 1)
        table.cancel("second", 1)

        _, dropped = table.expire(2, terminal_limit=0)
        assert dropped == 2
        assert table.get("first") is None
        assert table.get("second") is None

    def test_negative_terminal_limit_is_rejected_without_mutation(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record = table.wait_for_event("transfer", "request", "hash", 1)

        with pytest.raises(ValueError, match="terminal_limit"):
            table.expire(2, terminal_limit=-1)
        assert table.get("transfer") is record
        assert record.state is SchedulerTransferState.WAITING_EVENT

    def test_same_hash_residency_uses_only_the_latest_completed_record(self):
        table = SchedulerTransferTable(resident_capacity=32, tombstone_ttl=30)
        first, _ = table.observe_ready(self.pushed_spec("first"), 10)
        table.begin_load("hash", 4, "first")
        table.complete_load("hash")
        table.release_ready("hash", 20)
        second, _ = table.observe_ready(self.pushed_spec("second"), 30)
        table.begin_load("hash", 4, "second")
        table.complete_load("hash")
        table.release_ready("hash", 40)
        third, _ = table.observe_ready(self.pushed_spec("third", "other"), 50)
        table.begin_load("other", 4, "third")
        table.complete_load("other")
        table.release_ready("other", 60)

        assert first.state is SchedulerTransferState.EXPIRED
        assert second.state is SchedulerTransferState.RESIDENT
        assert third.state is SchedulerTransferState.RESIDENT
        assert table.resident_bytes == 32


class TestECMooncakeSchedulerMetadata:
    """Validate Scheduler decisions and per-step Worker metadata."""

    def test_cancel_confirms_topology_and_retries_only_failed_shards(self):
        scheduler = object.__new__(ECMooncakeScheduler)
        scheduler._topology = Mock(spec=ShardTopology)
        scheduler._topology.discover.side_effect = [
            None,
            ["shard-0", "shard-1", "shard-2"],
        ]
        scheduler._control_client = Mock(spec=ControlClient)
        called = []

        def request(addr, _payload):
            called.append(addr)
            if addr == "shard-0" and called.count(addr) == 1:
                raise RuntimeError("cancel shard failed")
            return {"cancelled": True}

        scheduler._control_client.request.side_effect = request
        assert scheduler._cancel_remote("base", "transfer", "reservation")
        assert scheduler._topology.discover.call_args_list == [
            call("base"),
            call("base"),
        ]
        assert called == ["shard-0", "shard-1", "shard-2", "shard-0"]

    def test_cancel_rejects_unconfirmed_topology_without_sending(self):
        scheduler = object.__new__(ECMooncakeScheduler)
        scheduler._topology = Mock(spec=ShardTopology)
        scheduler._topology.discover.return_value = None
        scheduler._control_client = Mock(spec=ControlClient)

        with pytest.raises(RuntimeError, match="discover every EC consumer shard"):
            scheduler._cancel_remote("base", "transfer", "reservation")

        assert scheduler._topology.discover.call_count == 2
        scheduler._control_client.request.assert_not_called()

    def test_missing_push_event_is_tracked(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0)
                assert (
                    scheduler._scheduler._consumer_scheduler_metrics["missing_event"]
                    == 1
                )
                record = scheduler._scheduler._transfers.get(f"{request.request_id}:0")
                assert record is not None
                assert record.state is SchedulerTransferState.WAITING_EVENT
            finally:
                scheduler.shutdown()

    def test_item_with_no_transfer_in_flight_is_reported_as_stalled(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """A push that never arrives must not wait silently forever."""
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
            "push_wait_timeout_s": 0.001,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                with (
                    patch.object(scheduler._scheduler, "_drain_push_notifications"),
                    patch(
                        "vllm.distributed.ec_transfer.ec_connector."
                        "mooncake.scheduler.time.monotonic",
                        side_effect=[10, 10.002, 10.003],
                    ),
                ):
                    assert not scheduler.ensure_cache_available(request, 0)
                    assert not scheduler.ensure_cache_available(request, 0)
                    assert (
                        scheduler._scheduler._consumer_scheduler_metrics["stalled"] == 1
                    )
                    record = scheduler._scheduler._transfers.get(
                        f"{request.request_id}:0"
                    )
                    assert record is not None
                    assert record.state is SchedulerTransferState.UNAVAILABLE
                    # The stall is reported once, not once per scheduling pass.
                    assert not scheduler.ensure_cache_available(request, 0)
                assert scheduler._scheduler._consumer_scheduler_metrics["stalled"] == 1
            finally:
                scheduler.shutdown()

    def test_pending_observation_ends_with_last_spec(self, mock_vllm_config_consumer):
        specs = [
            ECMooncakeLoadSpec(
                mm_hash="hash",
                num_token=1,
                nbytes=32,
                shape=(8,),
                dtype="float32",
                pushed=True,
                transfer_id=f"transfer-{index}",
            )
            for index in range(2)
        ]

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                for spec in specs:
                    scheduler._scheduler._transfers.observe_ready(spec, 10)
                scheduler._scheduler._transfers.cancel("transfer-0", 0)
                available = scheduler._scheduler._transfers.records_for_hash(
                    "hash", (SchedulerTransferState.AVAILABLE,)
                )
                assert [record.transfer_id for record in available] == ["transfer-1"]
                scheduler._scheduler._transfers.cancel("transfer-1", 0)
                assert (
                    scheduler._scheduler._transfers.first_for_hash(
                        "hash", (SchedulerTransferState.AVAILABLE,)
                    )
                    is None
                )
            finally:
                scheduler.shutdown()

    def test_available_expiry_is_cancelled_before_tombstone_cleanup(
        self, mock_vllm_config_consumer
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            record, _ = scheduler._scheduler._transfers.observe_ready(
                TestSchedulerTransferTable.pushed_spec("expired"), 0
            )
            try:
                with patch.object(scheduler._scheduler, "_queue_cancel") as cancel:
                    scheduler._scheduler._expire_transfers()
                cancel.assert_called_once_with("expired")
                assert record.state is SchedulerTransferState.EXPIRED
                assert scheduler._scheduler._transfers.get("expired") is record
            finally:
                scheduler.shutdown()

    def test_local_cache_hit_keeps_the_transfer(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """`local_cache_hashes` is a snapshot, so the transfer must survive it.

        Cancelling on a cache hit strands the request when the entry is
        evicted before it is scheduled: the item is then unreachable.
        """
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                request = mock_request_with_3_mm
                request.mm_features = request.mm_features[:1]
                mm_hash = request.mm_features[0].identifier
                request.ec_transfer_params = {
                    "ec_items": [
                        {"mm_hash": mm_hash, "transfer_id": "request-transfer"}
                    ]
                }
                scheduler._scheduler._transfers.observe_ready(
                    ECMooncakeLoadSpec(
                        mm_hash=mm_hash,
                        num_token=0,
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        pushed=True,
                        transfer_id="request-transfer",
                        reservation_id="reservation",
                    ),
                    10,
                )
                with (
                    patch.object(scheduler._scheduler, "_drain_push_notifications"),
                    patch.object(scheduler._scheduler, "_queue_cancel") as cancel,
                ):
                    assert scheduler.ensure_cache_available(request, 0, {mm_hash})
                    cancel.assert_not_called()
                    record = scheduler._scheduler._transfers.get("request-transfer")
                    assert record is not None
                    assert record.state is SchedulerTransferState.AVAILABLE

                    # Once the entry is evicted the request can still get it.
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert (
                    scheduler._scheduler._transfers.first_for_hash(
                        mm_hash, (SchedulerTransferState.LOADING,)
                    )
                    is not None
                )
            finally:
                scheduler.shutdown()

    def test_consumed_item_releases_its_transfer_immediately(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """The buffer goes back as soon as the item is consumed.

        Holding it until `request_finished` would pin a pool slot for the
        whole generation, long after the embedding was used.
        """
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier
        request.ec_transfer_params = {
            "ec_items": [{"mm_hash": mm_hash, "transfer_id": "consumed-transfer"}]
        }

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._transfers.observe_ready(
                    ECMooncakeLoadSpec(
                        mm_hash=mm_hash,
                        num_token=0,
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        pushed=True,
                        transfer_id="consumed-transfer",
                        reservation_id="reservation",
                    ),
                    10,
                )
                scheduler.update_state_after_free(request, 0)
                record = scheduler._scheduler._transfers.get("consumed-transfer")
                assert record is not None
                assert record.state is SchedulerTransferState.CANCELLED
            finally:
                scheduler.shutdown()

    def test_ready_hash_eviction_does_not_strand_a_later_transfer(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """The long-tail stall: an event arrives while the hash is still ready.

        Dropping it as redundant left the next request with no transfer at
        all once the encoder cache entry was freed, and nothing could bring
        the item back.
        """
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier
        request.ec_transfer_params = {
            "ec_items": [{"mm_hash": mm_hash, "transfer_id": "later-transfer"}]
        }
        event = {
            "mm_hash": mm_hash,
            "transfer_id": "later-transfer",
            "ready": True,
            "reservation_id": "later",
            "nbytes": 16,
            "shape": [4],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                current = ECMooncakeLoadSpec(
                    mm_hash=mm_hash,
                    num_token=0,
                    nbytes=16,
                    shape=(4,),
                    dtype="float32",
                    pushed=True,
                    transfer_id="current-transfer",
                    reservation_id="current",
                )
                scheduler._scheduler._transfers.observe_ready(
                    current, time.monotonic() + _LEASE_TTL_SECONDS
                )
                scheduler._scheduler._transfers.begin_load(
                    mm_hash, 4, "current-transfer"
                )
                scheduler._scheduler._transfers.take_loads_to_dispatch()
                scheduler._scheduler._transfers.complete_load(mm_hash)
                scheduler._scheduler._event_inbox.drain = Mock(return_value=[event])
                with patch.object(scheduler._scheduler, "_queue_cancel") as cancel:
                    scheduler._scheduler._drain_push_notifications()
                cancel.assert_not_called()
                later = scheduler._scheduler._transfers.get("later-transfer")
                assert later is not None
                assert later.state is SchedulerTransferState.AVAILABLE

                # The scheduler frees the encoder cache entry.
                scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[mm_hash])
                )
                assert (
                    scheduler._scheduler._transfers.first_for_hash(
                        mm_hash, (SchedulerTransferState.READY,)
                    )
                    is None
                )

                # The request that owns the transfer can still pick it up.
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert later.state is SchedulerTransferState.LOADING
            finally:
                scheduler.shutdown()

    def test_cancelled_transfer_ignores_late_ready_events(
        self, mock_vllm_config_consumer
    ):
        """Cancelled is terminal even when a ready event was already queued."""
        transfer_id = "cancelled-transfer"
        ports = [19101, 19102, 19103, 19104]
        event = {
            "mm_hash": "hash",
            "transfer_id": transfer_id,
            "ready": True,
            "reservation_id": "reservation",
            "nbytes": 16,
            "shape": [4],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._event_inbox.shard_count = len(ports)
                scheduler._scheduler._transfers.cancel(
                    transfer_id, time.monotonic(), mm_hash="hash"
                )
                scheduler._scheduler._event_inbox.drain = Mock(
                    return_value=[{**event, "shard": port} for port in ports]
                )

                scheduler._scheduler._drain_push_notifications()

                record = scheduler._scheduler._transfers.get(transfer_id)
                assert record is not None
                assert record.state is SchedulerTransferState.CANCELLED
                assert transfer_id not in scheduler._scheduler._event_ready_shards
                assert scheduler._scheduler._consumer_scheduler_metrics[
                    "events_cancelled"
                ] == len(ports)
            finally:
                scheduler.shutdown()

    def test_cancel_rpc_failure_keeps_tombstone_and_rejects_late_ready(
        self, mock_vllm_config_consumer
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            spec = TestSchedulerTransferTable.pushed_spec("transfer")
            record, _ = scheduler._scheduler._transfers.observe_ready(spec, 10)
            scheduler._scheduler._transfers.cancel("transfer", 1)
            failed = Mock()
            failed.done.return_value = True
            failed.result.side_effect = RuntimeError("unknown remote result")
            scheduler._scheduler._pending_cancels["transfer"] = failed
            try:
                scheduler._scheduler._poll_pending_cancels()
                assert record.state is SchedulerTransferState.CANCELLED
                assert record.spec is spec
                same, accepted = scheduler._scheduler._transfers.observe_ready(spec, 20)
                assert same is record and not accepted
            finally:
                scheduler.shutdown()

    def test_cancel_between_shards_drops_the_partial_readiness(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """A cancel mid-aggregation leaves nothing for the late shards to finish.

        The early shards are already counted when the request releases the
        item. Only clearing them keeps the remaining notifications from
        completing the set and rebuilding a spec for a buffer the worker
        freed as it cancelled.
        """
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier
        transfer_id = "half-reported-transfer"
        request.ec_transfer_params = {
            "ec_items": [{"mm_hash": mm_hash, "transfer_id": transfer_id}]
        }
        event = {
            "mm_hash": mm_hash,
            "transfer_id": transfer_id,
            "ready": True,
            "reservation_id": "reservation",
            "nbytes": 16,
            "shape": [4],
            "dtype": "float32",
        }

        def deliver(scheduler, *shards):
            scheduler._scheduler._event_inbox.drain.return_value = [
                {**event, "shard": shard} for shard in shards
            ]
            scheduler._scheduler._drain_pending = True
            scheduler._scheduler._drain_push_notifications()

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._reservation_zmq_addr = "tcp://127.0.0.1:19101"
                scheduler._scheduler._event_inbox.shard_count = 4
                scheduler._scheduler._event_inbox.drain = Mock()

                deliver(scheduler, 0, 1)
                assert scheduler._scheduler._event_ready_shards[transfer_id] == {0, 1}

                with patch.object(
                    scheduler._scheduler, "_cancel_remote", return_value=True
                ):
                    scheduler.update_state_after_free(request, 0)
                record = scheduler._scheduler._transfers.get(transfer_id)
                assert record is not None
                assert record.state is SchedulerTransferState.CANCELLED
                assert transfer_id not in scheduler._scheduler._event_ready_shards

                deliver(scheduler, 2, 3)

                assert record.state is SchedulerTransferState.CANCELLED
                assert transfer_id not in scheduler._scheduler._event_ready_shards
                assert (
                    scheduler._scheduler._consumer_scheduler_metrics["events_cancelled"]
                    == 2
                )
            finally:
                scheduler.shutdown()

    def test_cancelled_transfer_ids_stay_bounded(self, mock_vllm_config_consumer):
        """The ignore list is swept, not accumulated.

        It is consulted for every readiness notification and grows by one
        entry per multimodal item the instance serves, so retaining ids the
        worker has itself forgotten leaks for the life of the process.
        """
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._reservation_zmq_addr = "tcp://127.0.0.1:19101"
                scheduler._scheduler._event_inbox.drain = Mock(return_value=[])
                with patch.object(
                    scheduler._scheduler, "_cancel_remote", return_value=True
                ):
                    for name in ("first", "second", "third"):
                        scheduler._scheduler._queue_cancel(name)

                now = time.monotonic()
                third = scheduler._scheduler._transfers.get("third")
                assert third is not None and third.deadline is not None
                assert third.deadline > now
                assert third.deadline <= now + _LEASE_TTL_SECONDS

                # Ignored for exactly as long as the worker refuses to reserve
                # the id again, and no longer. The drain is what sweeps.
                first = scheduler._scheduler._transfers.get("first")
                assert first is not None
                first.deadline = 0.0
                scheduler._scheduler._drain_pending = True
                scheduler._scheduler._drain_push_notifications()
                assert scheduler._scheduler._transfers.get("first") is None
                assert scheduler._scheduler._transfers.get("second") is not None
                assert scheduler._scheduler._transfers.get("third") is third

                # The count is the backstop for a rate that outruns the TTL.
                with patch(
                    "vllm.distributed.ec_transfer.ec_connector."
                    "mooncake.scheduler._MAX_TERMINAL_TRANSFER_RECORDS",
                    1,
                ):
                    scheduler._scheduler._drain_pending = True
                    scheduler._scheduler._drain_push_notifications()
                assert scheduler._scheduler._transfers.get("second") is None
                assert scheduler._scheduler._transfers.get("third") is third
                assert (
                    scheduler._scheduler._consumer_scheduler_metrics[
                        "cancel_records_dropped"
                    ]
                    == 2
                )
            finally:
                scheduler.shutdown()

    def test_item_that_never_arrives_fails_the_request(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """A push that never lands must end the request, not hold it forever.

        The failure is retryable: the caller re-issues, the encode runs again
        and produces a fresh transfer. Deferring instead left the request
        parked until the client timed out.
        """
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
            "push_wait_timeout_s": 0.001,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                with (
                    patch.object(scheduler._scheduler, "_drain_push_notifications"),
                    patch.object(
                        scheduler._scheduler._control_client,
                        "request",
                        return_value=None,
                    ),
                    patch(
                        "vllm.distributed.ec_transfer.ec_connector."
                        "mooncake.scheduler.time.monotonic",
                        side_effect=[10, 10.002],
                    ),
                ):
                    assert not scheduler.ensure_cache_available(request, 0, set())
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert scheduler.take_unavailable_requests() == {request.request_id}
                # Draining clears it: the scheduler acts on each id once.
                assert scheduler.take_unavailable_requests() == set()
                record = scheduler._scheduler._transfers.get(f"{request.request_id}:0")
                assert record is not None
                assert record.state is SchedulerTransferState.UNAVAILABLE
            finally:
                scheduler.shutdown()

    def test_readiness_needs_every_consumer_shard(self, mock_vllm_config_consumer):
        """A sharded consumer is only ready once every rank reports.

        Each rank runs its own control channel and pushes its own readiness
        notifications. Subscribing to the first rank alone strands the other
        ranks' queues and lets a load be scheduled that the last rank cannot
        serve, which only `aggregate`'s `loaded` intersection then catches.
        """
        ports = [19101, 19102, 19103]
        event_ports = {port: 19201 + index for index, port in enumerate(ports)}

        def fake_send(addr: str, request: dict):
            port = int(addr.rsplit(":", 1)[1])
            if request["op"] == "peers":
                return {"ports": ports}
            if request["op"] == "event_port":
                return event_ports[port]
            return {}

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._reservation_zmq_addr = (
                    f"tcp://127.0.0.1:{ports[0]}"
                )
                with patch.object(
                    scheduler._scheduler._control_client,
                    "request",
                    side_effect=fake_send,
                ) as send_control:
                    scheduler._scheduler._drain_push_notifications()

                subscribed = [
                    call.args[0]
                    for call in send_control.call_args_list
                    if call.args[1]["op"] == "event_port"
                ]
                assert len(subscribed) == len(ports)
                assert scheduler._scheduler._event_shard_count == len(ports)

                event = {"transfer_id": "transfer-0"}
                assert not scheduler._scheduler._note_shard_ready(
                    {**event, "shard": ports[0]}
                )
                # The same rank reporting twice is not two ranks.
                assert not scheduler._scheduler._note_shard_ready(
                    {**event, "shard": ports[0]}
                )
                assert not scheduler._scheduler._note_shard_ready(
                    {**event, "shard": ports[1]}
                )
                assert scheduler._scheduler._note_shard_ready(
                    {**event, "shard": ports[2]}
                )
                # Nothing is retained once the transfer is handed on.
                assert "transfer-0" not in scheduler._scheduler._event_ready_shards
            finally:
                scheduler.shutdown()

    def test_evicted_item_is_reloaded_from_the_pool_without_a_transfer(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """The stall at high concurrency: more needs than transfers.

        Requests sharing an image get one transfer each, but a load consumes
        one spec and serves everyone at once. After an eviction the remaining
        requests need the item again with no spec left. The receive pool still
        holds it, so the reload must come from there.
        """
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
            "consumer_buffer_pool_size": 1 << 20,
        }
        first = mock_request_with_3_mm
        first.mm_features = first.mm_features[:1]
        mm_hash = first.mm_features[0].identifier
        first.ec_transfer_params = {
            "ec_items": [{"mm_hash": mm_hash, "transfer_id": "only-transfer"}]
        }
        second = copy.copy(first)
        second.request_id = "second-request"
        second.ec_transfer_params = None

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                scheduler._scheduler._event_inbox.drain = Mock(
                    return_value=[
                        {
                            "mm_hash": mm_hash,
                            "transfer_id": "only-transfer",
                            "ready": True,
                            "reservation_id": "r0",
                            "nbytes": 16,
                            "shape": [4],
                            "dtype": "float32",
                        }
                    ]
                )
                scheduler._scheduler._drain_push_notifications()

                assert not scheduler.ensure_cache_available(first, 0, set())
                meta = scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[])
                )
                assert [spec.transfer_id for spec in meta.loads] == ["only-transfer"]
                scheduler.update_connector_output(
                    SimpleNamespace(
                        ec_connector_worker_meta=ECMooncakeWorkerMetadata(
                            loaded={mm_hash}
                        )
                    )
                )
                record = scheduler._scheduler._transfers.get("only-transfer")
                assert record is not None
                assert record.state is SchedulerTransferState.READY

                # The encoder cache evicts the entry.
                scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[mm_hash])
                )

                # The second request has no transfer of its own, and the only
                # transfer is spent. It must still be served.
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert scheduler.has_cache_item(mm_hash)
                    assert not scheduler.ensure_cache_available(second, 0, set())
                assert record.state is SchedulerTransferState.LOADING
                reload = scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[])
                )
                assert [spec.local for spec in reload.loads] == [True]
            finally:
                scheduler.shutdown()

    def test_reclaimed_item_stops_being_offered_as_resident(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """Residency is a mirror of the worker's pool, not a promise."""
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
            "consumer_buffer_pool_size": 1 << 20,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                spec = ECMooncakeLoadSpec(
                    mm_hash=mm_hash,
                    num_token=0,
                    nbytes=16,
                    shape=(4,),
                    dtype="float32",
                    transfer_id="transfer",
                )
                table = scheduler._scheduler._transfers
                table.observe_ready(spec, time.monotonic() + _LEASE_TTL_SECONDS)
                table.begin_load(mm_hash, 4, "transfer")
                table.take_loads_to_dispatch()
                table.complete_load(mm_hash)
                table.release_ready(mm_hash, time.monotonic())
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert scheduler.has_cache_item(mm_hash)

                scheduler.update_connector_output(
                    SimpleNamespace(
                        ec_connector_worker_meta=ECMooncakeWorkerMetadata(
                            reclaimed={mm_hash}
                        )
                    )
                )
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert not scheduler.has_cache_item(mm_hash)
                assert table.resident_bytes == 0
            finally:
                scheduler.shutdown()

    def test_reclaim_keeps_ready_cache_visible_until_it_is_freed(
        self, mock_vllm_config_consumer
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            table = scheduler._scheduler._transfers
            spec = ECMooncakeLoadSpec(
                mm_hash="hash",
                num_token=0,
                nbytes=16,
                shape=(4,),
                dtype="float32",
                transfer_id="transfer",
            )
            try:
                table.observe_ready(spec, time.monotonic() + _LEASE_TTL_SECONDS)
                table.begin_load("hash", 4, "transfer")
                table.take_loads_to_dispatch()
                table.complete_load("hash")

                scheduler.update_connector_output(
                    SimpleNamespace(
                        ec_connector_worker_meta=ECMooncakeWorkerMetadata(
                            reclaimed={"hash"}
                        )
                    )
                )
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert scheduler.has_cache_item("hash")
                scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=["hash"])
                )
                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert not scheduler.has_cache_item("hash")
            finally:
                scheduler.shutdown()

    def test_retains_new_completion_while_same_hash_is_loading(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
        }
        event = {
            "mm_hash": "hash",
            "transfer_id": "next-transfer",
            "ready": True,
            "reservation_id": "next",
            "nbytes": 64,
            "shape": [2, 8],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            scheduler._scheduler._event_inbox.drain = Mock(return_value=[event])
            current = ECMooncakeLoadSpec(
                mm_hash="hash",
                num_token=0,
                nbytes=64,
                shape=(2, 8),
                dtype="float32",
                transfer_id="current-transfer",
            )
            scheduler._scheduler._transfers.observe_ready(current, time.monotonic() + 1)
            scheduler._scheduler._transfers.begin_load("hash", 2, "current-transfer")

            scheduler._scheduler._drain_push_notifications()

            pending = scheduler._scheduler._transfers.get("next-transfer")
            assert pending is not None and pending.spec is not None
            assert pending.state is SchedulerTransferState.AVAILABLE
            assert pending.spec.reservation_id == "next"

    def test_build_connector_meta_clears_pending(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            mm_hash = mock_request_with_3_mm.mm_features[0].identifier
            load_spec = ECMooncakeLoadSpec(
                mm_hash=mm_hash,
                num_token=0,
                nbytes=32,
                shape=(2, 4),
                dtype="float32",
                transfer_id="transfer",
            )
            scheduler._scheduler._transfers.observe_ready(
                load_spec, time.monotonic() + _LEASE_TTL_SECONDS
            )
            scheduler._scheduler._transfers.begin_load(mm_hash, 100, "transfer")
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )
            assert isinstance(meta, ECMooncakeConnectorMetadata)
            assert len(meta.loads) == 1
            assert meta.loads[0].mm_hash == mm_hash
            assert meta.loads[0].num_token == 100
            assert scheduler._scheduler._transfers.take_loads_to_dispatch() == []
            record = scheduler._scheduler._transfers.get("transfer")
            assert record is not None
            assert record.state is SchedulerTransferState.LOADING

    def test_producer_does_not_build_load_metadata(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            scheduler.update_state_after_alloc(mock_request_with_3_mm, 0)
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )

        assert isinstance(meta, ECMooncakeConnectorMetadata)
        assert meta.loads == []

    def test_producer_builds_push_metadata_after_preprocessing(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        request.ec_transfer_params = {
            "consumer_zmq": "tcp://decode:19019",
            "ec_items": [{"mm_hash": "img_hash_1", "transfer_id": "transfer-1"}],
        }
        mock_vllm_config_producer.model_config.dtype = torch.float32
        mock_vllm_config_producer.model_config.hf_config = None
        mock_vllm_config_producer.model_config.get_inputs_embeds_size.return_value = 16

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            scheduler.update_state_after_alloc(request, 0)
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )

            # The same request remains visible on a later scheduler step, but
            # its worker push metadata must not be emitted a second time.
            assert scheduler.ensure_cache_available(request, 0)
            next_meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )

            scheduler.request_finished(request)

        assert meta.loads == []
        assert meta.pushes == [
            ECMooncakePushSpec(
                mm_hash="img_hash_1",
                nbytes=100 * 16 * 4,
                shape=(100, 16),
                dtype="float32",
                consumer_zmq="tcp://decode:19019",
                transfer_id="transfer-1",
                request_id="test_req_123",
            )
        ]
        assert next_meta.pushes == []
        assert "transfer-1" not in scheduler._scheduler._prepared_push_transfer_ids

    def test_producer_uses_deepstack_encoder_cache_width(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        request = mock_request_with_3_mm
        request.ec_transfer_params = {
            "consumer_zmq": "tcp://decode:19019",
            "ec_items": [{"mm_hash": "img_hash_1", "transfer_id": "transfer-1"}],
        }
        mock_vllm_config_producer.model_config.dtype = torch.bfloat16
        mock_vllm_config_producer.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                out_hidden_size=2560,
                deepstack_visual_indexes=[5, 11, 17],
            )
        )

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            scheduler.update_state_after_alloc(request, 0)
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )

        num_tokens = request.get_num_encoder_embeds(0)
        spec = meta.pushes[0]
        assert spec.shape == (num_tokens, 10240)
        assert spec.nbytes == num_tokens * 10240 * torch.bfloat16.itemsize

    def test_producer_reports_proxy_rewrite_metadata(self, mock_vllm_config_producer):
        feature = SimpleNamespace(
            identifier="image_uuid",
            modality="image",
            data=SimpleNamespace(
                get_data=lambda: {
                    "image_grid_thw": torch.tensor([1, 32, 48]),
                    "pixel_values": torch.ones(2),
                }
            ),
        )
        request = SimpleNamespace(mm_features=[feature])

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            with patch.object(
                scheduler._scheduler,
                "_placeholder_metadata_fields",
                return_value={"image_grid_thw"},
            ):
                delay_free, params = scheduler.request_finished(request)

        assert not delay_free
        assert params == {
            "ec_items": [{"mm_hash": "image_uuid", "image_grid_thw": [1, 32, 48]}]
        }


class TestConsumerReservationManager:
    """Validate Consumer destination ownership and cancellation races."""

    @staticmethod
    def manager():
        pool = Mock()
        pool.lock = threading.RLock()
        pool.acquire_cached.return_value = None
        allocation = memory.MemoryAllocation(0, 64, torch.empty(16))
        pool.try_allocate.return_value = allocation
        pool.reclaim_and_allocate.return_value = None
        return ConsumerReservationManager(pool, 300, 16), pool, allocation

    @staticmethod
    def reserve(manager: ConsumerReservationManager):
        record, write, reused, _ = manager.reserve(
            "transfer", "hash", 64, (16,), "float32", torch.float32
        )
        assert record is not None
        return record, write, reused

    def test_writing_ready_and_repeated_completion_use_one_state_record(self):
        manager, _, _ = self.manager()
        record, write, _ = self.reserve(manager)

        assert write
        assert record.state is ConsumerReservationState.WRITING
        assert manager.status("transfer") is record
        completed = manager.complete("transfer", record.reservation_id)
        repeated = manager.complete("transfer", record.reservation_id)

        assert completed.accepted and completed.became_ready
        assert repeated.accepted and repeated.repeated
        assert record.state is ConsumerReservationState.READY
        with pytest.raises(RuntimeError):
            manager._transition(record, ConsumerReservationState.WRITING)

    def test_writing_cancel_defers_the_only_allocation_release(self):
        manager, pool, allocation = self.manager()
        record, _, _ = self.reserve(manager)

        assert manager.cancel("transfer", "wrong-id") == (
            CancellationOutcome.REJECTED,
            0,
        )
        outcome, dropped = manager.cancel("transfer", record.reservation_id)
        assert outcome is CancellationOutcome.DEFERRED
        assert dropped == 0
        assert record.state is ConsumerReservationState.CANCEL_PENDING
        pool.free.assert_not_called()

        completed = manager.complete("transfer", record.reservation_id)
        repeated = manager.complete("transfer", record.reservation_id)
        assert completed.accepted and completed.discarded
        assert not repeated.accepted
        assert record.state is ConsumerReservationState.CANCELLED
        assert record.allocation is None
        pool.free.assert_called_once_with(allocation)

    def test_ready_expiry_releases_once_and_keeps_a_tombstone(self):
        manager, pool, allocation = self.manager()
        record, _, _ = self.reserve(manager)
        manager.complete("transfer", record.reservation_id)
        record.expires_at = 0

        first_expired, _, _ = manager.expire()
        second_expired, _, _ = manager.expire()

        assert first_expired == 1
        assert second_expired == 0
        assert manager.status("transfer") is None
        assert record.state is ConsumerReservationState.EXPIRED
        pool.free.assert_called_once_with(allocation)

    def test_failed_allocation_returns_deferred_and_tombstone_counts(self):
        manager, pool, _ = self.manager()
        writing, _, _ = self.reserve(manager)
        writing.expires_at = 1.5
        assert manager.cancel("stale", "") == (
            CancellationOutcome.PRE_RESERVED,
            0,
        )
        manager.get("stale").expires_at = 0
        pool.try_allocate.side_effect = [None, None]

        monotonic = (
            "vllm.distributed.ec_transfer.ec_connector.mooncake."
            "reservation.time.monotonic"
        )
        with patch(monotonic, side_effect=[1.0, 2.0]):
            record, write, reused, counts = manager.reserve(
                "new", "new-hash", 64, (16,), "float32", torch.float32
            )

        assert record is None and not write and not reused
        assert counts == (0, 1, 1)
        assert writing.state is ConsumerReservationState.EXPIRE_PENDING
        assert manager.get("stale") is None
        pool.free.assert_not_called()

    def test_expired_writer_refresh_precedes_re_reserve_and_old_completion(self):
        manager, pool, old_allocation = self.manager()
        new_allocation = memory.MemoryAllocation(256, 64, torch.ones(16))
        pool.try_allocate.side_effect = [old_allocation, new_allocation]
        old, _, _ = self.reserve(manager)
        old.expires_at = 0
        _, deferred, _ = manager.expire()

        with pytest.raises(RuntimeError, match="still has an active writer"):
            self.reserve(manager)
        assert old.allocation is old_allocation
        pool.free.assert_not_called()

        (refreshed, dropped) = manager.cancel(
            "transfer", old.reservation_id, abandon=True, refresh=True
        )
        new, write, _ = self.reserve(manager)
        late = manager.complete("transfer", old.reservation_id)

        assert deferred == 1
        assert refreshed is CancellationOutcome.CANCELLED
        assert dropped == 0
        assert write and new.reservation_id != old.reservation_id
        assert new.state is ConsumerReservationState.WRITING
        assert new.allocation is new_allocation
        assert not late.accepted
        pool.free.assert_called_once_with(old_allocation)

    def test_expired_writer_single_slot_is_reused_only_after_refresh_abandon(self):
        transfer_engine = Mock()
        transfer_engine.register_memory.return_value = 0
        pool = ConsumerMemoryPool(256, transfer_engine)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        manager = ConsumerReservationManager(pool, 300, 16)

        old, _, _, _ = manager.reserve(
            "transfer", "hash", 64, (16,), "float32", torch.float32
        )
        assert old is not None
        assert old.allocation is not None
        old_offset = old.allocation.offset
        old.expires_at = 0
        manager.expire()

        with pytest.raises(RuntimeError, match="still has an active writer"):
            manager.reserve("transfer", "hash", 64, (16,), "float32", torch.float32)
        assert pool.try_allocate(64, (16,), torch.float32) is None

        outcome, dropped = manager.cancel(
            "transfer", old.reservation_id, abandon=True, refresh=True
        )
        new, write, _, _ = manager.reserve(
            "transfer", "hash", 64, (16,), "float32", torch.float32
        )
        assert new is not None
        assert outcome is CancellationOutcome.CANCELLED
        assert dropped == 0
        assert write and new.reservation_id != old.reservation_id
        assert old.allocation is None
        assert new.allocation is not None
        assert new.allocation.offset == old_offset == 0

    def test_expired_writer_completion_releases_before_re_reserve(self):
        manager, pool, old_allocation = self.manager()
        new_allocation = memory.MemoryAllocation(256, 64, torch.ones(16))
        pool.try_allocate.side_effect = [old_allocation, new_allocation]
        old, _, _ = self.reserve(manager)
        old.expires_at = 0
        manager.expire()

        completed = manager.complete("transfer", old.reservation_id)
        new, write, _ = self.reserve(manager)

        assert completed.accepted and completed.discarded
        assert old.state is ConsumerReservationState.EXPIRED
        assert old.allocation is None
        assert write and new.allocation is new_allocation
        assert new.reservation_id != old.reservation_id
        pool.free.assert_called_once_with(old_allocation)

    def test_expired_writer_cancel_stays_deferred_until_completion(self):
        manager, pool, allocation = self.manager()
        record, _, _ = self.reserve(manager)
        record.expires_at = 0
        _, deferred, _ = manager.expire()

        cancelled, dropped = manager.cancel("transfer", record.reservation_id)
        assert deferred == 1
        assert cancelled is CancellationOutcome.DEFERRED
        assert dropped == 0
        assert record.state is ConsumerReservationState.EXPIRE_PENDING
        assert record.allocation is allocation
        pool.free.assert_not_called()

        completed = manager.complete("transfer", record.reservation_id)
        assert completed.accepted and completed.discarded
        assert record.state is ConsumerReservationState.EXPIRED
        pool.free.assert_called_once_with(allocation)

    def test_expired_writer_abandon_releases_once(self):
        manager, pool, allocation = self.manager()
        record, _, _ = self.reserve(manager)
        record.expires_at = 0
        manager.expire()

        abandoned, first_dropped = manager.cancel(
            "transfer", record.reservation_id, abandon=True
        )
        repeated, second_dropped = manager.cancel(
            "transfer", record.reservation_id, abandon=True
        )

        assert abandoned is CancellationOutcome.CANCELLED
        assert repeated is CancellationOutcome.PRE_RESERVED
        assert first_dropped == second_dropped == 0
        assert record.state is ConsumerReservationState.CANCELLED
        assert record.allocation is None
        pool.free.assert_called_once_with(allocation)

    def test_cached_take_returns_the_memory_pool_canonical_allocation(self):
        manager, pool, cached = self.manager()
        lease = SimpleNamespace(value=cached)
        canonical = memory.MemoryAllocation(256, 64, torch.ones(16))
        pool.acquire_cached.return_value = lease
        pool.publish.return_value = canonical

        record, write, _ = self.reserve(manager)
        assert not write and record.lease is lease
        taken = manager.take("transfer", "hash")

        assert taken is canonical
        assert record.state is ConsumerReservationState.RESIDENT
        assert record.allocation is None and record.lease is None
        pool.publish.assert_called_once_with("hash", cached, lease)
        pool.free.assert_not_called()
        pool.release_cached.assert_not_called()

    def test_tombstone_indexes_reap_by_prefix_without_scanning_records(self):
        manager, _, _ = self.manager()

        class NoScanDict(dict):
            """Fail if reservation code scans the complete record mapping."""

            def __iter__(self):
                raise AssertionError("record table must not be scanned")

            def items(self):
                raise AssertionError("record table must not be scanned")

            def values(self):
                raise AssertionError("record table must not be scanned")

        manager._tombstone_limit = 3
        manager._records = NoScanDict(manager._records)
        for transfer_id in ("a", "b", "c"):
            assert manager.cancel(transfer_id, "") == (
                CancellationOutcome.PRE_RESERVED,
                0,
            )
        assert manager.cancel("a", "") == (CancellationOutcome.PRE_RESERVED, 0)
        assert manager.cancel("d", "") == (CancellationOutcome.PRE_RESERVED, 1)
        assert list(manager._tombstones) == ["c", "a", "d"]
        assert set(manager._records.keys()) == {"c", "a", "d"}

        manager.get("c").expires_at = 0
        _, _, dropped = manager.expire()
        assert dropped == 1
        assert list(manager._tombstones) == ["a", "d"]
        assert set(manager._records.keys()) == {"a", "d"}
        assert not manager._active_ids

    def test_active_index_tracks_reserve_complete_take_and_expiry(self):
        manager, pool, allocation = self.manager()
        pool.publish.return_value = allocation
        record, _, _ = self.reserve(manager)
        assert list(manager._active_ids) == ["transfer"]
        assert not manager._tombstones

        manager.complete("transfer", record.reservation_id)
        manager.take("transfer", "hash")
        assert manager.get("transfer") is None
        assert not manager._active_ids and not manager._tombstones

        replacement, _, _ = self.reserve(manager)
        manager.complete("transfer", replacement.reservation_id)
        replacement.expires_at = 0
        manager.expire()
        assert not manager._active_ids
        assert list(manager._tombstones) == ["transfer"]
        assert manager.get("transfer").state is ConsumerReservationState.EXPIRED


class TestECMooncakeWorkerTransfer:
    """Validate end-to-end Worker reservation, push, load, and cleanup flows."""

    def test_allocation_retry_accounts_for_expiry_after_outer_sweep(self):
        transfer_engine = Mock()
        transfer_engine.register_memory.return_value = 0
        transfer_engine.local_session.return_value = "local-session"
        pool = ConsumerMemoryPool(256, transfer_engine)
        pool.prepare(torch.device("cpu"), receiving_rank=True, allow_host=True)
        manager = ConsumerReservationManager(pool, 300, 16)
        old, _, _, counts = manager.reserve(
            "old", "old-hash", 64, (16,), "float32", torch.float32
        )
        assert counts == (0, 0, 0)
        assert old is not None
        assert old.allocation is not None
        old_offset = old.allocation.offset
        manager.complete("old", old.reservation_id)
        old.expires_at = 1.5

        worker = object.__new__(ECMooncakeWorker)
        worker._consumer_worker_metrics = Counter()
        worker._reservations = manager
        worker._transfer = transfer_engine
        payload = {
            "transfer_id": "replacement",
            "mm_hash": "replacement-hash",
            "nbytes": 64,
            "shape": [16],
            "dtype": "float32",
        }
        monotonic = (
            "vllm.distributed.ec_transfer.ec_connector.mooncake."
            "reservation.time.monotonic"
        )
        with patch(monotonic, side_effect=[1.0, 1.0, 2.0, 2.0]):
            replacement = worker._reserve_push_destination(payload)

        assert replacement["write"]
        assert manager.get("old").state is ConsumerReservationState.EXPIRED
        assert manager.get("replacement").allocation.offset == old_offset == 0
        assert worker._consumer_worker_metrics["reservations_expired"] == 1
        assert worker._consumer_worker_metrics["cancellations_deferred"] == 0
        assert worker._consumer_worker_metrics["cancel_records_dropped"] == 0

    def test_failed_allocation_still_accounts_inner_expiry(self):
        pool = Mock()
        pool.lock = threading.RLock()
        pool.acquire_cached.return_value = None
        ready_allocation = memory.MemoryAllocation(0, 64, torch.empty(16))
        writing_allocation = memory.MemoryAllocation(64, 64, torch.empty(16))
        pool.try_allocate.side_effect = [ready_allocation, writing_allocation]
        pool.reclaim_and_allocate.return_value = None
        manager = ConsumerReservationManager(pool, 300, 16)
        ready, _, _, _ = manager.reserve(
            "ready", "ready-hash", 64, (16,), "float32", torch.float32
        )
        writing, _, _, _ = manager.reserve(
            "writing", "writing-hash", 64, (16,), "float32", torch.float32
        )
        assert ready is not None and writing is not None
        manager.complete("ready", ready.reservation_id)
        assert manager.cancel("stale", "") == (
            CancellationOutcome.PRE_RESERVED,
            0,
        )
        ready.expires_at = writing.expires_at = 1.5
        manager.get("stale").expires_at = 1.5
        pool.try_allocate.side_effect = [None, None]

        worker = object.__new__(ECMooncakeWorker)
        worker._consumer_worker_metrics = Counter()
        worker._reservations = manager
        payload = {
            "transfer_id": "failed",
            "mm_hash": "failed-hash",
            "nbytes": 64,
            "shape": [16],
            "dtype": "float32",
        }
        monotonic = (
            "vllm.distributed.ec_transfer.ec_connector.mooncake."
            "reservation.time.monotonic"
        )
        with patch(monotonic, side_effect=[1.0, 1.0, 2.0, 2.0, 2.0]):
            with pytest.raises(RuntimeError, match="^EC consumer buffer pool is full$"):
                worker._reserve_push_destination(payload)
            metrics = dict(worker._consumer_worker_metrics)
            worker._expire_push_reservations()

        assert metrics == {
            "reservations_expired": 1,
            "cancellations_deferred": 1,
            "cancel_records_dropped": 1,
        }
        assert dict(worker._consumer_worker_metrics) == metrics
        assert ready.state is ConsumerReservationState.EXPIRED
        assert writing.state is ConsumerReservationState.EXPIRE_PENDING
        assert manager.get("stale") is None
        pool.free.assert_called_once_with(ready_allocation)

    def test_stale_shards_are_abandoned_before_remote_re_reserve(self):
        worker = object.__new__(ECMooncakeWorker)
        worker._control_client = Mock()
        events: list[tuple[str, str] | tuple[str]] = []

        def request(addr, payload):
            events.append(("abandon", payload["reservation_id"]))
            assert payload["abandon"] and payload["refresh"]
            return {"cancelled": True}

        worker._control_client.request.side_effect = request
        replacement = [{"reservation_id": "new"}]

        def reserve_remote(spec):
            events.append(("reserve",))
            return replacement

        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
        )
        shards = [
            {
                "addr": f"tcp://consumer:{19019 + rank}",
                "reservation_id": f"old-{rank}",
                "ready": False,
            }
            for rank in range(2)
        ]

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(worker, "_shard_executor", return_value=executor),
            patch.object(worker, "_reserve_remote", side_effect=reserve_remote),
        ):
            assert worker._refresh_remote_reservations(spec, shards) is replacement
        assert set(events[:2]) == {
            ("abandon", "old-0"),
            ("abandon", "old-1"),
        }
        assert events[2] == ("reserve",)

    def test_cancel_retry_only_retries_failed_shards(self):
        worker = object.__new__(ECMooncakeWorker)
        worker._control_client = Mock()
        attempts: Counter[str] = Counter()

        def request(_addr, payload):
            reservation_id = payload["reservation_id"]
            attempts[reservation_id] += 1
            if reservation_id == "r0" and attempts[reservation_id] == 1:
                raise RuntimeError("transient cancel failure")
            return {"cancelled": True}

        worker._control_client.request.side_effect = request
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
        )
        reservations = [
            {
                "addr": f"tcp://consumer:{19019 + rank}",
                "reservation_id": f"r{rank}",
            }
            for rank in range(3)
        ]

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(worker, "_shard_executor", return_value=executor),
        ):
            worker._retry_cancel_reservations(spec, reservations)

        assert attempts == Counter({"r0": 2, "r1": 1, "r2": 1})

    def test_partial_refresh_cleans_only_the_failed_shard_and_keeps_first_error(
        self,
    ):
        worker = object.__new__(ECMooncakeWorker)
        worker._control_client = Mock()
        calls: list[tuple[str, bool]] = []

        def request(_addr, payload):
            reservation_id = payload["reservation_id"]
            refreshing = payload.get("refresh", False)
            calls.append((reservation_id, refreshing))
            if refreshing and reservation_id == "r0":
                raise RuntimeError("refresh shard failed")
            return {"cancelled": True}

        worker._control_client.request.side_effect = request
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
        )
        reservations = [
            {
                "addr": f"tcp://consumer:{19019 + rank}",
                "reservation_id": f"r{rank}",
                "ready": False,
            }
            for rank in range(3)
        ]

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(worker, "_shard_executor", return_value=executor),
            pytest.raises(RuntimeError, match="^refresh shard failed$"),
        ):
            worker._refresh_remote_reservations(spec, reservations)

        assert Counter(calls) == Counter(
            {("r0", True): 1, ("r1", True): 1, ("r2", True): 1, ("r0", False): 1}
        )

    def test_reservation_snapshot_and_resident_retirement_are_atomic(self):
        worker = object.__new__(ECMooncakeWorker)
        worker._resolve_consumer_rank = Mock()
        worker._is_receiving_rank = True
        worker._transfer = Mock()
        worker._buffer_device = "cpu"
        worker._consumer_memory = ConsumerMemoryPool(256, Mock())
        worker._reservations = ConsumerReservationManager(
            worker._consumer_memory, _LEASE_TTL_SECONDS, 16
        )
        retire_entered = threading.Event()
        finish_retire = threading.Event()
        lock_acquired = threading.Event()

        def retire_stale(*args):
            retire_entered.set()
            assert finish_retire.wait(2)

        def load():
            worker.start_load_caches(ECMooncakeConnectorMetadata(), {})

        def update_reservations():
            with worker._consumer_memory.lock:
                lock_acquired.set()

        with patch.object(
            worker._consumer_memory, "retire_stale", side_effect=retire_stale
        ):
            load_thread = threading.Thread(target=load)
            load_thread.start()
            assert retire_entered.wait(2)
            update_thread = threading.Thread(target=update_reservations)
            update_thread.start()
            assert not lock_acquired.wait(0.05)
            finish_retire.set()
            load_thread.join(2)
            update_thread.join(2)

        assert not load_thread.is_alive()
        assert not update_thread.is_alive()
        assert lock_acquired.is_set()

    def test_control_server_start_failure_closes_server(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096

        with (
            patch_ec_mooncake_deps(),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.mooncake."
                "worker.ConsumerControlServer"
            ) as server_cls,
        ):
            server_cls.return_value.start.side_effect = RuntimeError("bind failed")
            connector = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                with pytest.raises(RuntimeError, match="bind failed"):
                    connector.start_worker_services()
                server_cls.return_value.close.assert_called_once_with()
                assert connector._worker._control_server is None
            finally:
                connector.shutdown()

    def test_abandon_retries_allocation_before_reclaiming_resident(
        self, mock_vllm_config_consumer
    ):
        config = mock_vllm_config_consumer
        config.ec_transfer_config.ec_buffer_device = "cpu"
        config.ec_transfer_config.ec_buffer_size = 512
        config.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 512

        def payload(transfer_id: str, mm_hash: str) -> dict[str, object]:
            return {
                "transfer_id": transfer_id,
                "mm_hash": mm_hash,
                "nbytes": 64,
                "shape": [16],
                "dtype": "float32",
            }

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(config, ECConnectorRole.WORKER)
            worker = connector._worker
            memory_pool = worker._consumer_memory
            try:
                memory_pool.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                resident = memory_pool.try_allocate(64, (16,), torch.float32)
                assert resident is not None
                memory_pool.publish("resident", resident)
                retire_event = MagicMock()
                retire_event.query.return_value = True
                with (
                    patch.object(memory.torch, "Event", return_value=retire_event),
                    patch.object(memory.torch.accelerator, "current_stream"),
                ):
                    memory_pool.retire_stale({}, set())
                old = worker._reserve_push_destination(payload("old", "old"))
                try_allocate = memory_pool.try_allocate
                first_attempt = True

                def abandon_between_attempts(*args):
                    nonlocal first_attempt
                    if first_attempt:
                        first_attempt = False
                        worker._cancel_push("old", old["reservation_id"], abandon=True)
                        return None
                    return try_allocate(*args)

                with patch.object(
                    memory_pool,
                    "try_allocate",
                    side_effect=abandon_between_attempts,
                ):
                    new = worker._reserve_push_destination(payload("new", "new"))

                assert new["dst_ptr"] == old["dst_ptr"]
                assert memory_pool.drain_reclaimed() == set()
            finally:
                connector.shutdown()

    def test_producer_push_state_owns_source_until_every_future_is_terminal(self):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        record, created = manager.reserve(spec, lambda: reservation)
        duplicate, duplicate_created = manager.reserve(spec, lambda: Future())
        assert created
        assert duplicate is record
        assert not duplicate_created
        changed = copy.copy(spec)
        changed.mm_hash = "other"
        with pytest.raises(ValueError, match="changed identity"):
            manager.reserve(changed, lambda: Future())

        source = torch.empty(16)
        manager.bind_source("hash", source, None)
        assert record.source is not None
        reservation.set_result([])
        assert manager.resolve_reservations(record) == []
        assert record.state is ProducerPushState.WAITING_SOURCE
        manager.begin_writing(record)
        manager.begin_notifying([record])

        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("one shard failed"))
        still_writing: Future[None] = Future()
        manager.track_shard_futures([record], [failed, still_writing])
        with pytest.raises(RuntimeError, match="source too early"):
            manager.fail([record], RuntimeError("write failed"))
        assert record.state is ProducerPushState.NOTIFYING
        assert record.source is not None
        assert record.source.tensor is source

        still_writing.set_result(None)
        manager.fail([record], RuntimeError("write failed"))
        assert record.state is ProducerPushState.FAILED
        assert record.source is None
        manager.fail([record], RuntimeError("duplicate failure"))
        with pytest.raises(RuntimeError, match="FAILED to NOTIFYING"):
            manager.begin_notifying([record])

        late, late_created = manager.reserve(spec, lambda: Future())
        assert late is record
        assert not late_created

    def test_reservation_failure_after_source_binding_releases_the_lease(self):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        source = torch.empty(16)
        manager.bind_source("hash", source, None)
        reservation.set_exception(RuntimeError("reserve failed"))

        assert record.state is ProducerPushState.RESERVING

        def run(records) -> None:
            try:
                manager.resolve_reservations(records[0])
            except RuntimeError as exc:
                manager.fail(records, exc)

        with ThreadPoolExecutor(max_workers=1) as executor:
            manager.submit_batches(executor, run, lambda: None)
        assert manager.poll() == [("hash", "reserve failed")]
        assert manager.poll() == []
        assert record.state is ProducerPushState.FAILED
        assert record.source is None

    def test_late_reservation_callback_cannot_replace_refreshed_results(
        self, mock_vllm_config_producer
    ):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        callback_started = threading.Event()
        finish_callback = threading.Event()

        def block_callback(_future) -> None:
            callback_started.set()
            assert finish_callback.wait(2)

        reservation.add_done_callback(block_callback)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        old = [{"addr": "old", "reservation_id": "old"}]
        refreshed = [{"addr": "new", "reservation_id": "new"}]
        setter = threading.Thread(target=reservation.set_result, args=(old,))
        setter.start()
        assert callback_started.wait(2)
        assert manager.resolve_reservations(record) == old
        manager.replace_reservations(record, refreshed)
        finish_callback.set()
        setter.join(2)
        assert not setter.is_alive()
        manager.settle_all([record])
        assert record.reservations == refreshed

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                with patch.object(
                    connector._worker._control_client,
                    "request",
                ) as request:
                    connector._worker._abandon_pushes([record])
                assert request.call_args.args[1]["reservation_id"] == "new"
            finally:
                connector.shutdown()

    def test_producer_hot_paths_do_not_scan_terminal_records(self):
        manager = ProducerPushManager()
        request_ids = set()
        limit = 4096
        with patch.object(producer, "_TERMINAL_LIMIT", limit):
            for index in range(limit + 2):
                reservation: Future[list[dict[str, Any]]] = Future()
                reservation.set_result([])
                request_id = f"request-{index}"
                request_ids.add(request_id)
                spec = ECMooncakePushSpec(
                    mm_hash=f"hash-{index}",
                    nbytes=64,
                    shape=(16,),
                    dtype="float32",
                    consumer_zmq="tcp://consumer:1",
                    transfer_id=f"transfer-{index}",
                    request_id=request_id,
                )
                manager.reserve(spec, lambda r=reservation: r)
            cancelled = manager.cancel_requests(request_ids)
            assert len(cancelled) == limit + 2
            for record in cancelled:
                manager.finish_cancel(record)

            pinned = manager.get("transfer-0")
            assert pinned is not None
            batch_started = threading.Event()
            finish_batch = threading.Event()

            def block_batch(_record) -> None:
                batch_started.set()
                assert finish_batch.wait(2)

            executor = ThreadPoolExecutor(max_workers=1)
            manager.submit_cancel(pinned, executor, block_batch)
            assert batch_started.wait(2)

            class NoScanRecords(OrderedDict):
                """Fail if Producer hot paths scan every transfer record."""

                def __iter__(self):
                    raise AssertionError("record table scanned")

                def items(self):
                    raise AssertionError("record table scanned")

                def values(self):
                    raise AssertionError("record table scanned")

            class NoScanIndex(OrderedDict):
                """Fail if Producer hot paths scan every lifecycle index."""

                def __iter__(self):
                    raise AssertionError("reapable index scanned")

                def items(self):
                    raise AssertionError("reapable index scanned")

                def values(self):
                    raise AssertionError("reapable index scanned")

            manager._records = NoScanRecords(manager._records)
            manager._reapable_terminal_ids = NoScanIndex(manager._reapable_terminal_ids)
            assert manager.pending
            manager.submit_batches(MagicMock(), MagicMock(), MagicMock())
            assert manager.poll() == []
            assert manager.get("transfer-0") is pinned
            assert manager.get("transfer-1") is None
            assert manager.get(f"transfer-{limit}") is not None
            assert len(manager._records) == limit + 1

            finish_batch.set()
            executor.shutdown(wait=True)
            assert manager.poll() == []
            assert manager.get("transfer-0") is pinned
            assert manager.get("transfer-2") is None
            assert len(manager._records) == limit

    def test_producer_push_cancel_handles_pending_and_late_reservations(self):
        manager = ProducerPushManager()
        pending: Future[list[dict[str, Any]]] = Future()
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="pending",
            request_id="request",
        )
        record, _ = manager.reserve(spec, lambda: pending)
        assert manager.pending
        assert manager.cancel_requests({"request"}) == [record]
        assert record.state is ProducerPushState.CANCEL_PENDING
        manager.bind_source("hash", torch.empty(16), None)
        assert record.source is None

        pending.set_result([])
        manager.resolve_reservations(record)
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager.submit_cancel(
                record,
                executor,
                lambda cancelled: manager.finish_cancel(cancelled),
            )
        manager.finish_cancel(record)
        manager.poll()
        assert record.state is ProducerPushState.CANCELLED
        assert not manager.pending
        assert manager.cancel_requests({"request"}) == []

        ready: Future[list[dict[str, Any]]] = Future()
        ready.set_result([])
        later = ECMooncakePushSpec(
            mm_hash="other",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="ready",
            request_id="request-2",
        )
        ready_record, _ = manager.reserve(later, lambda: ready)
        manager.resolve_reservations(ready_record)
        assert manager.cancel_requests({"request-2"}) == [ready_record]
        assert ready_record.state is ProducerPushState.CANCEL_PENDING
        manager.finish_cancel(ready_record)
        assert ready_record.state is ProducerPushState.CANCELLED

    def test_same_source_has_one_lease_per_transfer(self):
        manager = ProducerPushManager()
        source = torch.empty(16)
        records = []
        for transfer_id in ("first", "second"):
            reservation: Future[list[dict[str, Any]]] = Future()
            reservation.set_result([])
            spec = ECMooncakePushSpec(
                mm_hash="hash",
                nbytes=source.nbytes,
                shape=tuple(source.shape),
                dtype="float32",
                consumer_zmq="tcp://consumer:1",
                transfer_id=transfer_id,
            )
            record, _ = manager.reserve(spec, lambda r=reservation: r)
            records.append(record)
        manager.bind_source("hash", source, None)
        assert all(record.source is not None for record in records)
        for record in records:
            manager.resolve_reservations(record)
            manager.begin_writing(record)
            manager.begin_notifying([record])
        manager.complete([records[0]])
        assert records[0].source is None
        assert records[1].source is not None
        manager.complete([records[1]])
        assert records[1].source is None

    def test_shard_submit_failure_waits_before_source_release(
        self, mock_vllm_config_producer
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.empty(16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        slow_started = threading.Event()
        finish_slow = threading.Event()
        slow_finished = threading.Event()
        released_after_slow: list[bool] = []

        def request(addr, payload):
            if payload["op"] == "reserve":
                index = int(addr.rsplit(":", 1)[1])
                return {
                    "reservation_id": f"reservation-{index}",
                    "dst_session": f"session-{index}",
                    "dst_ptr": 1000 + index,
                    "nbytes": source.nbytes,
                    "write": True,
                    "ready": False,
                }
            return {}

        def write(session, sources, destinations, lengths):
            if session == "session-1":
                slow_started.set()
                assert finish_slow.wait(2)
                slow_finished.set()

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = producer._worker
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[spec]))
            try:
                with (
                    patch.object(
                        worker._topology,
                        "shards",
                        return_value=[
                            "tcp://consumer:0",
                            "tcp://consumer:1",
                            "tcp://consumer:2",
                        ],
                    ),
                    patch.object(
                        worker._control_client, "request", side_effect=request
                    ),
                    patch.object(worker._producer_memory, "stage", return_value=None),
                    patch.object(
                        worker._transfer,
                        "acquire_sources",
                        return_value=[source.data_ptr()],
                    ),
                    patch.object(
                        worker._transfer,
                        "release_sources",
                        side_effect=lambda _: released_after_slow.append(
                            slow_finished.is_set()
                        ),
                    ),
                    patch.object(worker._transfer, "write", side_effect=write),
                ):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    record = worker._producer_pushes.get("transfer")
                    assert record is not None
                    record.reservation_futures[0].result(timeout=2)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        submit_count = 0

                        def submit(fn, *args):
                            nonlocal submit_count
                            submit_count += 1
                            if submit_count == 1:
                                return executor.submit(fn, *args)
                            raise RuntimeError("second shard submit failed")

                        shard_executor = MagicMock()
                        shard_executor.submit.side_effect = submit
                        with patch.object(
                            worker,
                            "_shard_executor",
                            return_value=shard_executor,
                        ):
                            assert producer.build_connector_worker_meta().pending_saves
                            assert slow_started.wait(2)
                            assert record.source is not None
                            assert released_after_slow == []
                            finish_slow.set()
                            _wait_for_worker_io(producer)

                record = worker._producer_pushes.get("transfer")
                assert record is not None
                assert record.state is ProducerPushState.FAILED
                assert record.source is None
                assert released_after_slow == [True]
            finally:
                finish_slow.set()
                producer.shutdown()

    def test_reserve_failure_waits_for_every_started_shard(
        self, mock_vllm_config_producer
    ):
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        slow_started = threading.Event()
        finish_slow = threading.Event()
        finished = threading.Event()
        errors: list[Exception] = []

        def reserve_one(addr, _spec):
            if addr.endswith(":0"):
                raise RuntimeError("first shard failed")
            if addr.endswith(":2"):
                slow_started.set()
                assert finish_slow.wait(2)
            return {"addr": addr}

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = producer._worker

            def reserve() -> None:
                try:
                    worker._reserve_remote(spec)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            try:
                with (
                    patch.object(
                        worker._topology,
                        "shards",
                        return_value=["shard:0", "shard:1", "shard:2"],
                    ),
                    patch.object(worker, "_reserve_one", side_effect=reserve_one),
                ):
                    thread = threading.Thread(target=reserve)
                    thread.start()
                    assert slow_started.wait(2)
                    assert not finished.wait(0.05)
                    finish_slow.set()
                    thread.join(2)
                assert not thread.is_alive()
                assert len(errors) == 1
                assert str(errors[0]) == "first shard failed"
            finally:
                finish_slow.set()
                producer.shutdown()

    def test_reserve_submit_failure_drains_started_shards(
        self, mock_vllm_config_producer
    ):
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:1",
            transfer_id="transfer",
        )
        slow_started = threading.Event()
        finish_slow = threading.Event()
        finished = threading.Event()
        errors: list[Exception] = []

        def reserve_one(addr, _spec):
            slow_started.set()
            assert finish_slow.wait(2)
            return {"addr": addr}

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = connector._worker

            def reserve() -> None:
                try:
                    worker._reserve_remote(spec)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    submit_count = 0

                    def submit(fn, *args):
                        nonlocal submit_count
                        submit_count += 1
                        if submit_count == 1:
                            return executor.submit(fn, *args)
                        raise RuntimeError("second shard submit failed")

                    shard_executor = MagicMock()
                    shard_executor.submit.side_effect = submit
                    with (
                        patch.object(
                            worker._topology,
                            "shards",
                            return_value=["shard:0", "shard:1", "shard:2"],
                        ),
                        patch.object(worker, "_reserve_one", side_effect=reserve_one),
                        patch.object(
                            worker,
                            "_shard_executor",
                            return_value=shard_executor,
                        ),
                    ):
                        thread = threading.Thread(target=reserve)
                        thread.start()
                        assert slow_started.wait(2)
                        assert not finished.wait(0.05)
                        finish_slow.set()
                        thread.join(2)
                assert not thread.is_alive()
                assert len(errors) == 1
                assert str(errors[0]) == "second shard submit failed"
            finally:
                finish_slow.set()
                connector.shutdown()

    @pytest.mark.parametrize("source_before_failure", [False, True])
    def test_partial_reserve_is_compensated_before_its_future_fails(
        self, mock_vllm_config_producer, source_before_failure
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.empty(16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq="tcp://consumer:0",
            transfer_id="transfer",
        )
        cancel_attempts = 0

        def reserve_one(addr, _spec):
            if addr.endswith(":1"):
                raise RuntimeError("reserve shard failed")
            return {"addr": addr, "reservation_id": "partial-r0"}

        def request(_addr, payload):
            nonlocal cancel_attempts
            assert payload == {
                "op": "cancel",
                "transfer_id": "transfer",
                "reservation_id": "partial-r0",
                "abandon": True,
            }
            cancel_attempts += 1
            if cancel_attempts == 1:
                raise RuntimeError("transient cleanup failure")
            return {"cancelled": True}

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = connector._worker
            connector.bind_connector_metadata(
                ECMooncakeConnectorMetadata(pushes=[spec])
            )
            try:
                with (
                    patch.object(
                        worker._topology,
                        "shards",
                        return_value=["tcp://consumer:0", "tcp://consumer:1"],
                    ),
                    patch.object(worker, "_reserve_one", side_effect=reserve_one),
                    patch.object(
                        worker._control_client, "request", side_effect=request
                    ),
                ):
                    connector.start_save_caches(
                        encoder_cache={"hash": source}
                        if source_before_failure
                        else None
                    )
                    record = worker._producer_pushes.get("transfer")
                    assert record is not None
                    with pytest.raises(
                        RuntimeError, match="^reserve shard failed$"
                    ) as e:
                        record.reservation_futures[0].result(timeout=2)
                    assert e.value.partial_reservations == [
                        {
                            "addr": "tcp://consumer:0",
                            "reservation_id": "partial-r0",
                        }
                    ]
                    assert cancel_attempts == 2

                    if source_before_failure:
                        assert record.source is not None
                        connector.build_connector_worker_meta()
                        _wait_for_worker_io(connector)
                    else:
                        assert record.state is ProducerPushState.FAILED
                        connector.save_caches({"hash": source}, "hash")
                    assert record.state is ProducerPushState.FAILED
                    assert record.source is None
            finally:
                connector.shutdown()

    def test_partial_complete_abandons_all_shards_before_releasing_source(
        self, mock_vllm_config_producer
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.empty(16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq="tcp://consumer:0",
            transfer_id="transfer",
        )
        reservations = [
            {
                "addr": f"tcp://consumer:{rank}",
                "reservation_id": f"r{rank}",
                "dst_session": f"session-{rank}",
                "dst_ptr": 1000 + rank,
                "nbytes": source.nbytes,
                "write": True,
                "ready": False,
            }
            for rank in range(2)
        ]
        slow_started = threading.Event()
        finish_slow = threading.Event()
        cancelled: list[str] = []

        def request(addr, payload):
            if payload["op"] == "complete_batch":
                if addr.endswith(":0"):
                    raise RuntimeError("complete shard failed")
                slow_started.set()
                assert finish_slow.wait(2)
                return {"items": [{"completed": True}]}
            assert payload["op"] == "cancel" and payload["abandon"]
            cancelled.append(payload["reservation_id"])
            return {"cancelled": True}

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = connector._worker
            connector.bind_connector_metadata(
                ECMooncakeConnectorMetadata(pushes=[spec])
            )
            try:
                with (
                    patch.object(worker, "_reserve_remote", return_value=reservations),
                    patch.object(
                        worker._control_client, "request", side_effect=request
                    ),
                    patch.object(worker._producer_memory, "stage", return_value=None),
                    patch.object(
                        worker._transfer,
                        "acquire_sources",
                        return_value=[source.data_ptr()],
                    ),
                    patch.object(worker._transfer, "release_sources"),
                    patch.object(worker._transfer, "write"),
                ):
                    connector.start_save_caches(encoder_cache={"hash": source})
                    assert connector.build_connector_worker_meta().pending_saves
                    record = worker._producer_pushes.get("transfer")
                    assert record is not None and record.batch_future is not None
                    assert slow_started.wait(2)
                    assert record.source is not None
                    assert not record.batch_future.done()
                    finish_slow.set()
                    record.batch_future.result(timeout=2)
                    connector.build_connector_worker_meta()

                assert Counter(cancelled) == Counter({"r0": 1, "r1": 1})
                assert record.state is ProducerPushState.FAILED
                assert record.source is None
                assert record.error == "complete shard failed"
                assert all(future.done() for future in record.shard_futures)
            finally:
                finish_slow.set()
                connector.shutdown()

    @pytest.mark.parametrize("permanent_failure", [False, True])
    def test_orphan_cancel_is_bounded_retryable_and_skips_cached_shards(
        self, mock_vllm_config_producer, permanent_failure
    ):
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:0",
            transfer_id="transfer",
            request_id="request",
        )
        reservation: Future[list[dict[str, Any]]] = Future()
        reservation.set_result(
            [
                {
                    "addr": "tcp://consumer:0",
                    "reservation_id": "active",
                },
                {
                    "addr": "tcp://consumer:1",
                    "reservation_id": "cached",
                    "cached": True,
                },
                {
                    "addr": "tcp://consumer:2",
                    "reservation_id": "cancelled",
                    "cancelled": True,
                },
            ]
        )
        attempts: Counter[str] = Counter()

        def request(_addr, payload):
            reservation_id = payload["reservation_id"]
            attempts[reservation_id] += 1
            assert reservation_id == "active"
            if permanent_failure or attempts[reservation_id] == 1:
                raise RuntimeError("orphan cancel failed")
            return {"cancelled": True}

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = connector._worker
            record, _ = worker._producer_pushes.reserve(spec, lambda: reservation)
            worker._producer_pushes.resolve_reservations(record)
            assert worker._producer_pushes.cancel_requests({"request"}) == [record]
            try:
                with patch.object(
                    worker._control_client, "request", side_effect=request
                ):
                    worker._producer_pushes.submit_cancel(
                        record,
                        worker._io_executor,
                        worker._cancel_orphaned_reservation,
                    )
                    assert record.batch_future is not None
                    if permanent_failure:
                        with pytest.raises(
                            RuntimeError, match="^orphan cancel failed$"
                        ):
                            record.batch_future.result(timeout=2)
                    else:
                        record.batch_future.result(timeout=2)
                    failures = worker._producer_pushes.poll()

                assert attempts == Counter({"active": 2})
                assert record.state is ProducerPushState.CANCELLED
                assert failures == (
                    [("hash", "orphan cancel failed")] if permanent_failure else []
                )
            finally:
                connector.shutdown()

    def test_source_contract_checks_shape_dtype_contiguity_and_size(self):
        tensors_and_specs = [
            (torch.empty(2, 8), (16,), "float32", 64, "shape"),
            (torch.empty(16, dtype=torch.float16), (16,), "float32", 32, "dtype"),
            (torch.empty(4, 4).t(), (4, 4), "float32", 64, "contiguous"),
            (torch.empty(16), (16,), "float32", 65, "size"),
        ]
        for index, (tensor, shape, dtype, nbytes, message) in enumerate(
            tensors_and_specs
        ):
            spec = ECMooncakePushSpec(
                mm_hash=f"hash-{index}",
                nbytes=nbytes,
                shape=shape,
                dtype=dtype,
                consumer_zmq="tcp://consumer:0",
                transfer_id=f"transfer-{index}",
            )
            reservation: Future[list[dict[str, Any]]] = Future()
            reservation.set_result([])
            manager = ProducerPushManager()
            record, _ = manager.reserve(spec, lambda future=reservation: future)
            manager.bind_source(spec.mm_hash, tensor, None)
            with pytest.raises(ValueError, match=message):
                ECMooncakeWorker._validate_push_source(record)

    def test_invalid_source_fails_asynchronously_before_staging(
        self, mock_vllm_config_producer
    ):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.empty(2, 8)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:0",
            transfer_id="transfer",
        )

        def request(_addr, payload):
            if payload["op"] == "reserve":
                return {
                    "reservation_id": "reservation",
                    "dst_session": "session",
                    "dst_ptr": 1000,
                    "nbytes": source.nbytes,
                    "write": True,
                    "ready": False,
                }
            assert payload["op"] == "cancel"
            return {"cancelled": True}

        with patch_ec_mooncake_deps():
            connector = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            worker = connector._worker
            connector.bind_connector_metadata(
                ECMooncakeConnectorMetadata(pushes=[spec])
            )
            try:
                with (
                    patch.object(
                        worker._topology,
                        "shards",
                        return_value=["tcp://consumer:0"],
                    ),
                    patch.object(
                        worker._control_client, "request", side_effect=request
                    ),
                    patch.object(worker._producer_memory, "stage") as stage,
                    patch.object(worker._transfer, "acquire_sources") as register,
                ):
                    connector.start_save_caches(encoder_cache={"hash": source})
                    assert connector.build_connector_worker_meta().pending_saves
                    record = worker._producer_pushes.get("transfer")
                    assert record is not None and record.batch_future is not None
                    record.batch_future.result(timeout=2)
                    connector.build_connector_worker_meta()

                assert record.state is ProducerPushState.FAILED
                assert record.source is None
                assert record.error == "EC source shape mismatch for mm_hash=hash"
                stage.assert_not_called()
                register.assert_not_called()
            finally:
                connector.shutdown()

    def test_batches_pushes_from_one_model_step(self, mock_vllm_config_producer):
        port = _find_free_port()
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        sources = {
            "first": torch.randn(4, 16),
            "second": torch.randn(8, 16),
        }
        pushes = [
            ECMooncakePushSpec(
                mm_hash=mm_hash,
                nbytes=tensor.nbytes,
                shape=tuple(tensor.shape),
                dtype="float32",
                consumer_zmq=f"tcp://127.0.0.1:{port}",
                transfer_id=f"transfer-{mm_hash}",
            )
            for mm_hash, tensor in sources.items()
        ]

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=pushes))
            try:
                producer.start_save_caches(encoder_cache=sources)
                _wait_for_worker_io(producer)

                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert len(engine.transfer_calls) == 1
                assert sorted(engine.transfer_calls[0]) == sorted(
                    tensor.nbytes for tensor in sources.values()
                )
                assert all(
                    reservation.state is ConsumerReservationState.READY
                    for reservation in consumer._worker._reservations.active_records()
                )
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_push_reserves_before_encoder_output_is_saved(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.randn(4, 16)
        push = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=f"tcp://127.0.0.1:{port}",
            transfer_id="transfer-1",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            consumer.start_worker_services()
            scheduler = ECMooncakeConnector(consumer_cfg, ECConnectorRole.SCHEDULER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[push]))
            try:
                producer.start_save_caches(encoder_cache={})
                push_record = producer._worker._producer_pushes.get("transfer-1")
                assert push_record is not None
                reservation = push_record.reservation_futures[0]
                shards = reservation.result(timeout=2)
                # One reservation per consumer shard; this consumer is single.
                assert len(shards) == 1
                reservation_data = shards[0]
                assert reservation_data["nbytes"] == source.nbytes
                old_reservation_id = reservation_data["reservation_id"]
                reservation_data["_received_at"] -= _LEASE_TTL_SECONDS
                consumer._worker._reservations.get("transfer-1").expires_at = 0
                with patch.object(
                    scheduler._scheduler._control_client,
                    "request",
                    wraps=scheduler._scheduler._control_client.request,
                ) as send_control:
                    assert not scheduler.has_cache_item("hash")
                    assert not scheduler.has_cache_item("hash")
                    # The channel is built once, not per call: the roster is
                    # fetched and every shard subscribed to on the first one.
                    assert [call.args[1] for call in send_control.call_args_list] == [
                        {"op": "peers"},
                        {"op": "event_port"},
                    ]
                    assert consumer._worker._reservations.status("transfer-1")

                    producer.save_caches({"hash": source}, "hash")
                    _wait_for_worker_io(producer)
                    assert (
                        consumer._worker._reservations.get("transfer-1").reservation_id
                        != old_reservation_id
                    )
                    deadline = time.monotonic() + 2
                    while not scheduler.has_cache_item("hash"):
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    # Still just the two setup requests: polling for readiness
                    # must not re-open the channel.
                    assert send_control.call_count == 2
                record = scheduler._scheduler._transfers.get("transfer-1")
                assert record is not None and record.spec is not None
                load = record.spec
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[load])
                )
                loaded: dict[str, torch.Tensor] = {}
                consumer.start_load_caches(loaded)
                first_meta = consumer.build_connector_worker_meta()
                assert first_meta.loaded == {"hash"}
                assert torch.equal(loaded["hash"], source)
                consumer_engine = consumer._worker._transfer._engine
                assert isinstance(consumer_engine, CopyingFakeTransferEngine)
                assert consumer_engine.transfer_calls == []
            finally:
                producer.shutdown()
                scheduler.shutdown()
                consumer.shutdown()

    def test_finished_request_cancels_unbound_reservation(
        self, mock_vllm_config_producer
    ):
        """A pre-reservation without an encoder tensor must not outlive its request."""
        port = _find_free_port()
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.randn(4, 16)
        push = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=f"tcp://127.0.0.1:{port}",
            transfer_id="transfer-1",
            request_id="request-1",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[push]))
            try:
                producer.start_save_caches(encoder_cache={})
                push_record = producer._worker._producer_pushes.get("transfer-1")
                assert push_record is not None
                reservation = push_record.reservation_futures[0]
                reservation.result(timeout=2)
                assert consumer._worker._reservations.status("transfer-1")

                producer.get_finished({"request-1"})
                _wait_for_worker_io(producer)
                push_record = producer._worker._producer_pushes.get("transfer-1")
                assert push_record is not None
                assert push_record.state is ProducerPushState.CANCELLED
                assert consumer._worker._reservations.status("transfer-1") is None
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_duplicate_pushes_share_one_transfer_per_reservation(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.randn(4, 16)
        push = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=f"tcp://127.0.0.1:{port}",
            transfer_id="transfer-1",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            scheduler = ECMooncakeConnector(consumer_cfg, ECConnectorRole.SCHEDULER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(
                ECMooncakeConnectorMetadata(pushes=[push, push])
            )
            try:
                producer.start_save_caches(encoder_cache={"hash": source})
                _wait_for_worker_io(producer)

                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert engine.transfer_calls == [[source.nbytes]]
                reservation = consumer._worker._reservations.get("transfer-1")
                assert reservation.state is ConsumerReservationState.READY
                assert (
                    consumer._worker._consumer_worker_metrics["completions_accepted"]
                    == 1
                )
                assert (
                    consumer._worker._consumer_worker_metrics["completions_repeated"]
                    == 0
                )

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                record = scheduler._scheduler._transfers.get("transfer-1")
                assert record is not None and record.spec is not None
                load = record.spec
                load.num_token = 4
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[load])
                )
                loaded: dict[str, torch.Tensor] = {}
                consumer.start_load_caches(loaded)

                cached_push = ECMooncakePushSpec(
                    mm_hash=push.mm_hash,
                    nbytes=push.nbytes,
                    shape=push.shape,
                    dtype=push.dtype,
                    consumer_zmq=push.consumer_zmq,
                    transfer_id="transfer-2",
                )
                producer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(pushes=[cached_push])
                )
                producer.start_save_caches(encoder_cache={"hash": source})
                _wait_for_worker_io(producer)
                assert engine.transfer_calls == [[source.nbytes]]
                cached = consumer._worker._reservations.get("transfer-2")
                assert cached is not None
                assert cached.state is ConsumerReservationState.READY
                assert cached.lease is not None
                assert (
                    consumer._worker._consumer_worker_metrics["reservations_cached"]
                    == 1
                )

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                record = scheduler._scheduler._transfers.get("transfer-2")
                assert record is not None and record.spec is not None
                cached_load = record.spec
                cached_load.num_token = 4
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[cached_load])
                )
                consumer.start_load_caches(loaded)
                cached_meta = consumer.build_connector_worker_meta()
                assert cached_meta.loaded == {"hash"}
                assert consumer._worker._reservations.status("transfer-2") is None
                assert torch.equal(loaded["hash"], source)
            finally:
                producer.shutdown()
                scheduler.shutdown()
                consumer.shutdown()

    def test_retired_item_reserved_again_still_serves_a_local_load(
        self, mock_vllm_config_producer
    ):
        """A push for a retired item makes it live, not gone.

        Reusing the allocation for a new reservation takes it out of the
        reclaim order. Looking the load up there instead of in the residency
        map failed it, and the request fell back to waiting for a transfer.
        """
        port = _find_free_port()
        cfg = Mock(spec=VllmConfig)
        cfg.parallel_config = mock_vllm_config_producer.parallel_config
        cfg.model_config = Mock()
        cfg.ec_transfer_config = Mock()
        cfg.ec_transfer_config.is_ec_producer = False
        cfg.ec_transfer_config.is_ec_consumer = True
        cfg.ec_transfer_config.ec_buffer_device = "cpu"
        cfg.ec_transfer_config.ec_buffer_size = 4096
        cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        spec = ECMooncakeLoadSpec(
            mm_hash="hash",
            num_token=0,
            nbytes=64,
            shape=(4, 4),
            dtype="float32",
            local=True,
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(cfg, ECConnectorRole.WORKER)
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                allocation = consumer._worker._consumer_memory.try_allocate(
                    spec.nbytes, spec.shape, torch.float32
                )
                assert allocation is not None
                tensor = allocation.tensor
                consumer._worker._consumer_memory.publish("hash", allocation)
                retire_event = MagicMock()
                retire_event.query.return_value = True
                with (
                    patch.object(memory.torch, "Event", return_value=retire_event),
                    patch.object(memory.torch.accelerator, "current_stream"),
                ):
                    consumer._worker._consumer_memory.retire_stale({}, set())

                # A later push reserves the retired copy instead of transferring.
                consumer._worker._reserve_push_destination(
                    {
                        "transfer_id": "t1",
                        "mm_hash": "hash",
                        "nbytes": spec.nbytes,
                        "shape": list(spec.shape),
                        "dtype": spec.dtype,
                    }
                )
                assert consumer._worker._consumer_memory.stats()[2] == 0

                assert (
                    consumer._worker._consumer_memory.take_resident(
                        spec.mm_hash, spec.shape, spec.dtype
                    )
                    is tensor
                )
            finally:
                consumer.shutdown()

    def test_cached_take_uses_newer_same_hash_canonical(
        self, mock_vllm_config_consumer
    ):
        config = mock_vllm_config_consumer
        config.ec_transfer_config.ec_buffer_device = "cpu"
        config.ec_transfer_config.ec_buffer_size = 768
        config.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 768
        shape = (16,)

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(config, ECConnectorRole.WORKER)
            worker = consumer._worker
            memory_pool = worker._consumer_memory
            try:
                memory_pool.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                first = memory_pool.try_allocate(64, shape, torch.float32)
                replacement = memory_pool.try_allocate(64, shape, torch.float32)
                assert first is not None and replacement is not None
                memory_pool.publish("hash", first)
                worker._reserve_push_destination(
                    {
                        "transfer_id": "cached",
                        "mm_hash": "hash",
                        "nbytes": 64,
                        "shape": list(shape),
                        "dtype": "float32",
                    }
                )
                memory_pool.publish("hash", replacement)
                memory_pool.retire_stale({}, {"hash"})
                spec = ECMooncakeLoadSpec(
                    mm_hash="hash",
                    num_token=1,
                    nbytes=64,
                    shape=shape,
                    dtype="float32",
                    pushed=True,
                    transfer_id="cached",
                )

                tensor, allocation = worker._take_pushed_tensor(spec)

                assert allocation is replacement
                assert tensor is replacement.tensor
                reused = memory_pool.try_allocate(64, shape, torch.float32)
                assert reused is not None
                assert reused.offset == first.offset
            finally:
                consumer.shutdown()

    def test_push_reaches_every_consumer_shard(self, mock_vllm_config_producer):
        """A sharded consumer gets one copy per rank, from one source.

        Each rank gathers from its own encoder cache, so the push has to land
        on all of them; the bytes are identical, so staging and registration
        happen once however many destinations there are.
        """
        shard_ports = [_find_free_port() for _ in range(3)]
        base = f"tcp://127.0.0.1:{shard_ports[0]}"
        source = torch.randn(4, 16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=base,
            transfer_id="transfer-0",
        )
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        destinations = [torch.zeros_like(source) for _ in shard_ports]

        def fake_send(addr: str, request: dict):
            if request["op"] == "peers":
                return {"ports": shard_ports}
            index = shard_ports.index(int(addr.rsplit(":", 1)[1]))
            if request["op"] == "reserve":
                return {
                    "reservation_id": f"r{index}",
                    "dst_session": f"session-{index}",
                    "dst_ptr": destinations[index].data_ptr(),
                    "nbytes": source.nbytes,
                    "write": True,
                    "ready": True,
                    "addr": addr,
                }
            if request["op"] == "complete_batch":
                return {"items": [{"completed": True} for _ in request["items"]]}
            return {}

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[spec]))
            try:
                with patch.object(
                    producer._worker._control_client,
                    "request",
                    side_effect=fake_send,
                ):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    _wait_for_worker_io(producer)

                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                # One write per rank, and every rank got the same bytes.
                assert len(engine.transfer_calls) == len(shard_ports)
                for destination in destinations:
                    assert torch.equal(destination, source)
                # The source is staged once, not once per destination.
                assert len(engine.register_calls) == 1
            finally:
                producer.shutdown()

    def test_pushes_stage_through_the_registered_pool(self, mock_vllm_config_producer):
        """Repeated content must not register overlapping source storage."""
        port = _find_free_port()
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.randn(4, 16)
        pushes = [
            ECMooncakePushSpec(
                mm_hash="hash",
                nbytes=source.nbytes,
                shape=tuple(source.shape),
                dtype="float32",
                consumer_zmq=f"tcp://127.0.0.1:{port}",
                transfer_id=f"transfer-{index}",
            )
            for index in range(2)
        ]

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=pushes))
            try:
                producer.start_save_caches(encoder_cache={"hash": source})
                _wait_for_worker_io(producer)

                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                # The staging pool is registered once; a transfer registers
                # nothing of its own.
                pool = producer._worker._producer_memory.tensor
                assert pool is not None
                assert engine.register_calls == [[pool.data_ptr()]]
                assert engine.batch_unregister_calls == []
                assert engine.transfer_calls == [[source.nbytes, source.nbytes]]
                assert all(
                    reservation.state is ConsumerReservationState.READY
                    for reservation in consumer._worker._reservations.active_records()
                )
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_push_falls_back_to_per_tensor_registration_without_a_pool(
        self, mock_vllm_config_producer
    ):
        """A pool that cannot be created must not break pushes."""
        port = _find_free_port()
        consumer_cfg = self._push_harness_config(mock_vllm_config_producer, port)
        source = torch.randn(4, 16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=f"tcp://127.0.0.1:{port}",
            transfer_id="transfer",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[spec]))
            try:
                with patch(
                    "vllm.distributed.ec_transfer.ec_connector."
                    "mooncake.memory.torch.empty",
                    side_effect=torch.OutOfMemoryError,
                ):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    _wait_for_worker_io(producer)
                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert producer._worker._producer_memory.tensor is None
                assert engine.register_calls == [[source.data_ptr()]]
                assert engine.batch_unregister_calls == [[source.data_ptr()]]
                assert engine.transfer_calls == [[source.nbytes]]
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_concurrent_pushes_hold_source_registration_until_last_release(
        self, mock_vllm_config_producer
    ):
        """Concurrent transfers share one MR until every user releases it."""
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        source = torch.randn(4, 16)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                first = producer._worker._transfer.acquire_sources([source])
                second = producer._worker._transfer.acquire_sources([source])
                engine = producer._worker._transfer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert len(engine.register_calls) == 1

                producer._worker._transfer.release_sources(first)
                assert engine.batch_unregister_calls == []
                producer._worker._transfer.release_sources(second)
                assert engine.batch_unregister_calls == [first]
            finally:
                producer.shutdown()

    def _push_harness_config(self, producer_cfg, port: int):
        consumer_cfg = Mock(spec=VllmConfig)
        consumer_cfg.parallel_config = producer_cfg.parallel_config
        consumer_cfg.model_config = Mock()
        consumer_cfg.ec_transfer_config = Mock()
        consumer_cfg.ec_transfer_config.is_ec_producer = False
        consumer_cfg.ec_transfer_config.is_ec_consumer = True
        consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        consumer_cfg.ec_transfer_config.ec_buffer_size = 4096
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": port,
            "consumer_buffer_pool_size": 4096,
        }
        producer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
        return consumer_cfg

    def test_batch_completion_sends_one_control_message(
        self, mock_vllm_config_producer
    ):
        """Completion is per batch, not per item: k items used to cost k RTTs."""
        port = _find_free_port()
        consumer_cfg = self._push_harness_config(mock_vllm_config_producer, port)
        sources = {"a": torch.randn(4, 16), "b": torch.randn(4, 16)}
        pushes = [
            ECMooncakePushSpec(
                mm_hash=mm_hash,
                nbytes=source.nbytes,
                shape=tuple(source.shape),
                dtype="float32",
                consumer_zmq=f"tcp://127.0.0.1:{port}",
                transfer_id=f"transfer-{mm_hash}",
            )
            for mm_hash, source in sources.items()
        ]

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=pushes))
            try:
                with patch.object(
                    producer._worker._control_client,
                    "request",
                    wraps=producer._worker._control_client.request,
                ) as send_control:
                    producer.start_save_caches(encoder_cache=sources)
                    _wait_for_worker_io(producer)
                ops = [call.args[1]["op"] for call in send_control.call_args_list]
                assert ops.count("complete_batch") == 1
                assert "complete" not in ops
                assert all(
                    reservation.state is ConsumerReservationState.READY
                    for reservation in consumer._worker._reservations.active_records()
                )
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_failed_push_is_reported_not_raised(self, mock_vllm_config_producer):
        """A transfer failure must not surface as a fatal engine error."""
        port = _find_free_port()
        consumer_cfg = self._push_harness_config(mock_vllm_config_producer, port)
        source = torch.randn(4, 16)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=source.nbytes,
            shape=tuple(source.shape),
            dtype="float32",
            consumer_zmq=f"tcp://127.0.0.1:{port}",
            transfer_id="transfer",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            consumer.start_worker_services()
            producer.bind_connector_metadata(ECMooncakeConnectorMetadata(pushes=[spec]))
            try:
                producer._worker._transfer.ensure_ready()
                engine = producer._worker._transfer._engine
                with patch.object(engine, "batch_transfer_sync_write", return_value=1):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    # No raise: the batch reports itself and gives up the
                    # consumer-side reservation.
                    _wait_for_worker_io(producer)
                assert consumer._worker._reservations.status("transfer") is None
            finally:
                producer.shutdown()
                consumer.shutdown()

    def test_complete_is_idempotent_without_republishing(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                reservation = consumer._worker._reserve_push_destination(
                    {
                        "mm_hash": "hash",
                        "transfer_id": "transfer-1",
                        "nbytes": 64,
                        "shape": [4, 4],
                        "dtype": "float32",
                    }
                )
                reservation_id = reservation["reservation_id"]

                first = consumer._worker._complete_push("transfer-1", reservation_id)
                repeated = consumer._worker._complete_push("transfer-1", reservation_id)

                assert first.accepted and first.became_ready
                assert repeated.accepted and not repeated.became_ready
            finally:
                consumer.shutdown()

    def test_cancel_pending_repeat_reserve_is_terminal_without_releasing(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096
        payload = {
            "mm_hash": "hash",
            "transfer_id": "transfer",
            "nbytes": 64,
            "shape": [4, 4],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            memory_pool = consumer._worker._consumer_memory
            try:
                memory_pool.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                first = consumer._worker._reserve_push_destination(payload)
                record = consumer._worker._reservations.get("transfer")
                assert record is not None and record.allocation is not None
                allocation = record.allocation
                with patch.object(memory_pool, "free", wraps=memory_pool.free) as free:
                    assert consumer._worker._cancel_push(
                        "transfer", first["reservation_id"]
                    )
                    repeated = consumer._worker._reserve_push_destination(payload)

                    assert repeated["cancelled"]
                    assert not repeated["write"] and not repeated["ready"]
                    assert record.state is ConsumerReservationState.CANCEL_PENDING
                    assert record.allocation is allocation
                    free.assert_not_called()

                    completed = consumer._worker._complete_push(
                        "transfer", first["reservation_id"]
                    )
                    assert completed.accepted and not completed.became_ready
                    assert record.state is ConsumerReservationState.CANCELLED
                    assert record.allocation is None
                    free.assert_called_once_with(allocation)
            finally:
                consumer.shutdown()

    def test_same_hash_transfers_have_independent_lifecycles(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096

        def payload(transfer_id: str) -> dict:
            return {
                "mm_hash": "shared-hash",
                "transfer_id": transfer_id,
                "nbytes": 64,
                "shape": [4, 4],
                "dtype": "float32",
            }

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                first = consumer._worker._reserve_push_destination(payload("first"))
                second = consumer._worker._reserve_push_destination(payload("second"))

                consumer._worker._complete_push("first", first["reservation_id"])
                assert (
                    consumer._worker._reservations.get("first").state
                    is ConsumerReservationState.READY
                )
                assert (
                    consumer._worker._reservations.get("second").state
                    is ConsumerReservationState.WRITING
                )

                assert consumer._worker._cancel_push("first", first["reservation_id"])
                assert consumer._worker._reservations.status("first") is None
                assert consumer._worker._reservations.status("second")
                assert consumer._worker._complete_push(
                    "second", second["reservation_id"]
                )
            finally:
                consumer.shutdown()

    def test_late_completion_cannot_complete_new_reservation(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096
        payload = {
            "mm_hash": "hash",
            "transfer_id": "transfer",
            "nbytes": 64,
            "shape": [4, 4],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                old = consumer._worker._reserve_push_destination(payload)
                consumer._worker._reservations.get("transfer").expires_at = 0
                consumer._worker._expire_push_reservations()
                assert (
                    consumer._worker._reservations.get("transfer").state
                    is ConsumerReservationState.EXPIRE_PENDING
                )
                assert consumer._worker._cancel_push(
                    "transfer",
                    old["reservation_id"],
                    abandon=True,
                    refresh=True,
                )
                new = consumer._worker._reserve_push_destination(payload)
                new_record = consumer._worker._reservations.get("transfer")
                assert new_record is not None and new_record.allocation is not None
                new_allocation = new_record.allocation

                assert old["reservation_id"] != new["reservation_id"]
                stale = consumer._worker._complete_push(
                    "transfer", old["reservation_id"]
                )
                assert not stale.accepted
                assert consumer._worker._reservations.get("transfer") is new_record
                assert new_record.allocation is new_allocation
                assert new_record.state is ConsumerReservationState.WRITING
            finally:
                consumer.shutdown()

    def test_ready_reservation_has_a_terminal_expiry(self, mock_vllm_config_consumer):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                reservation = consumer._worker._reserve_push_destination(
                    {
                        "mm_hash": "hash",
                        "transfer_id": "transfer-1",
                        "nbytes": 64,
                        "shape": [4, 4],
                        "dtype": "float32",
                    }
                )
                consumer._worker._complete_push(
                    "transfer-1", reservation["reservation_id"]
                )
                consumer._worker._reservations.get("transfer-1").expires_at = 0

                assert consumer._worker._expire_push_reservations() == 1
                assert consumer._worker._reservations.status("transfer-1") is None
            finally:
                consumer.shutdown()

    def test_cancel_before_reserve_creates_bounded_tombstone(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096
        payload = {
            "mm_hash": "hash",
            "transfer_id": "cancelled-transfer",
            "nbytes": 64,
            "shape": [4, 4],
            "dtype": "float32",
        }

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                assert consumer._worker._cancel_push("cancelled-transfer", "")
                cancelled = consumer._worker._reserve_push_destination(payload)
                assert cancelled["cancelled"] and not cancelled["write"]
                assert (
                    consumer._worker._reservations.status("cancelled-transfer") is None
                )

                consumer._worker._reservations.get("cancelled-transfer").expires_at = 0
                consumer._worker._expire_push_reservations()
                replacement = consumer._worker._reserve_push_destination(payload)
                assert replacement["write"]
            finally:
                consumer.shutdown()

    def test_repeated_cancel_does_not_strand_older_tombstones(
        self, mock_vllm_config_consumer
    ):
        """Re-cancelling refreshes a tombstone without breaking the sweep order.

        The sweep stops at the first live record, so a refreshed one that kept
        its original position would shield every older record behind it and
        the table would grow for the life of the process.
        """
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(
                    torch.device("cpu"), receiving_rank=True, allow_host=True
                )
                assert consumer._worker._cancel_push("refreshed-transfer", "")
                assert consumer._worker._cancel_push("stale-transfer", "")
                consumer._worker._reservations.get("stale-transfer").expires_at = 0.0
                assert consumer._worker._cancel_push("refreshed-transfer", "")

                consumer._worker._expire_push_reservations()

                assert consumer._worker._reservations.get("stale-transfer") is None
                assert (
                    consumer._worker._reservations.get("refreshed-transfer").state
                    is ConsumerReservationState.CANCELLED
                )
                assert (
                    consumer._worker._consumer_worker_metrics["cancel_records_dropped"]
                    == 1
                )
            finally:
                consumer.shutdown()

    def test_missing_push_reservation_reports_failed_load(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        spec = ECMooncakeLoadSpec(
            mm_hash="hash",
            num_token=1,
            nbytes=32,
            shape=(8,),
            dtype="float32",
            pushed=True,
            transfer_id="missing-transfer",
            reservation_id="missing-reservation",
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[spec])
                )
                cache: dict[str, torch.Tensor] = {}
                consumer.start_load_caches(cache)
                meta = consumer.build_connector_worker_meta()
                assert meta.failed_loads == {"hash"}
                assert cache == {}
            finally:
                consumer.shutdown()

    def test_producer_scheduler_has_cache_item_false(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            mm_hash = mock_request_with_3_mm.mm_features[0].identifier
            assert not scheduler.has_cache_item(mm_hash)
