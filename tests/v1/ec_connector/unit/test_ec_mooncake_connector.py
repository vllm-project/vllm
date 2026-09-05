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
import socket
import threading
import time
import weakref
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
import zmq

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector import mooncake_ec_connector
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.mooncake import (
    control,
    memory,
    transfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    _RESERVATION_TTL_SECONDS,
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
    EventInbox,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ContiguousAllocator,
    ProducerMemoryPool,
    ResidentPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushRecord,
    ProducerPushState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    ECMooncakeWorker,
    _FanoutError,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput

pytest_plugins = ("tests.v1.ec_connector.unit.test_ec_example_connector",)
pytestmark = pytest.mark.skip_global_cleanup


class CopyingFakeTransferEngine:
    """Model Mooncake registration rules while copying bytes in-process."""

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
        if not meta.pending_saves:
            return meta
        time.sleep(0.01)
    raise TimeoutError("EC Mooncake worker I/O did not finish")


def _drain_until_subscribed(scheduler: Any, timeout: float = 5.0) -> None:
    """Drain until the event-channel discovery lands.

    Subscribing runs on the control executor so an unreachable consumer costs
    the Scheduler nothing, which means the first drain only starts it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        scheduler._drain_pending = True
        scheduler._drain_push_notifications()
        if scheduler._event_inbox._socket is not None:
            return
        time.sleep(0.005)
    raise TimeoutError("EC Mooncake event channel was never subscribed")


def _bind_extra_config(config: VllmConfig) -> None:
    config.ec_transfer_config.get_from_extra_config.side_effect = lambda key, default: (
        config.ec_transfer_config.ec_connector_extra_config.get(key, default)
    )


class TestECMooncakeControlPlane:
    """Validate ZMQ client reuse, shard discovery, events, and server RPCs."""

    def test_event_transport_failure_stops_draining(self):
        inbox = EventInbox(Mock())
        socket = Mock()
        socket.recv_json.side_effect = zmq.ZMQError(zmq.ENOTSOCK)
        inbox._socket = socket
        assert inbox.drain("unused") == []
        socket.recv_json.assert_called_once()
        inbox.close()
        assert inbox._socket is None

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
            call(zmq.IPV6, 1),
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
        client = object.__new__(ControlClient)
        client._topologies = {}
        client.request = Mock()
        client.request.side_effect = [
            {"ports": [19019, 19020]},
            RuntimeError("old consumer"),
            {"ports": [19029, 19030]},
        ]
        assert client.discover_shards("tcp://consumer:19019") == [
            "tcp://consumer:19019",
            "tcp://consumer:19020",
        ]
        assert client.discover_shards("tcp://consumer:19019") == [
            "tcp://consumer:19019",
            "tcp://consumer:19020",
        ]
        assert client.discover_shards("tcp://legacy:19019") is None
        assert client.discover_shards("tcp://legacy:19019") == [
            "tcp://legacy:19029",
            "tcp://legacy:19030",
        ]
        assert client.discover_shards("tcp://legacy:19019") == [
            "tcp://legacy:19029",
            "tcp://legacy:19030",
        ]
        assert client.request.call_args_list == [
            call("tcp://consumer:19019", {"op": "peers"}),
            call("tcp://legacy:19019", {"op": "peers"}),
            call("tcp://legacy:19019", {"op": "peers"}),
        ]

    def test_event_inbox_retries_until_every_shard_is_connected(self):
        client = object.__new__(ControlClient)
        client._topologies = {}
        client.request = Mock()
        client.request.side_effect = [
            RuntimeError("peers not ready"),
            {"ports": [19019, 19020]},
            20001,
            RuntimeError("event port not ready"),
            20001,
            20002,
        ]
        context = MagicMock()
        socket = context.socket.return_value
        event = {"transfer_id": "transfer", "ready": True}
        socket.recv_json.side_effect = [event, zmq.Again()]

        with patch.object(control.zmq, "Context", return_value=context) as create:
            inbox = EventInbox(client)
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
            return True, True

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

    def test_server_keeps_serving_after_an_undecodable_request(self):
        """One bad frame must not take the shard's control channel down.

        The loop's `finally` closes both sockets, so an escaping exception ends
        the thread silently and every later reserve against this shard surfaces
        only as a control timeout.
        """
        port = _find_free_port()
        server = ConsumerControlServer(
            "127.0.0.1",
            port,
            reserve=lambda request: {"nbytes": request["nbytes"], "ready": False},
            status=lambda transfer_id: None,
            complete=lambda transfer_id, reservation_id: (True, True),
            cancel=lambda transfer_id, reservation_id, abandon, refresh: True,
            reap=lambda: 0,
            peer_ports=[port],
        )
        server.start()
        addr = f"tcp://127.0.0.1:{port}"
        context = zmq.Context()
        raw = context.socket(zmq.REQ)
        raw.setsockopt(zmq.RCVTIMEO, 2000)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(addr)
        client = ControlClient(2000)
        try:
            raw.send(b"{not json")
            assert raw.recv_json()["ok"] is False

            # An unknown op raises inside the handler; both paths must leave
            # the channel able to answer the next caller.
            with pytest.raises(RuntimeError):
                client.request(addr, {"op": "nonsense"})

            assert client.request(addr, {"op": "peers"}) == {"ports": [port]}
        finally:
            raw.close(linger=0)
            context.term()
            client.close()
            server.close()

    def test_an_unconfirmed_writer_keeps_its_destination_reserved(self):
        """Only writer confirmation makes a timed-out destination reusable."""
        engine = MagicMock(spec=MooncakeTransfer)
        engine.register_memory.return_value = 0
        engine.unregister_memory.return_value = True
        pool = ConsumerMemoryPool(768, engine)
        pool.prepare(torch.device("cpu"))
        manager = ConsumerReservationManager(pool, 300.0, 16)

        def reserve(request: dict[str, Any]) -> dict[str, Any]:
            """The Worker's own handler, minus the dtype plumbing."""
            manager.expire()
            reservation, _ = manager.reserve(
                str(request["transfer_id"]),
                str(request["mm_hash"]),
                int(request["nbytes"]),
                tuple(int(value) for value in request["shape"]),
                str(request["dtype"]),
                torch.float32,
            )
            if reservation is None:
                raise RuntimeError("EC consumer buffer pool is full")
            return {"ready": reservation.state is ConsumerReservationState.READY}

        port = _find_free_port()
        server = ConsumerControlServer(
            "127.0.0.1",
            port,
            reserve=reserve,
            status=manager.status,
            complete=manager.complete,
            cancel=manager.cancel,
            reap=manager.expire,
            peer_ports=[port],
        )
        server.start()
        addr = f"tcp://127.0.0.1:{port}"
        client = ControlClient(2000)

        def request_destination(transfer_id: str) -> None:
            client.request(
                addr,
                {
                    "op": "reserve",
                    "transfer_id": transfer_id,
                    "mm_hash": f"hash-{transfer_id}",
                    "nbytes": 64,
                    "shape": [16],
                    "dtype": "float32",
                },
            )

        try:
            abandoned = 0
            for index in range(64):
                try:
                    request_destination(f"t{index}")
                except RuntimeError as error:
                    assert "pool is full" in str(error)
                    break
                # The writer dies here, so no completion ever arrives; the
                # Scheduler times the transfer out and cancels it.
                assert client.request(
                    addr, control.make_cancel_request(f"t{index}")
                ) == {"cancelled": True}
                abandoned += 1
            assert abandoned, "the pool should accept a writer before filling up"

            for record in manager._records.values():
                record.expires_at = 0
            with pytest.raises(RuntimeError, match="pool is full"):
                request_destination("after-the-grace")
            for transfer_id in list(manager._records):
                client.request(
                    addr, control.make_cancel_request(transfer_id, abandon=True)
                )
            request_destination("after-writer-confirmation")
        finally:
            client.close()
            server.close()


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
    config.ec_transfer_config.ec_ip = "127.0.0.1"
    config.ec_transfer_config.ec_port = 19019
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
    }
    _bind_extra_config(config)
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
    config.ec_transfer_config.ec_ip = "127.0.0.1"
    config.ec_transfer_config.ec_port = 19019
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
    }
    _bind_extra_config(config)
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
            "transfer._MOONCAKE_IMPORT_ERROR",
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
    """Validate factory registration."""

    def test_factory_registers_connector(self):
        cls = ECConnectorFactory.get_connector_class(
            Mock(ec_connector="ECMooncakeConnector")
        )
        assert cls is ECMooncakeConnector
        assert (
            cls.__module__
            == "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector"
        )


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
        pool.insert("oldest", "first")
        pool.insert("next", "second")
        pool.retire("oldest")
        pool.retire("next")

        evicted = pool.evict_lru(lambda key, _: key != "oldest")

        assert evicted == "next"
        assert pool.get("oldest") == "first"
        assert pool.insert("oldest", "replacement") == "first"
        assert pool.get("oldest") == "replacement"

    def test_displaced_entry_waits_for_every_lease(self):
        pool = ResidentPool[str]()
        pool.insert("hash", "original")
        first = pool.acquire("hash")
        second = pool.acquire("hash")
        assert first is not None and second is not None

        assert pool.insert("hash", "replacement") is None
        assert pool.release(first) is None
        assert pool.release(second) == "original"
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
        pool.prepare(torch.device("cpu"))
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
        pool.prepare(torch.device("cpu"))
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
        pool.prepare(torch.device("cpu"))
        allocation = pool.try_allocate(64, (16,), torch.float32)
        assert allocation is not None
        pool.publish("hash", allocation)
        event = self._Event(complete=False)

        with patch.object(pool, "_record_release_event", return_value=event):
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

        pool.prepare(torch.device("cpu"))
        pool.prepare(torch.device("cpu"))

        assert pool.tensor is None
        mooncake_transfer.register_memory.assert_called_once()

    def test_consumer_close_unregisters_once_and_releases_parent(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        mooncake_transfer.unregister_memory.return_value = True
        pool = ConsumerMemoryPool(256, mooncake_transfer)
        pool.prepare(torch.device("cpu"))
        parent = pool.tensor

        pool.close()
        pool.close()

        mooncake_transfer.unregister_memory.assert_called_once_with(parent)
        assert pool.tensor is None

    def test_producer_reuses_staging_and_unregisters_parent_on_close(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        mooncake_transfer.unregister_memory.return_value = True
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

        assert pool.tensor is None
        mooncake_transfer.unregister_memory.assert_called_once_with(parent)

    @pytest.mark.parametrize("pool_type", [ProducerMemoryPool, ConsumerMemoryPool])
    def test_failed_unregister_keeps_the_registered_buffer_until_retry(self, pool_type):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        mooncake_transfer.unregister_memory.side_effect = [False, True]
        pool = pool_type(256, mooncake_transfer)
        if isinstance(pool, ProducerMemoryPool):
            staged = pool.stage([torch.ones(16)])
            assert staged is not None
            pool.release(staged)
        else:
            pool.prepare(torch.device("cpu"))
        parent = pool.tensor
        assert parent is not None

        pool.close()
        assert pool.tensor is parent
        pool.close()
        assert pool.tensor is None
        pool.close()
        assert mooncake_transfer.unregister_memory.call_args_list == [
            call(parent),
            call(parent),
        ]

    def test_producer_failed_batch_returns_reserved_regions(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        mooncake_transfer.register_memory.return_value = 0
        pool = ProducerMemoryPool(256, mooncake_transfer)
        source = torch.arange(16, dtype=torch.float32)

        assert pool.stage([source, source]) is None
        staged = pool.stage([source])
        assert staged is not None
        assert torch.equal(staged.tensors[0], source)
        mooncake_transfer.register_memory.assert_called_once()
        pool.release(staged)
        pool.close()

    def test_producer_falls_back_when_staging_pool_allocation_fails(self):
        mooncake_transfer = MagicMock(spec=MooncakeTransfer)
        pool = ProducerMemoryPool(256, mooncake_transfer)

        with patch.object(memory.torch, "empty", side_effect=torch.OutOfMemoryError):
            assert pool.stage([torch.ones(16)]) is None
            assert pool.stage([torch.ones(16)]) is None

        mooncake_transfer.register_memory.assert_not_called()


class TestMooncakeECConfig:
    def test_defaults_are_a_frozen_snapshot(self, mock_vllm_config_producer):
        config = MooncakeECConfig.from_vllm_config(mock_vllm_config_producer)

        assert (
            config.protocol,
            config.buffer_device,
            config.control_timeout_ms,
            config.push_wait_timeout_s,
            config.pool_size,
        ) == ("tcp", "cuda", 30_000, 60, 1_000_000_000)
        with pytest.raises(FrozenInstanceError):
            config.protocol = "rdma"  # type: ignore[misc]

    def test_derives_rank_local_port_and_custom_resources(
        self, mock_vllm_config_consumer
    ):
        source = mock_vllm_config_consumer
        source.parallel_config.tensor_parallel_size = 2
        source.parallel_config.data_parallel_index = 1
        source.ec_transfer_config.ec_buffer_size = 2048
        source.ec_transfer_config.ec_port = 5000
        source.ec_transfer_config.ec_connector_extra_config.update(
            {
                "control_timeout_s": 1.5,
                "push_wait_timeout_s": 2.5,
            }
        )

        config = MooncakeECConfig.from_vllm_config(source)

        assert config.control_port == 5002
        assert config.control_addr == "tcp://127.0.0.1:5002"
        assert (
            config.control_timeout_ms,
            config.push_wait_timeout_s,
            config.pool_size,
        ) == (1500, 2.5, 2048)

    @pytest.mark.parametrize("key", ["control_timeout_s", "push_wait_timeout_s"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
    def test_rejects_invalid_timeouts(self, mock_vllm_config_producer, key, value):
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[key] = (
            value
        )

        with pytest.raises(ValueError, match=key):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_producer)

    @pytest.mark.parametrize(
        "value", [1.5, True, float("nan"), float("inf"), float("-inf")]
    )
    def test_rejects_invalid_registered_buffer(self, mock_vllm_config_producer, value):
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = value

        with pytest.raises(ValueError, match="ec_buffer_size"):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_producer)

    @pytest.mark.parametrize(
        ("attribute", "message"),
        [
            ("tensor_parallel_size", "tensor_parallel_size=1"),
            ("pipeline_parallel_size", "pipeline parallelism"),
            ("data_parallel_size", "data_parallel_size=1"),
        ],
    )
    def test_rejects_sharded_producer(
        self, mock_vllm_config_producer, attribute, message
    ):
        setattr(mock_vllm_config_producer.parallel_config, attribute, 2)
        with pytest.raises(ValueError, match=message):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_producer)

    @pytest.mark.parametrize("port", [0, 65536])
    def test_rejects_out_of_range_port(self, mock_vllm_config_consumer, port):
        mock_vllm_config_consumer.ec_transfer_config.ec_port = port
        with pytest.raises(ValueError, match="1..65535"):
            MooncakeECConfig.from_vllm_config(mock_vllm_config_consumer)

    def test_uses_upstream_ip_and_port(self, mock_vllm_config_consumer):
        mock_vllm_config_consumer.ec_transfer_config.ec_ip = "consumer"
        mock_vllm_config_consumer.ec_transfer_config.ec_port = 19100

        config = MooncakeECConfig.from_vllm_config(mock_vllm_config_consumer)

        assert config.control_addr == "tcp://consumer:19100"


class TestECMooncakeConnectorValidation:
    @pytest.mark.parametrize(
        "role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER]
    )
    def test_requires_mooncake_dependency(self, mock_vllm_config_producer, role):
        with (
            patch.object(transfer, "_MOONCAKE_IMPORT_ERROR", ImportError("missing")),
            pytest.raises(ImportError, match="mooncake-transfer-engine"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, role)

    @pytest.mark.parametrize(
        ("role", "active", "inactive"),
        [
            (ECConnectorRole.SCHEDULER, "_scheduler", "_worker"),
            (ECConnectorRole.WORKER, "_worker", "_scheduler"),
        ],
    )
    def test_constructs_one_delegate_and_closes_once(
        self, mock_vllm_config_producer, role, active, inactive
    ):
        scheduler = Mock()
        worker = Mock()
        with (
            patch.object(
                mooncake_ec_connector,
                "ECMooncakeScheduler",
                return_value=scheduler,
            ),
            patch.object(
                mooncake_ec_connector,
                "ECMooncakeWorker",
                return_value=worker,
            ),
        ):
            connector = ECMooncakeConnector(mock_vllm_config_producer, role)
            assert getattr(connector, active) is not None
            assert getattr(connector, inactive) is None
            connector.shutdown()
            connector.shutdown()

        if role == ECConnectorRole.SCHEDULER:
            scheduler.close.assert_called_once_with()
            worker.close.assert_not_called()
        else:
            worker.close.assert_called_once_with()
            scheduler.close.assert_not_called()


class TestECMooncakeMetadata:
    @pytest.mark.parametrize(
        "value",
        [
            ECMooncakeConnectorMetadata(
                loads=[
                    ECMooncakeLoadSpec(
                        mm_hash="load",
                        nbytes=8,
                        shape=(2, 4),
                        dtype="float16",
                        transfer_id="transfer",
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
                pending_saves=True,
            ),
        ],
    )
    def test_pickle_round_trip(self, value):
        assert ForkingPickler.loads(ForkingPickler.dumps(value)) == value


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
    @staticmethod
    def pushed_spec(transfer_id: str, mm_hash: str = "hash") -> ECMooncakeLoadSpec:
        return ECMooncakeLoadSpec(
            mm_hash=mm_hash,
            nbytes=16,
            shape=(4,),
            dtype="float32",
            transfer_id=transfer_id,
        )

    def test_load_completion_and_resident_reload(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, accepted = table.observe_ready(self.pushed_spec("transfer"), 10)

        assert accepted
        assert table.begin_load("hash", "transfer", "request") is record
        assert table.take_loads_to_dispatch() == [record]
        assert table.complete_load("hash")
        table.release_ready("hash", 1)
        assert record.state is SchedulerTransferState.RESIDENT
        assert table.begin_load("hash") is record
        assert record.spec is not None and record.spec.local

    def test_same_hash_index_preserves_order_and_identity(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        first, _ = table.observe_ready(self.pushed_spec("first"), 10)
        second, _ = table.observe_ready(self.pushed_spec("second"), 10)

        assert table.records_for_hash("hash", tuple(SchedulerTransferState)) == [
            first,
            second,
        ]
        assert table.begin_load("hash") is first
        assert (
            table.first_for_hash("hash", (SchedulerTransferState.AVAILABLE,)) is second
        )
        # A colliding transfer id is refused rather than raised: transfer ids
        # arrive on the request, so the engine must not fail on one.
        collided, accepted = table.observe_ready(
            self.pushed_spec("first", "other-hash"), 10
        )
        assert collided is first and not accepted
        assert first.mm_hash == "hash"

    def test_unavailable_notification_is_drained_once_and_rejects_late_ready(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record = table.wait_for_event("transfer", "request", "hash", 1)
        table.mark_unavailable("transfer", 2)

        assert table.take_unavailable_requests() == {"request"}
        assert table.take_unavailable_requests() == set()
        _, accepted = table.observe_ready(self.pushed_spec("transfer"), 40)
        assert not accepted
        assert record.state is SchedulerTransferState.UNAVAILABLE

    def test_cancel_and_duplicate_completion_are_idempotent(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        table.wait_for_event("cancelled", "request", "hash", 10)
        assert table.cancel("cancelled", 1)
        assert not table.cancel("cancelled", 2)
        _, accepted = table.observe_ready(self.pushed_spec("cancelled"), 40)
        assert not accepted

        record, _ = table.observe_ready(self.pushed_spec("completed", "other"), 10)
        table.begin_load("other", "completed")
        assert table.complete_load("other")
        assert table.complete_load("other")
        assert record.state is SchedulerTransferState.READY

    def test_terminal_records_expire_and_are_bounded(self):
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("failed"), 10)
        table.begin_load("hash", "failed")
        table.fail_load("hash", 20)
        assert record.deadline == 50
        table.expire(51, terminal_limit=100)
        assert table.get("failed") is None

        for transfer_id in ("first", "second", "third"):
            table.cancel(transfer_id, 60)
        table.expire(61, terminal_limit=1)
        assert table.get("first") is None
        assert table.get("second") is None
        assert table.get("third") is not None

    def test_same_hash_keeps_only_latest_resident(self):
        table = SchedulerTransferTable(resident_capacity=32, tombstone_ttl=30)
        first, _ = table.observe_ready(self.pushed_spec("first"), 10)
        table.begin_load("hash", "first")
        table.complete_load("hash")
        table.release_ready("hash", 20)
        second, _ = table.observe_ready(self.pushed_spec("second"), 30)
        table.begin_load("hash", "second")
        table.complete_load("hash")
        table.release_ready("hash", 40)

        assert first.state is SchedulerTransferState.EXPIRED
        assert second.state is SchedulerTransferState.RESIDENT
        assert table.drain_orphaned() == ["first"]
        assert table.drain_orphaned() == []

    def test_a_load_that_never_reports_back_fails_its_request(self):
        """A dispatched load needs a deadline of its own.

        LOADING used to carry none, so a lost Worker report left the hash
        loading for good and deferred every later request for it, silently
        and without even a retriable failure.
        """
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        record, _ = table.observe_ready(self.pushed_spec("transfer"), 10)

        assert table.begin_load("hash", "transfer", "request", 20) is record
        assert table.take_loads_to_dispatch() == [record]

        assert table.expire(15, 16) == []
        assert table.expire(21, 16) == [record]
        assert record.state is SchedulerTransferState.UNAVAILABLE
        assert table.take_unavailable_requests() == {"request"}

    def test_a_colliding_transfer_id_fails_only_the_request_that_named_it(self):
        """Transfer ids arrive on the request, so a collision is reachable.

        Raising took the engine down with it; the transfer already under that
        id has to survive and only the newcomer may fail.
        """
        table = SchedulerTransferTable(resident_capacity=64, tombstone_ttl=30)
        first = table.wait_for_event("shared", "req-a", "hash", 1)

        assert first is not None
        assert table.wait_for_event("shared", "req-b", "other-hash", 1) is None
        assert table.take_unavailable_requests() == {"req-b"}
        assert first.mm_hash == "hash"
        assert first.request_id == "req-a"
        assert not table.cancel("shared", 2, "other-hash", "req-b")
        assert first.state is SchedulerTransferState.WAITING_EVENT


class TestECMooncakeSchedulerMetadata:
    """Validate Scheduler decisions and per-step Worker metadata."""

    @pytest.mark.parametrize("payload", [[], "ready", 1, None])
    def test_non_object_events_are_discarded(self, payload):
        scheduler = object.__new__(ECMooncakeScheduler)
        scheduler._drain_pending = True
        scheduler._control_addr = "unused"
        scheduler._poll_pending_cancels = Mock()
        scheduler._expire_transfers = Mock()
        scheduler._event_inbox = Mock()
        scheduler._event_inbox.drain.return_value = [payload]
        scheduler._accept_ready_event = Mock()
        scheduler._drain_push_notifications()
        scheduler._accept_ready_event.assert_not_called()

    def test_an_unreachable_consumer_does_not_block_the_scheduler(
        self, mock_vllm_config_consumer
    ):
        """Subscribing must never cost the Scheduler a control timeout.

        `has_cache_item` runs inside `schedule()`, and discovery is one
        blocking request per shard. Doing it inline froze every request in the
        engine for `control_timeout_s` on each drain while a shard was
        unreachable, and `discover_shards` caches only successes, so the cost
        repeated for as long as the shard stayed down.
        """
        timeout_s = 0.4
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "control_timeout_s": timeout_s,
        }
        mock_vllm_config_consumer.ec_transfer_config.ec_port = _find_free_port()
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                elapsed = []
                for _ in range(3):
                    started = time.monotonic()
                    assert scheduler.has_cache_item("hash") is False
                    elapsed.append(time.monotonic() - started)
            finally:
                scheduler.shutdown()

        assert max(elapsed) < timeout_s, elapsed

    def test_a_duplicate_transfer_id_fails_the_request_not_the_engine(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """`ec_transfer_params` is a request field, so ids can collide.

        `ensure_cache_available` runs inside `schedule()`: raising there took
        EngineCore down on input any client could send.
        """
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                first = mock_request_with_3_mm
                first.mm_features = first.mm_features[:1]
                first.request_id = "req-a"
                first.ec_transfer_params = {
                    "ec_items": [
                        {
                            "mm_hash": first.mm_features[0].identifier,
                            "transfer_id": "shared",
                        }
                    ]
                }
                second = copy.copy(first)
                second.request_id = "req-b"
                second.mm_features = [
                    replace(first.mm_features[0], identifier="another_hash")
                ]
                second.ec_transfer_params = {
                    "ec_items": [{"mm_hash": "another_hash", "transfer_id": "shared"}]
                }

                with patch.object(scheduler._scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(first, 0)
                    assert not scheduler.ensure_cache_available(second, 0)

                assert scheduler.take_unavailable_requests() == {"req-b"}
                record = scheduler._scheduler._transfers.get("shared")
                assert record is not None
                assert record.request_id == "req-a"
            finally:
                scheduler.shutdown()

    def test_cancel_confirms_topology_and_retries_only_failed_shards(self):
        scheduler = object.__new__(ECMooncakeScheduler)
        scheduler._control_client = Mock(spec=ControlClient)
        scheduler._control_client.discover_shards.side_effect = [
            None,
            ["shard-0", "shard-1", "shard-2"],
        ]
        called = []

        def request(addr, _payload):
            called.append(addr)
            if addr == "shard-0" and called.count(addr) == 1:
                raise RuntimeError("cancel shard failed")
            return {"cancelled": True}

        scheduler._control_client.request.side_effect = request
        assert scheduler._cancel_remote("base", "transfer")
        assert scheduler._control_client.discover_shards.call_args_list == [
            call("base"),
            call("base"),
        ]
        assert called == ["shard-0", "shard-1", "shard-2", "shard-0"]

    def test_cancel_rejects_unconfirmed_topology_without_sending(self):
        scheduler = object.__new__(ECMooncakeScheduler)
        scheduler._control_client = Mock(spec=ControlClient)
        scheduler._control_client.discover_shards.return_value = None

        with pytest.raises(RuntimeError, match="discover every EC consumer shard"):
            scheduler._cancel_remote("base", "transfer")

        assert scheduler._control_client.discover_shards.call_count == 2
        scheduler._control_client.request.assert_not_called()

    def test_item_with_no_transfer_in_flight_is_reported_as_stalled(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """A push that never arrives must not wait silently forever."""
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
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
                    ) as control_request,
                    patch(
                        "vllm.distributed.ec_transfer.ec_connector."
                        "mooncake.scheduler.time.monotonic",
                        side_effect=[10, 10.002, 10.003],
                    ),
                ):
                    assert not scheduler.ensure_cache_available(request, 0)
                    assert not scheduler.ensure_cache_available(request, 0)
                    record = scheduler._scheduler._transfers.get(
                        f"{request.request_id}:0"
                    )
                    assert record is not None
                    assert record.state is SchedulerTransferState.UNAVAILABLE
                    assert scheduler.take_unavailable_requests() == {request.request_id}
                    assert not scheduler.ensure_cache_available(request, 0)
                    assert scheduler.take_unavailable_requests() == set()
                    control_request.assert_not_called()
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
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        transfer_id="request-transfer",
                    ),
                    10,
                )
                with (
                    patch.object(scheduler._scheduler, "_drain_push_notifications"),
                    patch.object(scheduler._scheduler, "_queue_cancel") as cancel,
                ):
                    # The local encoder cache is mirrored from the Scheduler's
                    # own alloc/free notifications, not passed in per call.
                    scheduler.update_state_after_alloc(request, 0)
                    assert scheduler.ensure_cache_available(request, 0)
                    cancel.assert_not_called()
                    record = scheduler._scheduler._transfers.get("request-transfer")
                    assert record is not None
                    assert record.state is SchedulerTransferState.AVAILABLE

                    # Once the entry is evicted the request can still get it.
                    scheduler._scheduler._local_cache.discard(mm_hash)
                    assert not scheduler.ensure_cache_available(request, 0)
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
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        transfer_id="consumed-transfer",
                    ),
                    10,
                )
                with patch.object(
                    scheduler._scheduler, "_cancel_remote", return_value=True
                ):
                    scheduler.update_state_after_free(request, 0)
                record = scheduler._scheduler._transfers.get("consumed-transfer")
                assert record is not None
                assert record.state is SchedulerTransferState.CANCELLED
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
                scheduler._scheduler._control_addr = f"tcp://127.0.0.1:{ports[0]}"
                with patch.object(
                    scheduler._scheduler._control_client,
                    "request",
                    side_effect=fake_send,
                ) as send_control:
                    _drain_until_subscribed(scheduler._scheduler)

                subscribed = [
                    call.args[0]
                    for call in send_control.call_args_list
                    if call.args[1]["op"] == "event_port"
                ]
                assert len(subscribed) == len(ports)
                assert scheduler._scheduler._event_inbox.shard_count == len(ports)

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

                assert not scheduler.ensure_cache_available(first, 0)
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
                    assert not scheduler.ensure_cache_available(second, 0)
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
                    nbytes=16,
                    shape=(4,),
                    dtype="float32",
                    transfer_id="transfer",
                )
                table = scheduler._scheduler._transfers
                table.observe_ready(spec, time.monotonic() + _RESERVATION_TTL_SECONDS)
                table.begin_load(mm_hash, "transfer")
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
                assert not table.has_state(mm_hash, (SchedulerTransferState.RESIDENT,))
            finally:
                scheduler.shutdown()

    def test_retains_new_completion_while_same_hash_is_loading(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
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
                nbytes=64,
                shape=(2, 8),
                dtype="float32",
                transfer_id="current-transfer",
            )
            scheduler._scheduler._transfers.observe_ready(current, time.monotonic() + 1)
            scheduler._scheduler._transfers.begin_load("hash", "current-transfer")

            scheduler._scheduler._drain_push_notifications()

            pending = scheduler._scheduler._transfers.get("next-transfer")
            assert pending is not None and pending.spec is not None
            assert pending.state is SchedulerTransferState.AVAILABLE
            assert pending.spec.transfer_id == "next-transfer"

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
                nbytes=32,
                shape=(2, 4),
                dtype="float32",
                transfer_id="transfer",
            )
            scheduler._scheduler._transfers.observe_ready(
                load_spec, time.monotonic() + _RESERVATION_TTL_SECONDS
            )
            scheduler._scheduler._transfers.begin_load(mm_hash, "transfer")
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )
            assert isinstance(meta, ECMooncakeConnectorMetadata)
            assert len(meta.loads) == 1
            assert meta.loads[0].mm_hash == mm_hash
            assert scheduler._scheduler._transfers.take_loads_to_dispatch() == []
            record = scheduler._scheduler._transfers.get("transfer")
            assert record is not None
            assert record.state is SchedulerTransferState.LOADING

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
                scheduler._scheduler._metadata_resolver,
                "fields_for",
                return_value={"image_grid_thw"},
            ):
                delay_free, params = scheduler.request_finished(request)

        assert not delay_free
        # Upstream shape: keyed by the engine's own identifier, with the
        # placeholder metadata nested so a connector can report its own
        # transfer coordinates alongside it.
        assert params == {"image_uuid": {"metadata": {"image_grid_thw": [1, 32, 48]}}}


class TestConsumerReservationManager:
    @pytest.mark.parametrize("cancelled", [None, "writer", "follower"])
    def test_inflight_duplicates_share_memory_and_cancel_independently(self, cancelled):
        engine = Mock()
        engine.register_memory.return_value = 0
        pool = ConsumerMemoryPool(256, engine)
        pool.prepare(torch.device("cpu"))
        manager = ConsumerReservationManager(pool, 300, 16)
        writer, write = manager.reserve(
            "writer", "hash", 64, (16,), "float32", torch.float32
        )
        follower, write_again = manager.reserve(
            "follower", "hash", 64, (16,), "float32", torch.float32
        )
        assert write and not write_again
        assert writer.allocation is follower.allocation
        tensor = writer.allocation.tensor
        tensor.fill_(7)
        if cancelled is not None:
            manager.cancel(cancelled, "")
        assert manager.complete("writer", writer.reservation_id)[0]
        for name in ("writer", "follower"):
            if name != cancelled:
                loaded = manager.take(name, "hash")
                assert loaded.tensor.data_ptr() == tensor.data_ptr()
                assert torch.all(loaded.tensor == 7)
        assert pool.try_allocate(64, (16,), torch.float32) is None

    def test_follower_refresh_does_not_abandon_the_shared_writer(self):
        manager, pool, _ = self.manager()
        writer, _ = self.reserve(manager)
        follower, _ = manager.reserve(
            "follower", "hash", 64, (16,), "float32", torch.float32
        )
        assert manager.cancel(
            "follower", follower.reservation_id, abandon=True, refresh=True
        )
        renewed, write = manager.reserve(
            "follower", "hash", 64, (16,), "float32", torch.float32
        )
        assert not write and renewed.writer_id == writer.transfer_id
        assert renewed.reservation_id != follower.reservation_id
        assert writer.state is ConsumerReservationState.WRITING
        pool.free.assert_not_called()

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
        record, write = manager.reserve(
            "transfer", "hash", 64, (16,), "float32", torch.float32
        )
        assert record is not None
        return record, write

    def test_complete_is_idempotent(self):
        manager, _, _ = self.manager()
        record, write = self.reserve(manager)

        assert write and record.state is ConsumerReservationState.WRITING
        assert manager.complete("transfer", record.reservation_id) == (True, True)
        assert manager.complete("transfer", record.reservation_id) == (True, False)
        assert record.state is ConsumerReservationState.READY

    def test_cancel_waits_for_an_active_writer(self):
        manager, pool, allocation = self.manager()
        record, _ = self.reserve(manager)

        assert not manager.cancel("transfer", "wrong-id")
        assert manager.cancel("transfer", record.reservation_id)
        assert record.state is ConsumerReservationState.CANCEL_PENDING
        pool.free.assert_not_called()

        assert manager.complete("transfer", record.reservation_id) == (True, False)
        assert record.state is ConsumerReservationState.CANCELLED
        assert record.allocation is None
        pool.free.assert_called_once_with(allocation)

    def test_shutdown_rejects_new_reservations(self):
        manager, _, _ = self.manager()
        manager.begin_shutdown()

        with pytest.raises(RuntimeError, match="shutting down"):
            self.reserve(manager)

    def test_expired_writer_is_replaced_only_after_refresh_abandon(self):
        manager, pool, old_allocation = self.manager()
        new_allocation = memory.MemoryAllocation(256, 64, torch.ones(16))
        pool.try_allocate.side_effect = [old_allocation, new_allocation]
        old, _ = self.reserve(manager)
        old.expires_at = 0
        manager.expire()

        with pytest.raises(RuntimeError, match="active writer"):
            self.reserve(manager)
        assert manager.cancel(
            "transfer", old.reservation_id, abandon=True, refresh=True
        )
        new, write = self.reserve(manager)

        assert write and new.reservation_id != old.reservation_id
        assert new.allocation is new_allocation
        assert manager.complete("transfer", old.reservation_id) == (False, False)
        pool.free.assert_called_once_with(old_allocation)

    def test_ready_expiry_releases_once(self):
        manager, pool, allocation = self.manager()
        record, _ = self.reserve(manager)
        manager.complete("transfer", record.reservation_id)
        record.expires_at = 0

        assert manager.expire() == 1
        assert manager.expire() == 0
        assert record.state is ConsumerReservationState.EXPIRED
        pool.free.assert_called_once_with(allocation)

    def test_cached_take_uses_the_pool_canonical_allocation(self):
        manager, pool, cached = self.manager()
        lease = SimpleNamespace(value=cached)
        canonical = memory.MemoryAllocation(256, 64, torch.ones(16))
        pool.acquire_cached.return_value = lease
        pool.publish.return_value = canonical

        record, write = self.reserve(manager)
        assert not write and record.lease is lease
        assert manager.take("transfer", "hash") is canonical
        pool.publish.assert_called_once_with("hash", cached, lease)

    def test_cancel_tombstones_are_bounded(self):
        manager, _, _ = self.manager()
        manager._tombstone_limit = 3

        for transfer_id in ("a", "b", "c", "a", "d"):
            assert manager.cancel(transfer_id, "")

        assert list(manager._tombstones) == ["c", "a", "d"]
        assert set(manager._records) == {"c", "a", "d"}

    @pytest.mark.parametrize(
        "deferred,terminal",
        [
            (
                ConsumerReservationState.CANCEL_PENDING,
                ConsumerReservationState.CANCELLED,
            ),
            (
                ConsumerReservationState.EXPIRE_PENDING,
                ConsumerReservationState.EXPIRED,
            ),
        ],
    )
    def test_a_deferred_release_requires_writer_completion(self, deferred, terminal):
        """A timeout cannot prove that a remote writer stopped using its address."""
        manager, pool, allocation = self.manager()
        record, _ = self.reserve(manager)

        if deferred is ConsumerReservationState.CANCEL_PENDING:
            assert manager.cancel("transfer", "")
        else:
            record.expires_at = 0
            manager.expire()
        assert record.state is deferred
        # Mooncake may still be writing, so the release waits first.
        pool.free.assert_not_called()

        record.expires_at = 0
        assert manager.expire() == 0
        pool.free.assert_not_called()
        assert manager.complete("transfer", record.reservation_id) == (True, False)
        assert record.state is terminal
        pool.free.assert_called_once_with(allocation)


class TestECMooncakeWorkerTransfer:
    """Validate end-to-end Worker reservation, push, load, and cleanup flows."""

    def test_partial_batch_reservation_cleans_only_the_failed_item(self):
        """An item rejected on one shard must not cancel its successful sibling."""
        worker = object.__new__(ECMooncakeWorker)
        worker._control_client = Mock()
        shards = ["tcp://consumer:0", "tcp://consumer:1"]
        worker._control_client.discover_shards.return_value = shards
        specs = [
            ECMooncakePushSpec(name, 64, (16,), "float32", shards[0], name)
            for name in ("good", "bad")
        ]

        def request(addr, payload):
            assert payload["op"] == "reserve_batch"
            return {
                "items": [
                    {"ok": True, "result": {"reservation_id": "good-" + addr}},
                    {"ok": False, "error": "full"}
                    if addr == shards[1]
                    else {"ok": True, "result": {"reservation_id": "partial"}},
                ]
            }

        worker._control_client.request.side_effect = request
        worker._run_fanout = lambda tasks: [task() for task in tasks]
        worker._retry_cancel_reservations = Mock()
        good, bad = worker._reserve_remote_many(specs)
        assert [item["addr"] for item in good] == shards
        assert isinstance(bad, _FanoutError) and str(bad) == "full"
        worker._retry_cancel_reservations.assert_called_once_with(specs[1], bad.results)
        assert [item["reservation_id"] for item in bad.results] == ["partial", ""]

    def test_reservation_completion_dispatches_without_another_model_step(
        self, mock_vllm_config_producer
    ):
        allow_reservation = threading.Event()
        transferred = threading.Event()
        source = torch.ones(16)
        spec = ECMooncakePushSpec("hash", 64, (16,), "float32", "unused", "transfer")
        with patch_ec_mooncake_deps():
            worker = ECMooncakeWorker(mock_vllm_config_producer)

            def reserve(_):
                assert allow_reservation.wait(5)
                return []

            def write(records):
                for record in records:
                    worker._producer_pushes.begin_writing(record)
                worker._producer_pushes.begin_notifying(records)
                worker._producer_pushes.complete(records)
                transferred.set()

            worker._reserve_remote = reserve
            worker._push_batch = write
            try:
                worker.start_save_caches(
                    ECMooncakeConnectorMetadata(pushes=[spec]), {"hash": source}
                )
                worker.build_connector_worker_meta()
                assert not transferred.is_set()
                allow_reservation.set()
                assert transferred.wait(5)
            finally:
                allow_reservation.set()
                worker.close()

    @pytest.mark.parametrize("waiting_for", ["reservation", "encoder"])
    def test_a_ready_push_does_not_wait_for_another_item(self, waiting_for):
        manager = ProducerPushManager()
        records = []
        for name in ("slow", "fast"):
            future: Future[list[dict[str, Any]]] = Future()
            spec = ECMooncakePushSpec(
                name, 64, (16,), "float32", "tcp://consumer:1", name
            )
            record, _ = manager.reserve(spec, lambda future=future: future)
            event = Mock()
            event.query.return_value = name == "fast" or waiting_for == "reservation"
            manager.bind_source(name, torch.empty(16), event)
            if name == "fast" or waiting_for == "encoder":
                future.set_result([])
            records.append(record)
        executor = Mock()
        executor.submit.return_value = Future()
        run = Mock()
        manager.submit_batches(executor, run)
        executor.submit.assert_called_once_with(run, [records[1]])

    def test_reservation_requires_confirmed_topology_before_any_rpc(self):
        worker = object.__new__(ECMooncakeWorker)
        worker._control_client = Mock()
        worker._control_client.discover_shards.return_value = None
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
        )

        with pytest.raises(RuntimeError, match="discover every EC consumer shard"):
            worker._reserve_remote(spec)

        assert worker._control_client.discover_shards.call_count == 2
        worker._control_client.request.assert_not_called()

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
        # Another request may legitimately name the same encoding.
        reasked = copy.copy(spec)
        reasked.request_id = "another-request"
        assert manager.reserve(reasked, lambda: Future()) == (record, False)

        # A different payload under the same id drops the newcomer instead of
        # failing the engine; the push in flight keeps the id.
        changed = copy.copy(spec)
        changed.mm_hash = "other"
        with pytest.raises(ValueError, match="Conflicting EC destination"):
            manager.reserve(changed, lambda: Future())
        assert record.spec.mm_hash == "hash"

        source = torch.empty(16)
        manager.bind_source("hash", source, None)
        assert record.source_tensor is source
        reservation.set_result([])
        assert manager.resolve_reservations(record) == []
        assert record.state is ProducerPushState.WAITING_INPUTS
        manager.begin_writing(record)
        manager.begin_notifying([record])

        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("one shard failed"))
        still_writing: Future[None] = Future()
        manager.track_shard_futures([record], [failed, still_writing])
        with pytest.raises(RuntimeError, match="source too early"):
            manager.fail([record], RuntimeError("write failed"))
        assert record.state is ProducerPushState.NOTIFYING
        assert record.source_tensor is source

        still_writing.set_result(None)
        manager.fail([record], RuntimeError("write failed"))
        assert record.state is ProducerPushState.FAILED
        assert record.source_tensor is None
        manager.fail([record], RuntimeError("duplicate failure"))
        with pytest.raises(RuntimeError, match="FAILED to NOTIFYING"):
            manager.begin_notifying([record])

        late, late_created = manager.reserve(spec, lambda: Future())
        assert late is record
        assert not late_created

    def test_cancel_requests_none_selects_every_source_less_waiter(self):
        manager = ProducerPushManager()

        def reserve(transfer_id: str, mm_hash: str, request_id: str):
            spec = ECMooncakePushSpec(
                mm_hash=mm_hash,
                nbytes=64,
                shape=(16,),
                dtype="float32",
                consumer_zmq="tcp://consumer:1",
                transfer_id=transfer_id,
                request_id=request_id,
            )
            return manager.reserve(spec, lambda: Future())[0]

        first = reserve("first", "hash-first", "request-first")
        second = reserve("second", "hash-second", "request-second")
        assert manager.cancel_requests({"request-first"}) == [first]
        assert manager.cancel_requests(None) == [second]
        assert first.state is ProducerPushState.CANCEL_PENDING
        assert second.state is ProducerPushState.CANCEL_PENDING

    def test_worker_close_cancels_orphaned_reservations_before_executor_shutdown(
        self,
    ):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="orphan",
            request_id="request",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        events: list[Any] = []

        class RecordingExecutor:
            def __init__(self, name: str):
                self.name = name

            def submit(self, function, *args):
                events.append(f"{self.name}.submit")
                future: Future[Any] = Future()
                try:
                    result = function(*args)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
                return future

            def shutdown(self, wait=True, **kwargs):
                events.append((f"{self.name}.shutdown", wait, kwargs))

        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = manager
        worker._io_executor = RecordingExecutor("io")
        worker._control_executor = RecordingExecutor("control")
        worker._shard_pool = RecordingExecutor("shard")
        worker._shutdown = False
        worker._control_client = Mock()
        worker._control_server = None
        worker._consumer_memory = Mock()
        worker._producer_memory = Mock()
        worker._transfer = Mock()
        worker._flush_pending_pushes = lambda: events.append("flush")

        def finish_cancel(orphan: ProducerPushRecord):
            events.append(f"cancel:{orphan.spec.transfer_id}")
            manager.finish_cancel(orphan)

        worker._cancel_orphaned_reservation = finish_cancel

        worker.close()
        worker.close()

        assert record.state is ProducerPushState.CANCELLED
        assert events == [
            "flush",
            "io.submit",
            "cancel:orphan",
            ("control.shutdown", True, {}),
            ("io.shutdown", True, {}),
            ("shard.shutdown", True, {}),
        ]
        worker._control_client.close.assert_called_once_with()
        worker._consumer_memory.close.assert_called_once_with()
        worker._producer_memory.close.assert_called_once_with()
        worker._transfer.close.assert_called_once_with()

    def test_worker_close_drains_an_unresolved_reservation_before_control_shutdown(
        self,
    ):
        manager = ProducerPushManager()
        control_executor = ThreadPoolExecutor(max_workers=1)

        def resolve_reservation():
            time.sleep(0.02)
            return [
                {
                    "addr": "tcp://consumer:19019",
                    "reservation_id": "r0",
                },
                {
                    "addr": "tcp://consumer:19020",
                    "reservation_id": "r1",
                },
            ]

        reservation = control_executor.submit(resolve_reservation)
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="orphan",
            request_id="request",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        io_executor = ThreadPoolExecutor(max_workers=1)
        shard_pool = ThreadPoolExecutor(max_workers=1)
        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = manager
        worker._io_executor = io_executor
        worker._control_executor = control_executor
        worker._shard_pool = shard_pool
        worker._shard_pool_lock = threading.Lock()
        worker._shutdown = False
        worker._control_client = Mock()
        worker._control_client.request.return_value = {"cancelled": True}
        worker._control_server = None
        worker._consumer_memory = Mock()
        worker._producer_memory = Mock()
        worker._transfer = Mock()
        worker._flush_pending_pushes = Mock()

        worker.close()

        assert record.state is ProducerPushState.CANCELLED
        assert worker._control_client.request.call_count == 2
        assert all(
            request.args[1]["op"] == "cancel" and request.args[1]["abandon"]
            for request in worker._control_client.request.call_args_list
        )

    def test_consumer_close_waits_for_remote_writer_before_releasing_pool(self):
        reservations, consumer_memory, _ = TestConsumerReservationManager.manager()
        record, _ = TestConsumerReservationManager.reserve(reservations)
        shutdown_started = threading.Event()
        original_begin_shutdown = reservations.begin_shutdown

        def begin_shutdown():
            original_begin_shutdown()
            shutdown_started.set()

        reservations.begin_shutdown = begin_shutdown
        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = ProducerPushManager()
        worker._io_executor = Mock()
        worker._control_executor = Mock()
        worker._shard_pool = None
        worker._shutdown = False
        worker._control_client = Mock()
        worker._control_server = Mock()
        worker._consumer_memory = consumer_memory
        worker._producer_memory = Mock()
        worker._transfer = Mock()
        worker._reservations = reservations
        worker._shutdown_drain_timeout_s = 1
        worker._flush_pending_pushes = Mock()

        close_thread = threading.Thread(target=worker.close)
        close_thread.start()
        assert shutdown_started.wait(1)
        assert record.state is ConsumerReservationState.CANCEL_PENDING
        consumer_memory.close.assert_not_called()
        worker._control_server.close.assert_not_called()

        assert reservations.complete("transfer", record.reservation_id) == (
            True,
            False,
        )
        close_thread.join(1)

        assert not close_thread.is_alive()
        consumer_memory.close.assert_called_once_with()
        worker._control_server.close.assert_called_once_with()

    def test_consumer_close_timeout_keeps_receive_pool_registered(self):
        reservations, consumer_memory, allocation = (
            TestConsumerReservationManager.manager()
        )
        record, _ = TestConsumerReservationManager.reserve(reservations)
        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = ProducerPushManager()
        worker._io_executor = Mock()
        worker._control_executor = Mock()
        worker._shard_pool = None
        worker._shutdown = False
        worker._control_client = Mock()
        worker._control_server = Mock()
        worker._consumer_memory = consumer_memory
        worker._producer_memory = Mock()
        worker._transfer = Mock()
        worker._reservations = reservations
        worker._shutdown_drain_timeout_s = 0
        worker._flush_pending_pushes = Mock()

        worker.close()

        assert record.state is ConsumerReservationState.CANCEL_PENDING
        assert record.allocation is allocation
        consumer_memory.close.assert_not_called()
        worker._control_server.close.assert_called_once_with()

    def test_permanent_orphan_cleanup_failure_marks_push_failed(self):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        reservation.set_result(
            [
                {
                    "addr": "tcp://consumer:19019",
                    "reservation_id": "reservation",
                    "cached": True,
                }
            ]
        )
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
            request_id="request",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        assert manager.cancel_requests({"request"}) == [record]

        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = manager
        worker._control_client = Mock()
        worker._control_client.request.side_effect = RuntimeError("cancel failed")
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            patch.object(worker, "_shard_executor", return_value=executor),
        ):
            worker._cancel_orphaned_reservation(record)

        assert record.state is ProducerPushState.FAILED
        assert record.error == "cancel failed"
        assert worker._control_client.request.call_count == 2
        assert manager.poll() == [("hash", "cancel failed")]
        assert manager.poll() == []

    def test_orphan_topology_failure_marks_push_failed_without_base_cancel(self):
        manager = ProducerPushManager()
        reservation: Future[list[dict[str, Any]]] = Future()
        spec = ECMooncakePushSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(16,),
            dtype="float32",
            consumer_zmq="tcp://consumer:19019",
            transfer_id="transfer",
            request_id="request",
        )
        record, _ = manager.reserve(spec, lambda: reservation)
        assert manager.cancel_requests({"request"}) == [record]
        reservation.set_exception(RuntimeError("topology unavailable"))

        worker = object.__new__(ECMooncakeWorker)
        worker._producer_pushes = manager
        worker._control_client = Mock()
        worker._cancel_orphaned_reservation(record)

        assert record.state is ProducerPushState.FAILED
        assert record.error == "topology unavailable"
        worker._control_client.request.assert_not_called()

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

        assert record.state is ProducerPushState.WAITING_INPUTS

        def run(records) -> None:
            try:
                manager.resolve_reservations(records[0])
            except RuntimeError as exc:
                manager.fail(records, exc)

        with ThreadPoolExecutor(max_workers=1) as executor:
            manager.submit_batches(executor, run)
        assert manager.poll() == [("hash", "reserve failed")]
        assert manager.poll() == []
        assert record.state is ProducerPushState.FAILED
        assert record.source_tensor is None

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
                        worker._control_client,
                        "discover_shards",
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
                    record = worker._producer_pushes._records.get("transfer")
                    assert record is not None
                    record.reservation_future.result(timeout=2)
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
                            assert record.source_tensor is source
                            assert released_after_slow == []
                            finish_slow.set()
                            _wait_for_worker_io(producer)

                record = worker._producer_pushes._records.get("transfer")
                assert record is not None
                assert record.state is ProducerPushState.FAILED
                assert record.source_tensor is None
                assert released_after_slow == [True]
            finally:
                finish_slow.set()
                producer.shutdown()

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
        cancel_attempts: Counter[str] = Counter()

        def reserve_one(addr, _spec):
            if addr.endswith(":1"):
                raise RuntimeError("reserve shard failed")
            return {"addr": addr, "reservation_id": "partial-r0"}

        def request(addr, payload):
            assert payload["op"] == "cancel" and payload["abandon"]
            assert payload["transfer_id"] == "transfer"
            reservation_id = str(payload["reservation_id"])
            cancel_attempts[reservation_id] += 1
            if (
                addr.endswith(":0")
                and reservation_id == "partial-r0"
                and cancel_attempts[reservation_id] == 1
            ):
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
                        worker._control_client,
                        "discover_shards",
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
                    record = worker._producer_pushes._records.get("transfer")
                    assert record is not None
                    with pytest.raises(
                        RuntimeError, match="^reserve shard failed$"
                    ) as e:
                        record.reservation_future.result(timeout=2)
                    assert e.value.results == [
                        {
                            "addr": "tcp://consumer:0",
                            "reservation_id": "partial-r0",
                        },
                        {"addr": "tcp://consumer:1", "reservation_id": ""},
                    ]
                    assert cancel_attempts == Counter({"partial-r0": 2, "": 1})

                    if source_before_failure:
                        assert record.source_tensor is source
                        connector.build_connector_worker_meta()
                        _wait_for_worker_io(connector)
                    else:
                        assert record.state is ProducerPushState.FAILED
                        connector.save_caches({"hash": source}, "hash")
                    assert record.state is ProducerPushState.FAILED
                    assert record.source_tensor is None
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
                    record = worker._producer_pushes._records.get("transfer")
                    assert record is not None and record.batch_future is not None
                    assert slow_started.wait(2)
                    assert record.source_tensor is source
                    assert not record.batch_future.done()
                    finish_slow.set()
                    record.batch_future.result(timeout=2)
                    connector.build_connector_worker_meta()

                assert Counter(cancelled) == Counter({"r0": 1, "r1": 1})
                assert record.state is ProducerPushState.FAILED
                assert record.source_tensor is None
                assert record.error == "complete shard failed"
                assert all(future.done() for future in record.shard_futures)
            finally:
                finish_slow.set()
                connector.shutdown()

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
                        worker._control_client,
                        "discover_shards",
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
                    record = worker._producer_pushes._records.get("transfer")
                    assert record is not None and record.batch_future is not None
                    record.batch_future.result(timeout=2)
                    connector.build_connector_worker_meta()

                assert record.state is ProducerPushState.FAILED
                assert record.source_tensor is None
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                    for reservation in consumer._worker._reservations._records.values()
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                push_record = producer._worker._producer_pushes._records.get(
                    "transfer-1"
                )
                assert push_record is not None
                reservation = push_record.reservation_future
                shards = reservation.result(timeout=2)
                # One reservation per consumer shard; this consumer is single.
                assert len(shards) == 1
                reservation_data = shards[0]
                assert reservation_data["nbytes"] == source.nbytes
                old_reservation_id = reservation_data["reservation_id"]
                reservation_data["_received_at"] -= _RESERVATION_TTL_SECONDS
                consumer._worker._reservations._records["transfer-1"].expires_at = 0
                with patch.object(
                    scheduler._scheduler._control_client,
                    "request",
                    wraps=scheduler._scheduler._control_client.request,
                ) as send_control:
                    assert not scheduler.has_cache_item("hash")
                    assert not scheduler.has_cache_item("hash")
                    _drain_until_subscribed(scheduler._scheduler)
                    # The channel is built once, not per drain: the roster is
                    # fetched and every shard subscribed to exactly once.
                    assert [call.args[1] for call in send_control.call_args_list] == [
                        {"op": "peers"},
                        {"op": "event_port"},
                    ]
                    assert consumer._worker._reservations.status("transfer-1")

                    producer.save_caches({"hash": source}, "hash")
                    _wait_for_worker_io(producer)
                    assert (
                        consumer._worker._reservations._records[
                            "transfer-1"
                        ].reservation_id
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                push_record = producer._worker._producer_pushes._records.get(
                    "transfer-1"
                )
                assert push_record is not None
                reservation = push_record.reservation_future
                reservation.result(timeout=2)
                assert consumer._worker._reservations.status("transfer-1")

                producer.get_finished({"request-1"})
                _wait_for_worker_io(producer)
                push_record = producer._worker._producer_pushes._records.get(
                    "transfer-1"
                )
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                reservation = consumer._worker._reservations._records.get("transfer-1")
                assert reservation.state is ConsumerReservationState.READY

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                record = scheduler._scheduler._transfers.get("transfer-1")
                assert record is not None and record.spec is not None
                load = record.spec
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
                cached = consumer._worker._reservations._records.get("transfer-2")
                assert cached is not None
                assert cached.state is ConsumerReservationState.READY
                assert cached.lease is not None

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                record = scheduler._scheduler._transfers.get("transfer-2")
                assert record is not None and record.spec is not None
                cached_load = record.spec
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
        cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        cfg.ec_transfer_config.ec_port = port
        cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(cfg)
        spec = ECMooncakeLoadSpec(
            mm_hash="hash",
            nbytes=64,
            shape=(4, 4),
            dtype="float32",
            transfer_id="local-transfer",
            local=True,
        )

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(cfg, ECConnectorRole.WORKER)
            try:
                consumer._worker._consumer_memory.prepare(torch.device("cpu"))
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
                assert not consumer._worker._consumer_memory._residents._evictable

                assert (
                    consumer._worker._consumer_memory.take_resident(
                        spec.mm_hash, spec.shape, spec.dtype
                    )
                    is tensor
                )
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                assert engine.transfer_calls == [[source.nbytes]]
                assert all(
                    reservation.state is ConsumerReservationState.READY
                    for reservation in consumer._worker._reservations._records.values()
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
        consumer_cfg.ec_transfer_config.ec_ip = "127.0.0.1"
        consumer_cfg.ec_transfer_config.ec_port = port
        consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        _bind_extra_config(consumer_cfg)
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
                    for reservation in consumer._worker._reservations._records.values()
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

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            try:
                consumer._worker._consumer_memory.prepare(torch.device("cpu"))
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

                first = consumer._worker._reservations.complete(
                    "transfer-1", reservation_id
                )
                repeated = consumer._worker._reservations.complete(
                    "transfer-1", reservation_id
                )

                assert first == (True, True)
                assert repeated == (True, False)
            finally:
                consumer.shutdown()

    def test_late_completion_cannot_complete_new_reservation(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
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
                consumer._worker._consumer_memory.prepare(torch.device("cpu"))
                old = consumer._worker._reserve_push_destination(payload)
                consumer._worker._reservations._records["transfer"].expires_at = 0
                consumer._worker._reservations.expire()
                assert (
                    consumer._worker._reservations._records["transfer"].state
                    is ConsumerReservationState.EXPIRE_PENDING
                )
                assert consumer._worker._reservations.cancel(
                    "transfer",
                    old["reservation_id"],
                    abandon=True,
                    refresh=True,
                )
                new = consumer._worker._reserve_push_destination(payload)
                new_record = consumer._worker._reservations._records.get("transfer")
                assert new_record is not None and new_record.allocation is not None
                new_allocation = new_record.allocation

                assert old["reservation_id"] != new["reservation_id"]
                stale = consumer._worker._reservations.complete(
                    "transfer", old["reservation_id"]
                )
                assert stale == (False, False)
                assert (
                    consumer._worker._reservations._records.get("transfer")
                    is new_record
                )
                assert new_record.allocation is new_allocation
                assert new_record.state is ConsumerReservationState.WRITING
            finally:
                consumer.shutdown()

    def test_missing_push_reservation_reports_failed_load(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_device = "cpu"
        spec = ECMooncakeLoadSpec(
            mm_hash="hash",
            nbytes=32,
            shape=(8,),
            dtype="float32",
            transfer_id="missing-transfer",
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
