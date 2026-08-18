# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECMooncakeConnector."""

from __future__ import annotations

import copy
import ctypes
import socket
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    _LEASE_TTL_SECONDS,
    ECMooncakeConnector,
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
    _ConsumerPoolAllocation,
    _ContiguousAllocator,
)
from vllm.v1.core.sched.output import SchedulerOutput

pytest_plugins = ("tests.v1.ec_connector.unit.test_ec_example_connector",)


class CopyingFakeTransferEngine:
    def __init__(self, *args, **kwargs):
        self.registered: set[int] = set()
        self.regions: dict[int, int] = {}
        self.register_calls: list[list[int]] = []
        self.unregister_calls: list[int] = []
        self.batch_unregister_calls: list[list[int]] = []
        self.transfer_calls: list[list[int]] = []

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        self.transfer_calls.append([int(length) for length in lengths])
        for src, dst, nbytes in zip(buffers, peer_buffer_addresses, lengths):
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


@pytest.fixture
def mock_vllm_config_producer():
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.data_parallel_size = 1
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
            "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector.TransferEngine",
            CopyingFakeTransferEngine,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector._MOONCAKE_IMPORT_ERROR",
            None,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector.get_ip",
            return_value="127.0.0.1",
        ),
    ):
        yield


class TestECMooncakeFactory:
    def test_factory_registers_connector(self):
        cls = ECConnectorFactory.get_connector_class(
            Mock(ec_connector="ECMooncakeConnector")
        )
        assert cls.__name__ == "ECMooncakeConnector"


class TestContiguousAllocator:
    def test_reuses_and_coalesces_contiguous_regions(self):
        allocator = _ContiguousAllocator(1024, alignment=256)

        first = allocator.allocate(1)
        second = allocator.allocate(300)
        assert first == (0, 256)
        assert second == (256, 512)
        assert allocator.allocate(300) is None

        allocator.free(*first)
        allocator.free(*second)
        assert allocator.allocate(1024) == (0, 1024)


class TestECMooncakeConnectorValidation:
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

    def test_rejects_data_parallel(self, mock_vllm_config_consumer):
        """One control channel per instance cannot address a replica."""
        mock_vllm_config_consumer.parallel_config.data_parallel_size = 2
        with (
            patch_ec_mooncake_deps(),
            pytest.raises(ValueError, match="data parallelism"),
        ):
            ECMooncakeConnector(mock_vllm_config_consumer, ECConnectorRole.SCHEDULER)


class TestECMooncakeWorkerMetadataAggregation:
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


class TestECMooncakeSchedulerMetadata:
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
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0)
                mm_hash = request.mm_features[0].identifier
                assert scheduler._consumer_scheduler_metrics["missing_event"] == 1
                assert mm_hash in scheduler._consumer_missing_since
            finally:
                scheduler.shutdown()

    def test_item_with_no_transfer_in_flight_is_reported_as_stalled(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        """A push that never arrives must not wait silently forever."""
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
            "reservation_zmq_port": 19019,
            "push_wait_timeout_s": 0,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0)
                    assert not scheduler.ensure_cache_available(request, 0)
                assert scheduler._consumer_scheduler_metrics["stalled"] == 1
                assert mm_hash in scheduler._stalled_hashes
                # The stall is reported once, not once per scheduling pass.
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0)
                assert scheduler._consumer_scheduler_metrics["stalled"] == 1
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
                    scheduler._index_pending_spec(spec)
                scheduler._pop_pending_spec("transfer-0")
                assert "hash" in scheduler._consumer_pending_since
                scheduler._pop_pending_spec("transfer-1")
                assert "hash" not in scheduler._consumer_pending_since
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
                scheduler._index_pending_spec(
                    ECMooncakeLoadSpec(
                        mm_hash=mm_hash,
                        num_token=0,
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        pushed=True,
                        transfer_id="request-transfer",
                        reservation_id="reservation",
                    )
                )
                with (
                    patch.object(scheduler, "_drain_push_notifications"),
                    patch.object(scheduler, "_queue_cancel") as cancel,
                ):
                    assert scheduler.ensure_cache_available(request, 0, {mm_hash})
                    cancel.assert_not_called()
                    assert "request-transfer" in scheduler._pending_specs

                    # Once the entry is evicted the request can still get it.
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert mm_hash in scheduler._loading_hashes
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
                scheduler._index_pending_spec(
                    ECMooncakeLoadSpec(
                        mm_hash=mm_hash,
                        num_token=0,
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                        pushed=True,
                        transfer_id="consumed-transfer",
                        reservation_id="reservation",
                    )
                )
                with patch.object(scheduler, "_queue_cancel") as cancel:
                    scheduler.update_state_after_free(request, 0)
                # Cancelled by transfer: a shard's reservation id means
                # nothing to its peers, so it is not passed along.
                cancel.assert_called_once_with("consumed-transfer")
                assert "consumed-transfer" not in scheduler._pending_specs
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
                scheduler._ready_hashes.add(mm_hash)
                scheduler._event_zmq_socket = Mock()
                scheduler._event_zmq_socket.recv_json.side_effect = [
                    event,
                    zmq.Again(),
                ]
                with patch.object(scheduler, "_queue_cancel") as cancel:
                    scheduler._drain_push_notifications()
                cancel.assert_not_called()
                assert "later-transfer" in scheduler._pending_specs

                # The scheduler frees the encoder cache entry.
                scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[mm_hash])
                )
                assert mm_hash not in scheduler._ready_hashes

                # The request that owns the transfer can still pick it up.
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert mm_hash in scheduler._loading_hashes
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
            "push_wait_timeout_s": 0,
        }
        request = mock_request_with_3_mm
        request.mm_features = request.mm_features[:1]
        mm_hash = request.mm_features[0].identifier

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            try:
                with (
                    patch.object(scheduler, "_drain_push_notifications"),
                    patch.object(scheduler, "_send_control", return_value=None),
                ):
                    assert not scheduler.ensure_cache_available(request, 0, set())
                assert scheduler.take_unavailable_requests() == {request.request_id}
                # Draining clears it: the scheduler acts on each id once.
                assert scheduler.take_unavailable_requests() == set()
                # A re-issued request gets a fresh window rather than the
                # expired one, or it would fail before its push could land.
                assert mm_hash not in scheduler._consumer_missing_since
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
                scheduler._reservation_zmq_addr = f"tcp://127.0.0.1:{ports[0]}"
                with patch.object(
                    scheduler, "_send_control", side_effect=fake_send
                ) as send_control:
                    scheduler._ensure_event_channel()

                subscribed = [
                    call.args[0]
                    for call in send_control.call_args_list
                    if call.args[1]["op"] == "event_port"
                ]
                assert len(subscribed) == len(ports)
                assert scheduler._event_shard_count == len(ports)

                event = {"transfer_id": "transfer-0"}
                assert not scheduler._note_shard_ready({**event, "shard": ports[0]})
                # The same rank reporting twice is not two ranks.
                assert not scheduler._note_shard_ready({**event, "shard": ports[0]})
                assert not scheduler._note_shard_ready({**event, "shard": ports[1]})
                assert scheduler._note_shard_ready({**event, "shard": ports[2]})
                # Nothing is retained once the transfer is handed on.
                assert "transfer-0" not in scheduler._event_ready_shards
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
                scheduler._event_zmq_socket = Mock()
                scheduler._event_zmq_socket.recv_json.side_effect = [
                    {
                        "mm_hash": mm_hash,
                        "transfer_id": "only-transfer",
                        "ready": True,
                        "reservation_id": "r0",
                        "nbytes": 16,
                        "shape": [4],
                        "dtype": "float32",
                    },
                    zmq.Again(),
                ]
                scheduler._drain_push_notifications()

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
                assert not scheduler._pending_specs

                # The encoder cache evicts the entry.
                scheduler.build_connector_meta(
                    SimpleNamespace(free_encoder_mm_hashes=[mm_hash])
                )

                # The second request has no transfer of its own, and the only
                # transfer is spent. It must still be served.
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert scheduler.has_cache_item(mm_hash)
                    assert not scheduler.ensure_cache_available(second, 0, set())
                assert mm_hash in scheduler._loading_hashes
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
                scheduler._note_resident(
                    ECMooncakeLoadSpec(
                        mm_hash=mm_hash,
                        num_token=0,
                        nbytes=16,
                        shape=(4,),
                        dtype="float32",
                    )
                )
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert scheduler.has_cache_item(mm_hash)

                scheduler.update_connector_output(
                    SimpleNamespace(
                        ec_connector_worker_meta=ECMooncakeWorkerMetadata(
                            reclaimed={mm_hash}
                        )
                    )
                )
                with patch.object(scheduler, "_drain_push_notifications"):
                    assert not scheduler.has_cache_item(mm_hash)
                assert scheduler._resident_bytes == 0
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
            scheduler._event_zmq_socket = Mock()
            scheduler._event_zmq_socket.recv_json.side_effect = [event, zmq.Again()]
            scheduler._loading_hashes.add("hash")

            scheduler._drain_push_notifications()

            assert scheduler._pending_specs["next-transfer"].reservation_id == "next"

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
            scheduler._index_pending_spec(load_spec)
            scheduler._load_specs[mm_hash] = load_spec
            scheduler._mm_datas_need_loads[mm_hash] = 100
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )
            assert isinstance(meta, ECMooncakeConnectorMetadata)
            assert len(meta.loads) == 1
            assert meta.loads[0].mm_hash == mm_hash
            assert meta.loads[0].num_token == 100
            assert scheduler._mm_datas_need_loads == {}
            assert "transfer" not in scheduler._pending_specs

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
        request.ec_transfer_params = {
            "consumer_zmq": "tcp://decode:19019",
            "ec_items": [{"mm_hash": "img_hash_1", "transfer_id": "transfer-1"}],
        }
        mock_vllm_config_producer.model_config.dtype = torch.float32
        mock_vllm_config_producer.model_config.get_hidden_size.return_value = 16

        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            scheduler.update_state_after_alloc(request, 0)
            meta = scheduler.build_connector_meta(
                Mock(spec=SchedulerOutput, free_encoder_mm_hashes=[])
            )

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
                scheduler,
                "_placeholder_metadata_fields",
                return_value={"image_grid_thw"},
            ):
                delay_free, params = scheduler.request_finished(request)

        assert not delay_free
        assert params == {
            "ec_items": [{"mm_hash": "image_uuid", "image_grid_thw": [1, 32, 48]}]
        }


class TestECMooncakeWorkerTransfer:
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

                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert len(engine.transfer_calls) == 1
                assert sorted(engine.transfer_calls[0]) == sorted(
                    tensor.nbytes for tensor in sources.values()
                )
                assert all(
                    reservation.ready
                    for reservation in consumer._push_reservations.values()
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
                _, reservation = producer._pending_reservations["hash"][0]
                shards = reservation.result(timeout=2)
                # One reservation per consumer shard; this consumer is single.
                assert len(shards) == 1
                reservation_data = shards[0]
                assert reservation_data["nbytes"] == source.nbytes
                old_reservation_id = reservation_data["reservation_id"]
                reservation_data["_received_at"] -= _LEASE_TTL_SECONDS
                consumer._push_reservations["transfer-1"].expires_at = 0
                with patch.object(
                    scheduler,
                    "_send_control",
                    wraps=scheduler._send_control,
                ) as send_control:
                    assert not scheduler.has_cache_item("hash")
                    assert not scheduler.has_cache_item("hash")
                    # The channel is built once, not per call: the roster is
                    # fetched and every shard subscribed to on the first one.
                    assert [call.args[1] for call in send_control.call_args_list] == [
                        {"op": "peers"},
                        {"op": "event_port"},
                    ]
                    assert "transfer-1" in consumer._push_reservations

                    producer.save_caches({"hash": source}, "hash")
                    _wait_for_worker_io(producer)
                    assert (
                        consumer._push_reservations["transfer-1"].reservation_id
                        != old_reservation_id
                    )
                    deadline = time.monotonic() + 2
                    while not scheduler.has_cache_item("hash"):
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    # Still just the two setup requests: polling for readiness
                    # must not re-open the channel.
                    assert send_control.call_count == 2
                load = scheduler._pending_specs["transfer-1"]
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[load])
                )
                loaded: dict[str, torch.Tensor] = {}
                consumer.start_load_caches(loaded)
                first_meta = consumer.build_connector_worker_meta()
                assert first_meta.loaded == {"hash"}
                assert torch.equal(loaded["hash"], source)
                consumer_engine = consumer._engine
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
                _, reservation = producer._pending_reservations["hash"][0]
                reservation.result(timeout=2)
                assert "transfer-1" in consumer._push_reservations

                producer.get_finished({"request-1"})
                _wait_for_worker_io(producer)
                assert "hash" not in producer._pending_reservations
                assert "transfer-1" not in consumer._push_reservations
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

                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert engine.transfer_calls == [[source.nbytes]]
                reservation = consumer._push_reservations["transfer-1"]
                assert reservation.ready
                assert consumer._consumer_worker_metrics["completions_accepted"] == 1
                assert consumer._consumer_worker_metrics["completions_repeated"] == 0

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                load = scheduler._pop_pending_spec("transfer-1")
                assert load is not None
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
                cached = consumer._push_reservations["transfer-2"]
                assert cached.ready and not cached.owns_allocation
                assert consumer._consumer_worker_metrics["reservations_cached"] == 1

                deadline = time.monotonic() + 2
                while not scheduler.has_cache_item("hash"):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                cached_load = scheduler._pop_pending_spec("transfer-2")
                assert cached_load is not None
                cached_load.num_token = 4
                consumer.bind_connector_metadata(
                    ECMooncakeConnectorMetadata(loads=[cached_load])
                )
                consumer.start_load_caches(loaded)
                cached_meta = consumer.build_connector_worker_meta()
                assert cached_meta.loaded == {"hash"}
                assert "transfer-2" not in consumer._push_reservations
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                pool = consumer._consumer_pool
                allocator = consumer._consumer_pool_allocator
                assert pool is not None and allocator is not None
                offset, size = allocator.allocate(spec.nbytes)
                tensor = (
                    pool.narrow(0, offset, spec.nbytes).view(torch.float32).view(4, 4)
                )
                allocation = _ConsumerPoolAllocation(offset, size, tensor)
                consumer._consumer_residents.insert("hash", allocation, size)
                consumer._consumer_residents.retire("hash")

                # A later push reserves the retired copy instead of transferring.
                consumer._reserve_push_destination(
                    {
                        "transfer_id": "t1",
                        "mm_hash": "hash",
                        "nbytes": spec.nbytes,
                        "shape": list(spec.shape),
                        "dtype": spec.dtype,
                    }
                )
                assert consumer._consumer_residents.num_evictable == 0

                assert consumer._take_resident_tensor(spec) is tensor
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
                with patch.object(producer, "_send_control", side_effect=fake_send):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    _wait_for_worker_io(producer)

                engine = producer._engine
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

                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                # The staging pool is registered once; a transfer registers
                # nothing of its own.
                pool = producer._producer_pool
                assert pool is not None
                assert engine.register_calls == [[pool.data_ptr()]]
                assert engine.batch_unregister_calls == []
                assert engine.transfer_calls == [[source.nbytes, source.nbytes]]
                assert all(
                    reservation.ready
                    for reservation in consumer._push_reservations.values()
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
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "producer_buffer_pool_size"
        ] = 0
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
                producer.start_save_caches(encoder_cache={"hash": source})
                _wait_for_worker_io(producer)
                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert producer._producer_pool is None
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
                first = producer._acquire_push_source_registrations([source])
                second = producer._acquire_push_source_registrations([source])
                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert len(engine.register_calls) == 1

                producer._release_push_source_registrations(first)
                assert engine.batch_unregister_calls == []
                producer._release_push_source_registrations(second)
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
                    producer, "_send_control", wraps=producer._send_control
                ) as send_control:
                    producer.start_save_caches(encoder_cache=sources)
                    _wait_for_worker_io(producer)
                ops = [call.args[1]["op"] for call in send_control.call_args_list]
                assert ops.count("complete_batch") == 1
                assert "complete" not in ops
                assert all(
                    reservation.ready
                    for reservation in consumer._push_reservations.values()
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
                engine = producer._engine or producer._ensure_engine()
                with patch.object(engine, "batch_transfer_sync_write", return_value=1):
                    producer.start_save_caches(encoder_cache={"hash": source})
                    # No raise: the batch reports itself and gives up the
                    # consumer-side reservation.
                    _wait_for_worker_io(producer)
                assert "transfer" not in consumer._push_reservations
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                reservation = consumer._reserve_push_destination(
                    {
                        "mm_hash": "hash",
                        "transfer_id": "transfer-1",
                        "nbytes": 64,
                        "shape": [4, 4],
                        "dtype": "float32",
                    }
                )
                reservation_id = reservation["reservation_id"]

                first = consumer._complete_push("transfer-1", reservation_id)
                repeated = consumer._complete_push("transfer-1", reservation_id)

                assert first.accepted and first.became_ready
                assert repeated.accepted and not repeated.became_ready
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                first = consumer._reserve_push_destination(payload("first"))
                second = consumer._reserve_push_destination(payload("second"))

                consumer._complete_push("first", first["reservation_id"])
                assert consumer._push_reservations["first"].ready
                assert not consumer._push_reservations["second"].ready

                assert consumer._cancel_push("first", first["reservation_id"])
                assert "first" not in consumer._push_reservations
                assert "second" in consumer._push_reservations
                assert consumer._complete_push("second", second["reservation_id"])
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                old = consumer._reserve_push_destination(payload)
                consumer._push_reservations["transfer"].expires_at = 0
                consumer._expire_push_reservations()
                new = consumer._reserve_push_destination(payload)

                assert old["reservation_id"] != new["reservation_id"]
                stale = consumer._complete_push("transfer", old["reservation_id"])
                assert not stale.accepted
                assert not consumer._push_reservations["transfer"].ready
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                reservation = consumer._reserve_push_destination(
                    {
                        "mm_hash": "hash",
                        "transfer_id": "transfer-1",
                        "nbytes": 64,
                        "shape": [4, 4],
                        "dtype": "float32",
                    }
                )
                consumer._complete_push("transfer-1", reservation["reservation_id"])
                consumer._push_reservations["transfer-1"].expires_at = 0

                assert consumer._expire_push_reservations() == 1
                assert "transfer-1" not in consumer._push_reservations
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
                consumer._ensure_consumer_pool(torch.device("cpu"), allow_host=True)
                assert consumer._cancel_push("cancelled-transfer", "")
                cancelled = consumer._reserve_push_destination(payload)
                assert cancelled["cancelled"] and not cancelled["write"]
                assert "cancelled-transfer" not in consumer._push_reservations

                consumer._cancelled_transfers["cancelled-transfer"] = 0
                consumer._expire_push_reservations()
                replacement = consumer._reserve_push_destination(payload)
                assert replacement["write"]
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
