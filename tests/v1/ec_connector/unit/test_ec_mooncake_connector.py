# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECMooncakeConnector and its HTTP registry."""

from __future__ import annotations

import ctypes
import socket
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakeRegistryServer,
    _ContiguousAllocator,
)
from vllm.v1.core.sched.output import SchedulerOutput

pytest_plugins = ("tests.v1.ec_connector.unit.test_ec_example_connector",)


class CopyingFakeTransferEngine:
    def __init__(self, *args, **kwargs):
        self.registered: set[int] = set()
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
        self.register_calls.append(addresses)
        self.registered.update(addresses)
        return 0

    def unregister_memory(self, buffer_address) -> int:
        address = int(buffer_address)
        self.unregister_calls.append(address)
        self.registered.discard(address)
        return 0

    def batch_unregister_memory(self, buffer_addresses) -> int:
        addresses = [int(addr) for addr in buffer_addresses]
        self.batch_unregister_calls.append(addresses)
        self.registered.difference_update(addresses)
        return 0


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


@pytest.fixture
def mock_vllm_config_producer():
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = True
    config.ec_transfer_config.is_ec_consumer = False
    config.ec_transfer_config.ec_buffer_device = "cuda"
    config.ec_transfer_config.ec_buffer_size = 1e9
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
        "registry_http_port": 19018,
    }
    return config


@pytest.fixture
def mock_vllm_config_consumer():
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = False
    config.ec_transfer_config.is_ec_consumer = True
    config.ec_transfer_config.ec_buffer_device = "cuda"
    config.ec_transfer_config.ec_buffer_size = 1e9
    config.ec_transfer_config.ec_connector_extra_config = {
        "mooncake_protocol": "tcp",
        "remote_registry_url": "http://127.0.0.1:19018",
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
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector.is_local_first_rank",
            return_value=True,
        ),
    ):
        yield


class TestECMooncakeRegistryServer:
    def test_publish_and_lookup(self):
        port = _find_free_port()
        registry = ECMooncakeRegistryServer("127.0.0.1", port)
        registry.start()
        try:
            payload = {
                "nbytes": 128,
                "shape": [4, 8],
                "dtype": "float32",
                "producer_zmq": "tcp://127.0.0.1:9999",
            }
            registry.publish("hash_a", payload)
            r = httpx.get(f"http://127.0.0.1:{port}/ec/info/hash_a", timeout=2.0)
            assert r.status_code == 200
            data = r.json()
            lease_id = data.pop("lease_id")
            assert data == payload
            assert registry.consume_lease("hash_a", lease_id)
            r404 = httpx.get(f"http://127.0.0.1:{port}/ec/info/missing", timeout=2.0)
            assert r404.status_code == 404
        finally:
            registry.shutdown()

    def test_unpublish_removes_entry(self):
        port = _find_free_port()
        registry = ECMooncakeRegistryServer("127.0.0.1", port)
        registry.start()
        try:
            registry.publish("h", {"nbytes": 1, "shape": [], "dtype": "float32"})
            registry.unpublish("h")
            r = httpx.get(f"http://127.0.0.1:{port}/ec/info/h", timeout=2.0)
            assert r.status_code == 404
        finally:
            registry.shutdown()


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
    def test_consumer_scheduler_requires_remote_registry(
        self, mock_vllm_config_consumer
    ):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        with (
            patch_ec_mooncake_deps(),
            pytest.raises(ValueError, match="remote_registry_url"),
        ):
            ECMooncakeConnector(mock_vllm_config_consumer, ECConnectorRole.SCHEDULER)

    def test_rejects_tensor_parallel_gt_one(self, mock_vllm_config_producer):
        mock_vllm_config_producer.parallel_config.tensor_parallel_size = 2
        with (
            patch_ec_mooncake_deps(),
            pytest.raises(ValueError, match="tensor_parallel_size"),
        ):
            ECMooncakeConnector(mock_vllm_config_producer, ECConnectorRole.WORKER)


class TestECMooncakeSchedulerMetadata:
    def test_has_cache_item_queries_registry(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        port = _find_free_port()
        registry = ECMooncakeRegistryServer("127.0.0.1", port)
        registry.start()
        try:
            mm_hash = mock_request_with_3_mm.mm_features[0].identifier
            registry.publish(
                mm_hash,
                {
                    "nbytes": 64,
                    "shape": [2, 4],
                    "dtype": "float32",
                    "producer_zmq": "tcp://127.0.0.1:1",
                },
            )
            mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
                "remote_registry_url"
            ] = f"http://127.0.0.1:{port}"
            with patch_ec_mooncake_deps():
                scheduler = ECMooncakeConnector(
                    mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
                )
                assert scheduler.has_cache_item(mm_hash)
                assert mm_hash in scheduler._pending_specs
                spec = scheduler._pending_specs[mm_hash]
                assert spec.shape == (2, 4)
                assert spec.dtype == "float32"
        finally:
            registry.shutdown()

    def test_has_cache_item_missing_returns_false(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        port = _find_free_port()
        registry = ECMooncakeRegistryServer("127.0.0.1", port)
        registry.start()
        try:
            mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
                "remote_registry_url"
            ] = f"http://127.0.0.1:{port}"
            with patch_ec_mooncake_deps():
                scheduler = ECMooncakeConnector(
                    mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
                )
                mm_hash = mock_request_with_3_mm.mm_features[0].identifier
                assert not scheduler.has_cache_item(mm_hash)
        finally:
            registry.shutdown()

    def test_build_connector_meta_clears_pending(
        self, mock_vllm_config_consumer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
            )
            mm_hash = mock_request_with_3_mm.mm_features[0].identifier
            scheduler._pending_specs[mm_hash] = ECMooncakeLoadSpec(
                mm_hash=mm_hash,
                num_token=0,
                nbytes=32,
                shape=(2, 4),
                dtype="float32",
                producer_zmq="tcp://127.0.0.1:1",
                lease_id="lease",
            )
            scheduler._mm_datas_need_loads[mm_hash] = 100
            meta = scheduler.build_connector_meta(Mock(spec=SchedulerOutput))
            assert isinstance(meta, ECMooncakeConnectorMetadata)
            assert len(meta.loads) == 1
            assert meta.loads[0].mm_hash == mm_hash
            assert meta.loads[0].num_token == 100
            assert scheduler._mm_datas_need_loads == {}
            assert mm_hash not in scheduler._pending_specs

    def test_producer_does_not_build_load_metadata(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            scheduler.update_state_after_alloc(mock_request_with_3_mm, 0)
            meta = scheduler.build_connector_meta(Mock(spec=SchedulerOutput))

        assert isinstance(meta, ECMooncakeConnectorMetadata)
        assert meta.loads == []

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
    @pytest.mark.skipif(
        not torch.accelerator.is_available(),
        reason="Requires an accelerator for registered pool",
    )
    def test_consumer_reuses_registered_cuda_pool(self, mock_vllm_config_consumer):
        mock_vllm_config_consumer.ec_transfer_config.ec_buffer_size = 4096
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config[
            "consumer_buffer_pool_size"
        ] = 4096
        specs = [
            ECMooncakeLoadSpec(
                mm_hash=f"hash_{index}",
                num_token=1,
                nbytes=256,
                shape=(32, 2),
                dtype="float32",
                producer_zmq="tcp://127.0.0.1:1",
                lease_id=f"lease_{index}",
            )
            for index in range(2)
        ]

        with patch_ec_mooncake_deps():
            consumer = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            consumer.bind_connector_metadata(ECMooncakeConnectorMetadata(loads=specs))
            cache: dict[str, torch.Tensor] = {}
            with patch.object(consumer, "_send_pull", return_value={"ok": True}):
                consumer.start_load_caches(cache)

            engine = consumer._engine
            pool = consumer._consumer_pool
            assert isinstance(engine, CopyingFakeTransferEngine)
            assert pool is not None
            assert engine.register_calls == [[pool.data_ptr()]]
            assert engine.batch_unregister_calls == []
            assert cache["hash_0"].data_ptr() == pool.data_ptr()
            assert cache["hash_1"].data_ptr() == pool.data_ptr() + 256

            cache.clear()
            consumer._release_stale_consumer_allocations(cache)
            torch.accelerator.synchronize()
            consumer._poll_consumer_pool_frees()
            assert consumer._consumer_pool_allocator is not None
            assert consumer._consumer_pool_allocator.allocate(4096) == (0, 4096)
            consumer.shutdown()

    def test_single_process_save_and_load(self, mock_vllm_config_producer):
        """Host-memory pull path (fake engine uses memcpy; CUDA ptrs need e2e)."""
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        mm_hash = "unit_test_hash"
        torch.manual_seed(7)
        source = torch.randn(4, 16, dtype=torch.float32)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            producer.save_caches({mm_hash: source}, mm_hash)
            for _ in range(100):
                if producer._zmq_listen_addr is not None:
                    break
                time.sleep(0.01)
            assert producer._zmq_listen_addr is not None

            url = f"http://127.0.0.1:{port}/ec/info/{mm_hash}"
            r = httpx.get(url, timeout=2.0)
            assert r.status_code == 200
            data = r.json()

            consumer_cfg = Mock(spec=VllmConfig)
            consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
            consumer_cfg.ec_transfer_config = Mock()
            consumer_cfg.ec_transfer_config.is_ec_producer = False
            consumer_cfg.ec_transfer_config.is_ec_consumer = True
            consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
            consumer_cfg.ec_transfer_config.ec_buffer_size = 1e9
            consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
                "mooncake_protocol": "tcp",
            }
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)
            spec = ECMooncakeLoadSpec(
                mm_hash=mm_hash,
                num_token=1,
                nbytes=int(data["nbytes"]),
                shape=tuple(int(x) for x in data["shape"]),
                dtype=str(data["dtype"]),
                producer_zmq=str(data["producer_zmq"]),
                lease_id=str(data["lease_id"]),
            )
            meta = ECMooncakeConnectorMetadata()
            meta.add_load(spec)
            consumer.bind_connector_metadata(meta)
            loaded: dict[str, torch.Tensor] = {}
            consumer.start_load_caches(loaded)
            assert mm_hash in loaded
            assert torch.allclose(loaded[mm_hash].cpu(), source.cpu())
            consumer_engine = consumer._engine
            assert isinstance(consumer_engine, CopyingFakeTransferEngine)
            assert consumer_engine.registered == set()
            assert consumer_engine.batch_unregister_calls == [
                [loaded[mm_hash].data_ptr()]
            ]
            consumer.shutdown()
            producer.shutdown()

    def test_batches_multi_item_transfer_and_reuses_socket(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        sources = {f"hash_{i}": torch.randn(4, 16) for i in range(3)}

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            for mm_hash, tensor in sources.items():
                producer.save_caches({mm_hash: tensor}, mm_hash)

            consumer_cfg = Mock(spec=VllmConfig)
            consumer_cfg.parallel_config = mock_vllm_config_producer.parallel_config
            consumer_cfg.ec_transfer_config = Mock()
            consumer_cfg.ec_transfer_config.is_ec_producer = False
            consumer_cfg.ec_transfer_config.is_ec_consumer = True
            consumer_cfg.ec_transfer_config.ec_buffer_device = "cpu"
            consumer_cfg.ec_transfer_config.ec_buffer_size = 1e9
            consumer_cfg.ec_transfer_config.ec_connector_extra_config = {
                "mooncake_protocol": "tcp"
            }
            consumer = ECMooncakeConnector(consumer_cfg, ECConnectorRole.WORKER)

            def make_spec(mm_hash: str) -> ECMooncakeLoadSpec:
                data = httpx.get(f"http://127.0.0.1:{port}/ec/info/{mm_hash}").json()
                return ECMooncakeLoadSpec(
                    mm_hash=mm_hash,
                    num_token=1,
                    nbytes=int(data["nbytes"]),
                    shape=tuple(data["shape"]),
                    dtype=str(data["dtype"]),
                    producer_zmq=str(data["producer_zmq"]),
                    lease_id=str(data["lease_id"]),
                )

            first_meta = ECMooncakeConnectorMetadata(
                loads=[make_spec("hash_0"), make_spec("hash_1")]
            )
            consumer.bind_connector_metadata(first_meta)
            loaded: dict[str, torch.Tensor] = {}
            consumer.start_load_caches(loaded)
            socket = next(iter(consumer._client_sockets.values()))

            second_meta = ECMooncakeConnectorMetadata(loads=[make_spec("hash_2")])
            consumer.bind_connector_metadata(second_meta)
            consumer.start_load_caches(loaded)

            producer_engine = producer._engine
            consumer_engine = consumer._engine
            assert isinstance(producer_engine, CopyingFakeTransferEngine)
            assert isinstance(consumer_engine, CopyingFakeTransferEngine)
            assert producer_engine.transfer_calls == [[256, 256], [256]]
            assert len(consumer._client_sockets) == 1
            assert next(iter(consumer._client_sockets.values())) is socket
            assert all(
                torch.equal(loaded[key], value) for key, value in sources.items()
            )
            register_sizes = [len(call) for call in consumer_engine.register_calls]
            assert register_sizes == [2, 1]
            assert [len(call) for call in consumer_engine.batch_unregister_calls] == [
                2,
                1,
            ]
            consumer.shutdown()
            producer.shutdown()

    def test_producer_evicts_lru_registration_at_capacity(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = 32
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        first = torch.randn(8)
        second = torch.randn(8)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                producer.save_caches({"first": first}, "first")
                producer.save_caches({"second": second}, "second")

                engine = producer._engine
                assert isinstance(engine, CopyingFakeTransferEngine)
                assert list(producer._tensor_by_hash) == ["second"]
                assert producer._registered_bytes == second.nbytes
                assert first.data_ptr() in engine.unregister_calls
                assert second.data_ptr() in engine.registered

                base_url = f"http://127.0.0.1:{port}/ec/info"
                assert httpx.get(f"{base_url}/first").status_code == 404
                assert httpx.get(f"{base_url}/second").status_code == 200
            finally:
                producer.shutdown()

    def test_producer_does_not_evict_in_flight_registration(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = 32
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        first = torch.randn(8)
        second = torch.randn(8)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                producer.save_caches({"first": first}, "first")
                producer._tensor_by_hash["first"].in_flight = 1
                with pytest.raises(RuntimeError, match="no evictable"):
                    producer.save_caches({"second": second}, "second")
                assert list(producer._tensor_by_hash) == ["first"]
            finally:
                producer._tensor_by_hash["first"].in_flight = 0
                producer.shutdown()

    def test_producer_does_not_evict_leased_registration(
        self, mock_vllm_config_producer
    ):
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_size = 32
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        first = torch.randn(8)
        second = torch.randn(8)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            try:
                producer.save_caches({"first": first}, "first")
                response = httpx.get(f"http://127.0.0.1:{port}/ec/info/first").json()

                with pytest.raises(RuntimeError, match="no evictable"):
                    producer.save_caches({"second": second}, "second")

                assert producer._registry is not None
                assert producer._registry.consume_lease("first", response["lease_id"])
                producer.save_caches({"second": second}, "second")
                assert list(producer._tensor_by_hash) == ["second"]
            finally:
                producer.shutdown()

    def test_shutdown_unregisters_all_producer_tensors(self, mock_vllm_config_producer):
        port = _find_free_port()
        mock_vllm_config_producer.ec_transfer_config.ec_buffer_device = "cpu"
        mock_vllm_config_producer.ec_transfer_config.ec_connector_extra_config[
            "registry_http_port"
        ] = port
        tensor = torch.randn(8)

        with patch_ec_mooncake_deps():
            producer = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.WORKER
            )
            producer.save_caches({"hash": tensor}, "hash")
            engine = producer._engine
            assert isinstance(engine, CopyingFakeTransferEngine)

            producer.shutdown()

            assert engine.batch_unregister_calls == [[tensor.data_ptr()]]
            assert engine.registered == set()
            assert producer._tensor_by_hash == {}
            assert producer._registered_bytes == 0

    def test_producer_scheduler_has_cache_item_false(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        with patch_ec_mooncake_deps():
            scheduler = ECMooncakeConnector(
                mock_vllm_config_producer, ECConnectorRole.SCHEDULER
            )
            mm_hash = mock_request_with_3_mm.mm_features[0].identifier
            assert not scheduler.has_cache_item(mm_hash)

    def test_consumer_worker_save_is_noop(self, mock_vllm_config_consumer):
        with patch_ec_mooncake_deps():
            worker = ECMooncakeConnector(
                mock_vllm_config_consumer, ECConnectorRole.WORKER
            )
            mm_hash = "noop_hash"
            tensor = torch.randn(2, 4)
            worker.save_caches({mm_hash: tensor}, mm_hash)
            assert mm_hash not in worker._tensor_by_hash
