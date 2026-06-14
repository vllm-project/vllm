# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECMooncakeConnector and its HTTP registry."""

from __future__ import annotations

import ctypes
import socket
import time
from contextlib import contextmanager
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
)
from vllm.v1.core.sched.output import SchedulerOutput

from tests.v1.ec_connector.unit.test_ec_example_connector import (
    mock_request_with_3_mm,
)


class CopyingFakeTransferEngine:
    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        for src, dst, nbytes in zip(buffers, peer_buffer_addresses, lengths):
            ctypes.memmove(int(dst), int(src), int(nbytes))
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
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
            assert r.json() == payload
            r404 = httpx.get(
                f"http://127.0.0.1:{port}/ec/info/missing", timeout=2.0
            )
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


class TestECMooncakeConnectorValidation:
    def test_consumer_scheduler_requires_remote_registry(self, mock_vllm_config_consumer):
        mock_vllm_config_consumer.ec_transfer_config.ec_connector_extra_config = {
            "mooncake_protocol": "tcp",
        }
        with patch_ec_mooncake_deps():
            with pytest.raises(ValueError, match="remote_registry_url"):
                ECMooncakeConnector(
                    mock_vllm_config_consumer, ECConnectorRole.SCHEDULER
                )

    def test_rejects_tensor_parallel_gt_one(self, mock_vllm_config_producer):
        mock_vllm_config_producer.parallel_config.tensor_parallel_size = 2
        with patch_ec_mooncake_deps():
            with pytest.raises(ValueError, match="tensor_parallel_size"):
                ECMooncakeConnector(
                    mock_vllm_config_producer, ECConnectorRole.WORKER
                )


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
            )
            scheduler._mm_datas_need_loads[mm_hash] = 100
            meta = scheduler.build_connector_meta(Mock(spec=SchedulerOutput))
            assert isinstance(meta, ECMooncakeConnectorMetadata)
            assert len(meta.loads) == 1
            assert meta.loads[0].mm_hash == mm_hash
            assert meta.loads[0].num_token == 100
            assert scheduler._mm_datas_need_loads == {}
            assert mm_hash not in scheduler._pending_specs


class TestECMooncakeWorkerTransfer:
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
            )
            meta = ECMooncakeConnectorMetadata()
            meta.add_load(spec)
            consumer.bind_connector_metadata(meta)
            loaded: dict[str, torch.Tensor] = {}
            consumer.start_load_caches(loaded)
            assert mm_hash in loaded
            assert torch.allclose(loaded[mm_hash].cpu(), source.cpu())

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
