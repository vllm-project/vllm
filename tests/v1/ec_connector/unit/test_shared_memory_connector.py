# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ECSharedMemoryConnector."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.ec_transfer import DEFAULT_EC_CONNECTOR_CAPACITY_EMBEDS
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.shared_memory_connector import (
    _HEADER_PREFIX_BYTES,
    _MAGIC,
    ECSharedMemoryConnector,
    ECSharedMemoryConnectorMetadata,
    _shm_name,
    _total_shm_size,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import SchedulerOutput

pytestmark = pytest.mark.cpu_test

_SHM_MODULE = "vllm.distributed.ec_transfer.ec_connector.shared_memory_connector"


class MockRequest:
    def __init__(self, request_id: str, mm_hashes: list[str]):
        self.request_id = request_id
        self.mm_features = []
        for mm_hash in mm_hashes:
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=100),
            )
            self.mm_features.append(feature)


def _make_config(*, is_producer: bool, is_consumer: bool) -> Mock:
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_ec_connector_capacity_embeds = Mock(
        return_value=DEFAULT_EC_CONNECTOR_CAPACITY_EMBEDS
    )
    config.ec_transfer_config.is_ec_producer = is_producer
    config.ec_transfer_config.is_ec_consumer = is_consumer
    config.model_config = Mock()
    config.model_config.get_hidden_size = Mock(return_value=768)
    config.model_config.dtype = torch.float16
    config.kv_transfer_config = None
    return config


@pytest.fixture
def mock_vllm_config():
    return _make_config(is_producer=True, is_consumer=False)


@pytest.fixture
def mock_vllm_config_consumer():
    return _make_config(is_producer=False, is_consumer=True)


@pytest.fixture
def mock_request_with_3_mm():
    return MockRequest("test_req_123", ["hash1", "hash2", "hash3"])


class TestShmName:
    def test_shm_name_normalizes_slash_identifier(self):
        name = _shm_name("my/lora:abc123")
        assert len(name) == 64
        assert "/" not in name

    def test_shm_name_deterministic(self):
        assert _shm_name("hash1") == _shm_name("hash1")
        assert _shm_name("hash1") != _shm_name("hash2")


class TestECSharedMemoryConnectorMetadata:
    def test_init(self):
        metadata = ECSharedMemoryConnectorMetadata()
        assert metadata.mm_hashes == []

    def test_add_mm_hash(self):
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash("hash1")
        metadata.add_mm_hash("hash2")
        assert metadata.mm_hashes == ["hash1", "hash2"]


class TestECSharedMemoryConnectorInit:
    def test_init_with_ec_transfer_config(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        assert connector.role == ECConnectorRole.WORKER
        assert len(connector._mm_hashes_need_loads) == 0

    def test_init_without_ec_transfer_config(self):
        config = Mock(spec=VllmConfig)
        config.ec_transfer_config = None
        with pytest.raises(ValueError, match="ec_transfer_config must be set"):
            ECSharedMemoryConnector(vllm_config=config, role=ECConnectorRole.WORKER)


class TestECSharedMemoryConnectorSerializeDeserialize:
    def test_serialize_deserialize_real(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        original = torch.randn(10, 768)
        serialized = connector._serialize_cache(original)
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.device_type = "cpu"
            restored = connector._deserialize_cache(serialized)
        assert torch.equal(original.cpu(), restored.cpu())

    def test_flat_format_roundtrip_large(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        original = torch.randn(16384, 2048, dtype=torch.bfloat16)
        serialized = connector._serialize_cache(original)
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.device_type = "cpu"
            restored = connector._deserialize_cache(memoryview(serialized))
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert torch.equal(original, restored)


class TestECSharedMemoryConnectorSaveCaches:
    @patch(f"{_SHM_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    def test_save_caches_saves_to_shared_memory(self, mock_tp_rank, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        mm_hash = "test_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}
        payload = connector._serialize_cache(encoder_cache[mm_hash])
        total = _total_shm_size(len(payload))

        with patch("multiprocessing.shared_memory.SharedMemory") as mock_shm:
            mock_shm_instance = MagicMock()
            mock_shm_instance.size = total
            mock_shm_instance.buf = bytearray(total)
            mock_shm.return_value = mock_shm_instance

            connector.save_caches(encoder_cache, mm_hash)

            mock_shm.assert_called()
            assert mm_hash in connector._pending_sends

    def test_save_caches_consumer_skips(self, mock_vllm_config_consumer):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer, role=ECConnectorRole.WORKER
        )
        with patch("multiprocessing.shared_memory.SharedMemory") as mock_shm:
            connector.save_caches({"test_hash": torch.randn(10, 768)}, "test_hash")
            mock_shm.assert_not_called()


class TestECSharedMemoryConnectorLoadCaches:
    def test_start_load_caches_loads_from_shared_memory(
        self, mock_vllm_config_consumer
    ):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer, role=ECConnectorRole.WORKER
        )
        mm_hash = "test_hash"
        tensor = torch.randn(10, 768)
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        serialized = connector._serialize_cache(tensor)
        size_bytes = len(serialized)
        total_size = _total_shm_size(size_bytes)
        mock_buf = bytearray(total_size)
        mock_buf[0:4] = _MAGIC
        mock_buf[4:_HEADER_PREFIX_BYTES] = size_bytes.to_bytes(8, "little")
        mock_buf[_HEADER_PREFIX_BYTES : _HEADER_PREFIX_BYTES + size_bytes] = serialized
        mock_buf[_HEADER_PREFIX_BYTES + size_bytes] = 0

        with patch("multiprocessing.shared_memory.SharedMemory") as mock_shm_class:
            mock_shm = MagicMock()
            mock_shm.size = total_size
            mock_shm.buf = mock_buf
            mock_shm_class.return_value = mock_shm
            with patch("vllm.platforms.current_platform") as mock_platform:
                mock_platform.device_type = "cpu"
                encoder_cache: dict[str, torch.Tensor] = {}
                connector.start_load_caches(encoder_cache)
                assert mm_hash in encoder_cache


class TestECSharedMemoryConnectorHasCacheItem:
    def test_has_cache_item_exists(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.SCHEDULER
        )
        with (
            patch(f"{_SHM_MODULE}.os.path.exists", return_value=True),
            patch(f"{_SHM_MODULE}._is_readable_shm_file", return_value=True),
        ):
            assert connector.has_cache_item("hash1") is True

    def test_has_cache_item_missing(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.SCHEDULER
        )
        with patch(f"{_SHM_MODULE}.os.path.exists", return_value=False):
            assert connector.has_cache_item("hash1") is False

    def test_has_cache_item_rejects_invalid_magic(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.SCHEDULER
        )
        with (
            patch(f"{_SHM_MODULE}.os.path.exists", return_value=True),
            patch(f"{_SHM_MODULE}._is_readable_shm_file", return_value=False),
        ):
            assert connector.has_cache_item("hash1") is False

    def test_start_load_caches_skips_invalid_magic(self, mock_vllm_config_consumer):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer, role=ECConnectorRole.WORKER
        )
        mm_hash = "bad_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        mock_buf = bytearray(64)
        mock_buf[:8] = b"pickle!!"

        with patch("multiprocessing.shared_memory.SharedMemory") as mock_shm_class:
            mock_shm = MagicMock()
            mock_shm.size = len(mock_buf)
            mock_shm.buf = mock_buf
            mock_shm_class.return_value = mock_shm
            encoder_cache: dict[str, torch.Tensor] = {}
            connector.start_load_caches(encoder_cache)
            assert mm_hash not in encoder_cache
            mock_shm.close.assert_called_once()


class TestECSharedMemoryConnectorStateManagement:
    def test_update_state_after_alloc_consumer(self, mock_vllm_config_consumer):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer, role=ECConnectorRole.SCHEDULER
        )
        request = MockRequest("req", ["hash1"])
        with patch.object(connector, "has_cache_item", return_value=True):
            connector.update_state_after_alloc(request, index=0)
        assert "hash1" in connector._mm_hashes_need_loads

    def test_update_state_after_alloc_producer_skips(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.SCHEDULER
        )
        request = MockRequest("req", ["hash1"])
        connector.update_state_after_alloc(request, index=0)
        assert "hash1" not in connector._mm_hashes_need_loads

    def test_build_connector_meta(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.SCHEDULER
        )
        connector._mm_hashes_need_loads.update({"hash1", "hash2"})
        metadata = connector.build_connector_meta(Mock(spec=SchedulerOutput))
        assert isinstance(metadata, ECSharedMemoryConnectorMetadata)
        assert set(metadata.mm_hashes) == {"hash1", "hash2"}
        assert len(connector._mm_hashes_need_loads) == 0


class TestECSharedMemoryConnectorLifecycle:
    def test_get_finished_producer_ack(self, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        mm_hash = "h1"
        connector._pending_sends.add(mm_hash)
        ser = connector._serialize_cache(torch.randn(2, 3))
        sz = len(ser)
        buf = bytearray(8 + sz + 1)
        buf[:8] = sz.to_bytes(8, "little")
        buf[8 : 8 + sz] = ser
        buf[8 + sz] = 1
        with (
            patch(f"{_SHM_MODULE}.os.path.exists", return_value=True),
            patch(f"{_SHM_MODULE}.os.open", return_value=5),
            patch(f"{_SHM_MODULE}.os.pread") as mock_pread,
            patch(f"{_SHM_MODULE}.os.close"),
            patch(f"{_SHM_MODULE}.os.fstat") as mock_fstat,
        ):
            mock_fstat.return_value.st_size = _HEADER_PREFIX_BYTES + sz + 1
            mock_pread.side_effect = [_MAGIC, sz.to_bytes(8, "little"), bytes([1])]
            sending, recving = connector.get_finished(set())
        assert sending is not None and mm_hash in sending
        assert recving is None

    def test_get_finished_consumer_pending_recvs(self, mock_vllm_config_consumer):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer, role=ECConnectorRole.WORKER
        )
        connector._pending_recvs.update({"a", "b"})
        sending, recving = connector.get_finished(set())
        assert sending is None
        assert recving == {"a", "b"}

    @patch(f"{_SHM_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    def test_free_physical_cache_unlinks(self, mock_tp_rank, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        connector._pending_sends.add("x")
        with patch.object(connector, "_unlink_shm") as mock_unlink:
            connector.free_physical_cache("x")
            mock_unlink.assert_called_once_with("x")
        assert "x" not in connector._pending_sends

    @patch(f"{_SHM_MODULE}.get_tensor_model_parallel_rank", return_value=1)
    def test_free_physical_cache_skips_non_tp0(self, mock_tp_rank, mock_vllm_config):
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config, role=ECConnectorRole.WORKER
        )
        with patch.object(connector, "_unlink_shm") as mock_unlink:
            connector.free_physical_cache("x")
            mock_unlink.assert_not_called()


class TestECSharedMemoryConnectorIntegration:
    @patch(f"{_SHM_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    def test_save_and_load_roundtrip_real_shm(self, mock_tp_rank):
        from multiprocessing import shared_memory

        producer_cfg = _make_config(is_producer=True, is_consumer=False)
        consumer_cfg = _make_config(is_producer=False, is_consumer=True)
        producer = ECSharedMemoryConnector(
            vllm_config=producer_cfg, role=ECConnectorRole.WORKER
        )
        consumer = ECSharedMemoryConnector(
            vllm_config=consumer_cfg, role=ECConnectorRole.WORKER
        )

        mm_hash = "integration_hash"
        shm_name = _shm_name(mm_hash)
        tensor = torch.randn(128, 768, dtype=torch.float16)
        encoder_cache = {mm_hash: tensor}

        try:
            producer.save_caches(encoder_cache, mm_hash)

            metadata = ECSharedMemoryConnectorMetadata()
            metadata.add_mm_hash(mm_hash)
            consumer.bind_connector_metadata(metadata)

            loaded: dict[str, torch.Tensor] = {}
            with patch("vllm.platforms.current_platform") as mock_platform:
                mock_platform.device_type = "cpu"
                consumer.start_load_caches(loaded)

            assert mm_hash in loaded
            assert torch.equal(tensor.cpu(), loaded[mm_hash].cpu())
        finally:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
