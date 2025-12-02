# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ECSharedStorageConnector.
"""

import os
from unittest.mock import Mock, patch

import pytest
import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.shared_storage_connector import (
    ECSharedStorageConnector,
    ECSharedStorageConnectorMetadata,
    MMMeta,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import SchedulerOutput


# ------------------ Mock Classes ------------------ #
class MockRequest:
    def __init__(self, request_id, mm_hashes: list[str], token_counts: list[int]):
        assert len(mm_hashes) == len(token_counts)
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=self._token_counts[i]),
            )
            self.mm_features.append(feature)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self._token_counts)
        return self._token_counts[input_id]


@pytest.fixture
def temp_storage(tmp_path):
    """Fixture providing temporary storage path."""
    return str(tmp_path)


@pytest.fixture
def mock_vllm_config_producer(temp_storage):
    """Fixture providing mock VllmConfig for producer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=temp_storage)
    config.ec_transfer_config.is_ec_producer = True
    return config


@pytest.fixture
def mock_vllm_config_consumer(temp_storage):
    """Fixture providing mock VllmConfig for consumer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=temp_storage)
    config.ec_transfer_config.is_ec_producer = False
    return config


@pytest.fixture
def mock_request_with_3_mm():
    """Fixture providing mock Request with 3 multimodal items."""
    request_id = "test_req_123"
    mm_hashes = ["img_hash_1", "img_hash_2", "img_hash_3"]
    token_counts = [100, 150, 200]

    request = MockRequest(request_id, mm_hashes, token_counts)
    return request


# ------------------ Unit Tests ------------------ #
class TestECSharedStorageConnectorBasics:
    """Test basic EC connector functionality."""

    def test_initialization_producer(self, mock_vllm_config_producer, temp_storage):
        """Test connector initializes correctly as producer."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        assert connector.role == ECConnectorRole.SCHEDULER
        assert connector.is_producer
        assert connector._storage_path == temp_storage
        assert connector._mm_datas_need_loads == {}

    def test_initialization_consumer(self, mock_vllm_config_consumer, temp_storage):
        """Test connector initializes correctly as consumer."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        assert connector.role == ECConnectorRole.WORKER
        assert not connector.is_producer
        assert connector._storage_path == temp_storage

    def test_role_assignment(self, mock_vllm_config_producer):
        """Test role is correctly assigned."""
        scheduler_connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )
        worker_connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        assert scheduler_connector.role == ECConnectorRole.SCHEDULER
        assert worker_connector.role == ECConnectorRole.WORKER


class TestCacheExistence:
    """Test cache existence checking using has_caches() API."""

    def test_has_caches_all_exist_3_items(
        self,
        mock_vllm_config_producer,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
    ):
        """Test has_caches returns True when all 3 caches exist."""
        # Test for producer first
        producer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Create cache files using save_caches (proper way)
        encoder_cache: dict[str, torch.Tensor] = {}

        for mm_feature in mock_request_with_3_mm.mm_features:
            mm_hash = mm_feature.identifier
            encoder_cache[mm_hash] = torch.randn(10, 768)
            producer.save_caches(encoder_cache, mm_hash)

        # Test using has_caches API
        producer_result = producer.has_caches(mock_request_with_3_mm)

        # Assert
        assert len(producer_result) == 3
        assert all(producer_result), f"Expected all True, got {producer_result}"

        # Also test consumer can check if cache exists
        consumer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Test using has_caches API
        consumer_result = consumer.has_caches(mock_request_with_3_mm)

        # Assert
        assert len(consumer_result) == 3
        assert all(consumer_result), f"Expected all True, got {consumer_result}"

    def test_has_caches_none_exist(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        """Test has_caches returns False when no caches exist."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Test without creating any files
        result = connector.has_caches(mock_request_with_3_mm)

        # Assert
        assert len(result) == 3
        assert not any(result), f"Expected all False, got {result}"

    def test_has_caches_partial_exist(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        """Test has_caches with some caches existing (1 of 3)."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Create only the second cache file
        mm_hash_second = mock_request_with_3_mm.mm_features[1].identifier
        encoder_cache = {mm_hash_second: torch.randn(10, 768)}
        connector.save_caches(encoder_cache, mm_hash_second)

        # Test
        result = connector.has_caches(mock_request_with_3_mm)

        # Assert
        assert len(result) == 3
        assert not result[0]  # First doesn't exist
        assert result[1]  # Second exists
        assert not result[2]  # Third doesn't exist


class TestStateManagement:
    """Test connector state management."""

    def test_update_state_after_alloc_3_items(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        """Test state update after allocation for 3 MM items."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Initial state should be empty
        assert len(connector._mm_datas_need_loads) == 0

        # Update state for all 3 items
        for i in range(3):
            connector.update_state_after_alloc(mock_request_with_3_mm, index=i)

        # Check state updated for all 3
        assert len(connector._mm_datas_need_loads) == 3
        assert "img_hash_1" in connector._mm_datas_need_loads
        assert "img_hash_2" in connector._mm_datas_need_loads
        assert "img_hash_3" in connector._mm_datas_need_loads
        assert connector._mm_datas_need_loads["img_hash_1"] == 100
        assert connector._mm_datas_need_loads["img_hash_2"] == 150
        assert connector._mm_datas_need_loads["img_hash_3"] == 200

    def test_build_connector_meta_3_items(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        """Test metadata building for 3 MM items."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Setup state for all 3 items
        for i in range(3):
            connector.update_state_after_alloc(mock_request_with_3_mm, index=i)

        # Build metadata
        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        # Assert
        assert isinstance(metadata, ECSharedStorageConnectorMetadata)
        assert len(metadata.mm_datas) == 3
        assert metadata.mm_datas[0].mm_hash == "img_hash_1"
        assert metadata.mm_datas[0].num_token == 100
        assert metadata.mm_datas[1].mm_hash == "img_hash_2"
        assert metadata.mm_datas[1].num_token == 150
        assert metadata.mm_datas[2].mm_hash == "img_hash_3"
        assert metadata.mm_datas[2].num_token == 200

        # State should be cleared after building
        assert len(connector._mm_datas_need_loads) == 0

    def test_build_connector_meta_empty(self, mock_vllm_config_producer):
        """Test metadata building with empty state."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        assert isinstance(metadata, ECSharedStorageConnectorMetadata)
        assert len(metadata.mm_datas) == 0

    def test_state_cleared_after_metadata_build(
        self, mock_vllm_config_producer, mock_request_with_3_mm
    ):
        """Test that state is properly cleared after building metadata."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Add state
        for i in range(3):
            connector.update_state_after_alloc(mock_request_with_3_mm, index=i)
        assert len(connector._mm_datas_need_loads) == 3

        # Build metadata (should clear state)
        scheduler_output = Mock(spec=SchedulerOutput)
        connector.build_connector_meta(scheduler_output)

        # State should be empty
        assert len(connector._mm_datas_need_loads) == 0

        # Build again should return empty metadata
        metadata2 = connector.build_connector_meta(scheduler_output)
        assert len(metadata2.mm_datas) == 0


class TestCacheSaving:
    """Test encoder cache saving (producer only)."""

    def test_save_caches_producer_3_items(
        self, mock_vllm_config_producer, mock_request_with_3_mm, temp_storage
    ):
        """Test cache saving as producer for 3 different MM items."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        # Create and save 3 different caches
        mm_hashes = [f.identifier for f in mock_request_with_3_mm.mm_features]
        encoder_cache: dict[str, torch.Tensor] = {}

        for mm_hash in mm_hashes:
            encoder_cache[mm_hash] = torch.randn(10, 768)
            connector.save_caches(encoder_cache, mm_hash)

        # Verify all files exist using has_caches
        result = connector.has_caches(mock_request_with_3_mm)
        assert all(result), f"Not all caches were saved: {result}"

        # Verify each file's content
        for mm_hash in mm_hashes:
            filename = connector._generate_filename_debug(mm_hash)
            loaded = safetensors.torch.load_file(filename)
            assert "ec_cache" in loaded
            assert torch.allclose(loaded["ec_cache"], encoder_cache[mm_hash].cpu())

    def test_save_caches_consumer_skips(self, mock_vllm_config_consumer):
        """Test cache saving is skipped for consumer."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "test_hash_consumer"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        # Save should not raise but also not create file
        connector.save_caches(encoder_cache, mm_hash)

        # Verify file doesn't exist using has_caches
        mock_request = MockRequest("req_consumer", [mm_hash], [10])
        result = connector.has_caches(mock_request)
        assert not result[0], "Consumer should not save caches"


class TestCacheLoading:
    """Test encoder cache loading (consumer)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_start_load_caches_consumer_3_items(
        self,
        mock_vllm_config_producer,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
        temp_storage,
    ):
        """Test consumer loads 3 caches from storage."""
        # First, create producer to save caches
        producer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        # Producer saves 3 caches
        mm_hashes = [f.identifier for f in mock_request_with_3_mm.mm_features]
        saved_caches = {}
        for mm_hash in mm_hashes:
            saved_caches[mm_hash] = torch.randn(10, 768)
            producer.save_caches(saved_caches, mm_hash)

        # Now consumer loads
        consumer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        # Setup metadata for all 3
        metadata = ECSharedStorageConnectorMetadata()
        for mm_hash in mm_hashes:
            metadata.add_mm_data(MMMeta.make_meta(mm_hash, 100))
        consumer.bind_connector_metadata(metadata)

        # Load
        encoder_cache: dict[str, torch.Tensor] = {}
        consumer.start_load_caches(encoder_cache=encoder_cache)

        # Verify all 3 loaded
        assert len(encoder_cache) == 3
        for mm_hash in mm_hashes:
            assert mm_hash in encoder_cache, f"{mm_hash} missing in encoder_cache"
            assert encoder_cache[mm_hash].is_cuda, (
                f"{mm_hash} cache is in {encoder_cache[mm_hash].device}"
            )
            assert torch.allclose(
                encoder_cache[mm_hash].cpu(), saved_caches[mm_hash]
            ), f"{mm_hash} cache saved and loaded tesnor are not the same"

    def test_start_load_caches_skip_existing(
        self, mock_vllm_config_producer, mock_vllm_config_consumer, temp_storage
    ):
        """Test cache loading skips already cached items."""
        # Setup: producer saves cache
        producer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "existing_hash"
        saved_cache = torch.randn(10, 768)
        producer.save_caches({mm_hash: saved_cache}, mm_hash)

        # Consumer setup
        consumer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = ECSharedStorageConnectorMetadata()
        metadata.add_mm_data(MMMeta.make_meta(mm_hash, 100))
        consumer.bind_connector_metadata(metadata)

        # Pre-populate encoder_cache with different value
        existing_cache = torch.randn(5, 512)
        encoder_cache = {mm_hash: existing_cache}

        # Load (should skip since already exists)
        with patch("safetensors.torch.load_file") as mock_load:
            consumer.start_load_caches(encoder_cache=encoder_cache)
            # Should not call load_file since cache exists
            mock_load.assert_not_called()

        # Verify original cache unchanged
        assert torch.equal(encoder_cache[mm_hash], existing_cache)

    def test_start_load_caches_empty_metadata(self, mock_vllm_config_consumer):
        """Test loading with empty metadata does nothing."""
        consumer = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        # Setup empty metadata
        metadata = ECSharedStorageConnectorMetadata()
        consumer.bind_connector_metadata(metadata)

        # Load (should not raise)
        encoder_cache: dict[str, torch.Tensor] = {}
        consumer.start_load_caches(encoder_cache=encoder_cache)

        # Cache should remain empty
        assert len(encoder_cache) == 0


class TestFilenameGeneration:
    """Test filename and path generation."""

    def test_generate_foldername(self, mock_vllm_config_producer, temp_storage):
        """Test folder name generation."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "test_folder_hash"
        folder = connector._generate_foldername_debug(mm_hash)

        assert folder == os.path.join(temp_storage, mm_hash)
        assert os.path.isdir(folder)  # Should be created

    def test_generate_filename(self, mock_vllm_config_producer, temp_storage):
        """Test filename generation."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "test_file_hash"
        filename = connector._generate_filename_debug(mm_hash)

        expected = os.path.join(temp_storage, mm_hash, "encoder_cache.safetensors")
        assert filename == expected
        assert os.path.isdir(os.path.dirname(filename))  # Folder created

    def test_generate_filename_consistency(self, mock_vllm_config_producer):
        """Test filename generation is consistent."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "consistency_hash"
        filename1 = connector._generate_filename_debug(mm_hash)
        filename2 = connector._generate_filename_debug(mm_hash)

        assert filename1 == filename2


class TestMetadataBindingLifecycle:
    """Test metadata binding and clearing lifecycle."""

    def test_bind_connector_metadata(self, mock_vllm_config_consumer):
        """Test binding connector metadata."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = ECSharedStorageConnectorMetadata()
        metadata.add_mm_data(MMMeta.make_meta("hash_1", 100))

        connector.bind_connector_metadata(metadata)

        assert connector._connector_metadata is metadata

    def test_clear_connector_metadata(self, mock_vllm_config_consumer):
        """Test clearing connector metadata."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = ECSharedStorageConnectorMetadata()
        connector.bind_connector_metadata(metadata)

        connector.clear_connector_metadata()

        assert connector._connector_metadata is None

    def test_get_connector_metadata(self, mock_vllm_config_consumer):
        """Test getting connector metadata."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = ECSharedStorageConnectorMetadata()
        connector.bind_connector_metadata(metadata)

        retrieved = connector._get_connector_metadata()

        assert retrieved is metadata

    def test_get_connector_metadata_not_set(self, mock_vllm_config_consumer):
        """Test getting metadata when not set raises."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        with pytest.raises(AssertionError):
            connector._get_connector_metadata()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_empty_cache(self, mock_vllm_config_producer):
        """Test saving empty tensor."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        mm_hash = "empty_hash"
        encoder_cache = {mm_hash: torch.empty(0)}

        # Should not raise
        connector.save_caches(encoder_cache, mm_hash)

    def test_load_nonexistent_cache(self, mock_vllm_config_consumer):
        """Test loading cache that doesn't exist raises error."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = ECSharedStorageConnectorMetadata()
        metadata.add_mm_data(MMMeta.make_meta("nonexistent_hash", 100))
        connector.bind_connector_metadata(metadata)

        encoder_cache: dict[str, torch.Tensor] = {}

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            connector.start_load_caches(encoder_cache=encoder_cache)

    def test_has_caches_empty_request(self, mock_vllm_config_producer):
        """Test has_caches with request that has no MM data."""
        connector = ECSharedStorageConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        mock_request = MockRequest("req_empty", [], [])

        result = connector.has_caches(mock_request)

        assert len(result) == 0
        assert result == []
