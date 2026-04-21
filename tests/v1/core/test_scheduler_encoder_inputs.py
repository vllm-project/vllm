# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for scheduler's encoder input scheduling logic.

This test file focuses on testing the scheduler's _try_schedule_encoder_inputs
method and related encoder cache management functionality.
"""

from unittest.mock import Mock

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


class MockECConnector:
    """Mock EC connector for testing."""

    def __init__(self):
        self.cache_items = set()

    def has_cache_item(self, identifier: str, request=None) -> bool:
        return identifier in self.cache_items

    def add_cache_item(self, identifier: str):
        self.cache_items.add(identifier)


def create_mock_request_with_mm(
    request_id: str,
    mm_hash: str,
    num_encoder_tokens: int = 100,
) -> Request:
    """Create a mock request with multimodal features."""
    request = Mock(spec=Request)
    request.request_id = request_id
    request.has_encoder_inputs = True
    request.num_computed_tokens = 0

    # Create multimodal feature
    mm_feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=0, length=num_encoder_tokens),
    )
    request.mm_features = [mm_feature]

    # Mock get_num_encoder_embeds
    request.get_num_encoder_embeds = Mock(return_value=num_encoder_tokens)

    return request


class TestTryScheduleEncoderInputsReturnValue:
    """Test _try_schedule_encoder_inputs return value structure."""

    def test_return_value_structure(self):
        """Test that _try_schedule_encoder_inputs returns correct structure."""
        from vllm.v1.core.sched.scheduler import Scheduler

        # Create a minimal scheduler mock
        scheduler = Mock(spec=Scheduler)
        scheduler.is_encoder_decoder = False
        scheduler.scheduler_config = Mock()
        scheduler.scheduler_config.disable_chunked_mm_input = False
        scheduler.ec_connector = None
        scheduler.encoder_cache_manager = Mock(spec=EncoderCacheManager)
        scheduler.encoder_cache_manager.check_and_update_cache = Mock(
            return_value=False
        )
        scheduler.encoder_cache_manager.can_allocate = Mock(return_value=True)

        # Create a request with no encoder inputs
        request = Mock(spec=Request)
        request.has_encoder_inputs = False

        # Call the method
        result = Scheduler._try_schedule_encoder_inputs(
            scheduler,
            request=request,
            num_computed_tokens=0,
            num_new_tokens=10,
            encoder_compute_budget=1000,
            shift_computed_tokens=0,
        )

        # Verify return value structure
        assert isinstance(result, tuple)
        assert len(result) == 4, f"Expected 4 elements, got {len(result)}"

        (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        ) = result

        # Verify types
        assert isinstance(encoder_inputs_to_schedule, list)
        assert isinstance(num_new_tokens, int)
        assert isinstance(encoder_compute_budget, int)
        assert isinstance(external_load_encoder_input, list)


class TestSchedulerEncoderInputLogic:
    """Test scheduler's encoder input scheduling logic."""

    def test_hbm_cache_hit_skips_scheduling(self):
        """
        Test that when EncoderCacheManager has cache, scheduler skips scheduling.
        The connector handles external storage sync separately.
        """
        from vllm.v1.core.sched.scheduler import Scheduler

        # Create scheduler mock
        scheduler = Mock(spec=Scheduler)
        scheduler.is_encoder_decoder = False
        scheduler.scheduler_config = Mock()
        scheduler.scheduler_config.disable_chunked_mm_input = False
        scheduler.ec_connector = MockECConnector()

        # Create encoder cache manager that HAS the cache
        scheduler.encoder_cache_manager = Mock(spec=EncoderCacheManager)
        scheduler.encoder_cache_manager.check_and_update_cache = Mock(return_value=True)

        # Create request
        mm_hash = "test_hash_hbm_has"
        request = create_mock_request_with_mm("req_1", mm_hash)

        # Call the method
        result = Scheduler._try_schedule_encoder_inputs(
            scheduler,
            request=request,
            num_computed_tokens=0,
            num_new_tokens=150,
            encoder_compute_budget=1000,
            shift_computed_tokens=0,
        )

        (
            encoder_inputs_to_schedule,
            _,
            _,
            external_load_encoder_input,
        ) = result

        # Verify all lists are empty (skipped because EncoderCacheManager has it)
        assert len(encoder_inputs_to_schedule) == 0
        assert len(external_load_encoder_input) == 0

    def test_external_cache_hit_schedules_load(self):
        """
        Test that when external storage has cache but EncoderCacheManager doesn't,
        scheduler marks it for loading.
        """
        from vllm.v1.core.sched.scheduler import Scheduler

        # Create scheduler mock
        scheduler = Mock(spec=Scheduler)
        scheduler.is_encoder_decoder = False
        scheduler.scheduler_config = Mock()
        scheduler.scheduler_config.disable_chunked_mm_input = False

        # Create EC connector that HAS the cache
        mm_hash = "test_hash_external_only"
        scheduler.ec_connector = MockECConnector()
        scheduler.ec_connector.add_cache_item(mm_hash)

        # Create encoder cache manager that DOESN'T have the cache
        scheduler.encoder_cache_manager = Mock(spec=EncoderCacheManager)
        scheduler.encoder_cache_manager.check_and_update_cache = Mock(
            return_value=False
        )
        scheduler.encoder_cache_manager.can_allocate = Mock(return_value=True)

        # Create request
        request = create_mock_request_with_mm("req_2", mm_hash, num_encoder_tokens=100)

        # Call the method
        result = Scheduler._try_schedule_encoder_inputs(
            scheduler,
            request=request,
            num_computed_tokens=0,
            num_new_tokens=150,
            encoder_compute_budget=1000,
            shift_computed_tokens=0,
        )

        (
            encoder_inputs_to_schedule,
            _,
            _,
            external_load_encoder_input,
        ) = result

        # Verify external_load_encoder_input contains the input
        assert len(external_load_encoder_input) == 1
        assert external_load_encoder_input[0] == 0

        # Verify it's NOT in compute list
        assert len(encoder_inputs_to_schedule) == 0

    def test_no_cache_schedules_compute(self):
        """
        Test that when neither EncoderCacheManager nor external has cache,
        scheduler marks it for computation.
        """
        from vllm.v1.core.sched.scheduler import Scheduler

        # Create scheduler mock
        scheduler = Mock(spec=Scheduler)
        scheduler.is_encoder_decoder = False
        scheduler.scheduler_config = Mock()
        scheduler.scheduler_config.disable_chunked_mm_input = False

        # Create EC connector that DOESN'T have the cache
        scheduler.ec_connector = MockECConnector()

        # Create encoder cache manager that DOESN'T have the cache
        scheduler.encoder_cache_manager = Mock(spec=EncoderCacheManager)
        scheduler.encoder_cache_manager.check_and_update_cache = Mock(
            return_value=False
        )
        scheduler.encoder_cache_manager.can_allocate = Mock(return_value=True)

        # Create request
        mm_hash = "test_hash_neither_has"
        request = create_mock_request_with_mm("req_3", mm_hash, num_encoder_tokens=100)

        # Call the method
        result = Scheduler._try_schedule_encoder_inputs(
            scheduler,
            request=request,
            num_computed_tokens=0,
            num_new_tokens=150,
            encoder_compute_budget=1000,
            shift_computed_tokens=0,
        )

        (
            encoder_inputs_to_schedule,
            _,
            _,
            external_load_encoder_input,
        ) = result

        # Verify encoder_inputs_to_schedule contains the input
        assert len(encoder_inputs_to_schedule) == 1
        assert encoder_inputs_to_schedule[0] == 0

        # Verify it's NOT in load list
        assert len(external_load_encoder_input) == 0


class TestEncoderCacheManagerHasCache:
    """Test the has_cache() method in EncoderCacheManager."""

    def test_has_cache_returns_true_when_cached(self):
        """Test has_cache returns True when mm_hash is in cached dict."""
        manager = EncoderCacheManager(cache_size=1000)
        mm_hash = "test_hash_1"

        # Add to cache
        manager.cached[mm_hash] = {"req_1"}

        # Test
        assert manager.has_cache(mm_hash) is True

    def test_has_cache_returns_false_when_not_cached(self):
        """Test has_cache returns False when mm_hash is not in cached dict."""
        manager = EncoderCacheManager(cache_size=1000)
        mm_hash = "test_hash_2"

        # Test without adding
        assert manager.has_cache(mm_hash) is False

    def test_has_cache_returns_true_for_freeable_entries(self):
        """Test has_cache returns True even for freeable (unreferenced) entries."""
        manager = EncoderCacheManager(cache_size=1000)
        mm_hash = "test_hash_3"

        # Add to cache with empty reference set (freeable)
        manager.cached[mm_hash] = set()
        manager.freeable[mm_hash] = 100

        # Test - should still return True
        assert manager.has_cache(mm_hash) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
