# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for scheduler's _update_requests_with_invalid_blocks function.
Tests for hybrid attention KV load failure recovery with recompute.

Run with: python -m pytest tests/v1/core/test_scheduler_kv_load_failure.py -v
"""

import pytest
from unittest.mock import MagicMock
from vllm.v1.core.sched.scheduler import Scheduler

def _make_mock_request(request_id: str, num_computed_tokens: int) -> MagicMock:
    """Create a mock Request object.

    Note: block_ids is obtained via kv_cache_manager.get_block_ids(),
    so it's not needed here.
    """
    request = MagicMock()
    request.request_id = request_id
    request.num_computed_tokens = num_computed_tokens
    return request


def _make_mock_kv_cache_config(block_size: int = 16) -> MagicMock:
    """Create a mock KVCacheConfig with single full-attention group."""
    config = MagicMock()
    group = MagicMock()
    group.kv_cache_spec.block_size = block_size
    config.kv_cache_groups = [group]
    return config


def _make_mock_kv_cache_manager(
        block_ids_map: dict[str, tuple[list[int], ...]]
) -> MagicMock:
    """Create a mock KVCacheManager that returns block IDs for requests."""
    manager = MagicMock()
    manager.get_block_ids.side_effect = lambda req_id: block_ids_map.get(req_id, ([],))
    manager.evict_blocks = MagicMock()
    return manager


class TestUpdateRequestsWithInvalidBlocks:
    """Tests for Scheduler._update_requests_with_invalid_blocks."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a minimal mock Scheduler with required attributes."""
        scheduler = MagicMock()
        scheduler.kv_cache_manager = _make_mock_kv_cache_manager({})
        scheduler.kv_cache_config = _make_mock_kv_cache_config(block_size=16)
        return scheduler

    def test_single_block_invalid_rewinds_correctly(self, mock_scheduler):
        """A single invalid block should rewind to the block boundary."""
        # Request has 48 tokens (3 blocks of 16 tokens each), block 1 is invalid
        mock_scheduler.kv_cache_manager = _make_mock_kv_cache_manager({
            "req-0": ([0, 1, 2],),
        })
        requests = [_make_mock_request("req-0", 48)]
        invalid_block_ids = {1}  # Block 1 (tokens 16-31) is invalid
        num_scheduled_tokens = {"req-0": 0}

        result = Scheduler._update_requests_with_invalid_blocks(
            mock_scheduler,
            requests,
            invalid_block_ids,
            num_scheduled_tokens,
            evict_blocks=True,
        )

        affected, tokens, blocks = result
        assert "req-0" in affected
        assert requests[0].num_computed_tokens == 16
        assert tokens == 32

    def test_multiple_requests_share_invalid_block(self, mock_scheduler):
        """When multiple requests share an invalid block, only first recomputes."""
        # Two requests share block 1 but have different block tables
        mock_scheduler.kv_cache_manager = _make_mock_kv_cache_manager({
            "req-0": ([0, 1, 2],),  # req-0 uses blocks 0,1,2
            "req-1": ([10, 1, 20],),  # req-1 uses blocks 10,1,20 (only block 1 overlaps)
        })
        requests = [
            _make_mock_request("req-0", 48),
            _make_mock_request("req-1", 48),
        ]
        invalid_block_ids = {1}  # Block 1 is invalid
        num_scheduled_tokens = {"req-0": 0, "req-1": 0}
        result = Scheduler._update_requests_with_invalid_blocks(
            mock_scheduler,
            requests,
            invalid_block_ids,
            num_scheduled_tokens,
            evict_blocks=True,
        )

        affected, tokens, blocks = result
        assert "req-0" in affected
        assert "req-1" in affected
        assert requests[0].num_computed_tokens == 16
        assert requests[1].num_computed_tokens == 48

    def test_multiple_groups_different_block_sizes(self, mock_scheduler):
        """Test handling of multiple KV cache groups with different block sizes."""
        # Create config with 2 groups: block_size=16 and block_size=8
        config = MagicMock()
        group0 = MagicMock()
        group0.kv_cache_spec.block_size = 16
        group1 = MagicMock()
        group1.kv_cache_spec.block_size = 8
        config.kv_cache_groups = [group0, group1]
        mock_scheduler.kv_cache_config = config

        # Request has 32 computed tokens:
        # - group0 (block_size=16): 32/16 = 2 blocks → [0, 1]
        # - group1 (block_size=8): 32/8 = 4 blocks → [10, 11, 12, 13]
        # Using different block IDs for each group to avoid ambiguity
        mock_scheduler.kv_cache_manager = _make_mock_kv_cache_manager({
            "req-0": ([0, 1], [10, 11, 12, 13]),
        })
        requests = [_make_mock_request("req-0", 32)]
        invalid_block_ids = {1,13}
        num_scheduled_tokens = {"req-0": 0}
        result = Scheduler._update_requests_with_invalid_blocks(
            mock_scheduler,
            requests,
            invalid_block_ids,
            num_scheduled_tokens,
            evict_blocks=True,
        )

        affected, tokens, blocks = result
        assert "req-0" in affected

        # Block 1 and 13 are invalid
        # Should rewind to 16 tokens
        assert requests[0].num_computed_tokens == 16
        assert tokens == 16
