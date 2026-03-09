# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the routed experts capturer.

Tests cover:
- _RoutedExpertsDeviceCache: Per-device GPU cache for capturing routing info
- _RoutedExpertsHostCache: Lazy per-request CPU cache
- _RoutedExpertsCapturerReal: Full capturer with DtoH sync
- Integration with FusedMoE.select_experts capture path
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    _RoutedExpertsDeviceCache,
    _RoutedExpertsHostCache,
    _RoutedExpertsCapturerReal,
    _RoutedExpertsCapturerNoop,
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)


class TestRoutedExpertsDeviceCache:
    """Tests for _RoutedExpertsDeviceCache."""

    def test_init_creates_buffer_with_correct_shape(self):
        """Test that initialization creates buffer with correct shape."""
        num_batched_tokens = 128
        num_hidden_layers = 24
        num_experts_per_tok = 2
        device = "cpu"  # Use CPU for testing without GPU

        cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=num_batched_tokens,
            num_hidden_layers=num_hidden_layers,
            num_experts_per_tok=num_experts_per_tok,
            num_fused_shared_experts=0,
            device=device,
        )

        assert cache.buffer.shape == (num_batched_tokens, num_hidden_layers, num_experts_per_tok)
        assert cache.buffer.dtype == torch.int16
        assert cache.buffer.device.type == device

    def test_capture_fwd_routed_experts(self):
        """Test capturing expert IDs during forward pass."""
        cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=64,
            num_hidden_layers=4,
            num_experts_per_tok=2,
            num_fused_shared_experts=0,
            device="cpu",
        )

        # Simulate topk_ids for a batch of 32 tokens
        batch_size = 32
        topk_ids = torch.randint(0, 8, (batch_size, 2), dtype=torch.int32)

        # Capture for layer 0
        cache.capture_fwd_routed_experts(layer_id=0, topk_ids=topk_ids)

        # Verify the captured data
        assert torch.equal(
            cache.buffer[:batch_size, 0, :],
            topk_ids.to(torch.int16)
        )

    def test_capture_multiple_layers(self):
        """Test capturing expert IDs for multiple layers."""
        cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=64,
            num_hidden_layers=4,
            num_experts_per_tok=2,
            num_fused_shared_experts=0,
            device="cpu",
        )

        batch_size = 16
        num_layers = 4

        # Capture different expert IDs for each layer
        layer_topk_ids = []
        for layer_id in range(num_layers):
            topk_ids = torch.randint(0, 8, (batch_size, 2), dtype=torch.int32)
            layer_topk_ids.append(topk_ids)
            cache.capture_fwd_routed_experts(layer_id=layer_id, topk_ids=topk_ids)

        # Verify each layer has correct data
        for layer_id in range(num_layers):
            assert torch.equal(
                cache.buffer[:batch_size, layer_id, :],
                layer_topk_ids[layer_id].to(torch.int16)
            )

    def test_get_buffer_size_bytes(self):
        """Test buffer size calculation."""
        cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=128,
            num_hidden_layers=24,
            num_experts_per_tok=2,
            num_fused_shared_experts=0,
            device="cpu",
        )

        expected_size = 128 * 24 * 2 * 2  # elements * sizeof(int16)
        assert cache.get_buffer_size_bytes() == expected_size


class TestRoutedExpertsHostCache:
    """Tests for _RoutedExpertsHostCache with lazy allocation."""

    def test_init_no_preallocated_buffer(self):
        """Test that initialization doesn't preallocate massive buffer."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=24,
            num_experts_per_tok=2,
            max_running_requests=256,
            max_model_len=32768,
            use_shared_memory=False,
        )

        # Should start with zero allocated bytes (lazy allocation)
        assert cache.get_buffer_size_bytes() == 0
        assert len(cache._req_buffers) == 0

    def test_get_or_grow_buffer_creates_new_buffer(self):
        """Test lazy buffer creation for new request."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        req_id = "test_req_1"
        max_pos = 99  # 100 tokens

        buf = cache.get_or_grow_buffer(req_id, max_pos)

        # Check buffer was created with correct shape
        assert buf.shape == (100, 4, 2)  # (max_pos + 1, layers, experts_per_tok)
        assert buf.dtype == torch.int16
        assert req_id in cache._req_buffers

        # Check memory tracking
        expected_bytes = 100 * 4 * 2 * 2  # elements * sizeof(int16)
        assert cache.get_buffer_size_bytes() == expected_bytes

    def test_get_or_grow_buffer_grows_existing(self):
        """Test buffer growth when more tokens are added."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        req_id = "test_req_1"

        # Initial allocation for 50 tokens
        buf1 = cache.get_or_grow_buffer(req_id, 49)
        assert buf1.shape[0] == 50
        initial_bytes = cache.get_buffer_size_bytes()

        # Write some data
        buf1[0, 0, 0] = 42

        # Grow to 150 tokens (exceeds 2x of 50=100, so will be exactly 150)
        buf2 = cache.get_or_grow_buffer(req_id, 149)
        assert buf2.shape[0] == 150

        # Old data should be preserved
        assert buf2[0, 0, 0] == 42

        # Memory should have increased
        assert cache.get_buffer_size_bytes() > initial_bytes

    def test_get_or_grow_buffer_no_shrink(self):
        """Test that buffer doesn't shrink when smaller max_pos is requested."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        req_id = "test_req_1"

        # Allocate for 100 tokens
        buf1 = cache.get_or_grow_buffer(req_id, 99)
        assert buf1.shape[0] == 100

        # Request smaller allocation - should return same buffer
        buf2 = cache.get_or_grow_buffer(req_id, 49)
        assert buf2.shape[0] == 100  # Still 100, not shrunk
        assert buf1 is buf2

    def test_get_buffer(self):
        """Test getting buffer for request."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        req_id = "test_req_1"

        # No buffer before allocation
        assert cache.get_buffer(req_id) is None

        # Allocate buffer
        buf1 = cache.get_or_grow_buffer(req_id, 99)
        buf2 = cache.get_buffer(req_id)
        assert buf1 is buf2

    def test_free_request(self):
        """Test freeing request buffer."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        req_id = "test_req_1"

        # Allocate buffer
        cache.get_or_grow_buffer(req_id, 99)
        assert cache.get_buffer_size_bytes() > 0
        buf = cache.get_buffer(req_id)
        assert buf is not None
        assert buf.shape[0] == 100

        # Free the request
        cache.free_request(req_id)

        # Should be cleaned up
        assert cache.get_buffer(req_id) is None
        assert cache.get_buffer_size_bytes() == 0

    def test_multiple_requests(self):
        """Test handling multiple concurrent requests."""
        cache = _RoutedExpertsHostCache(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            max_running_requests=16,
            max_model_len=1024,
            use_shared_memory=False,
        )

        # Create multiple requests with different sizes
        requests = [
            ("req_1", 100),
            ("req_2", 200),
            ("req_3", 50),
        ]

        for req_id, seqlen in requests:
            cache.get_or_grow_buffer(req_id, seqlen - 1)

        # Verify each has correct buffer size
        for req_id, seqlen in requests:
            buf = cache.get_buffer(req_id)
            assert buf is not None
            assert buf.shape[0] == seqlen

        # Free one request
        cache.free_request("req_2")
        assert cache.get_buffer("req_2") is None

        # Others should still be present
        assert cache.get_buffer("req_1") is not None
        assert cache.get_buffer("req_3") is not None


class TestRoutedExpertsCapturerReal:
    """Tests for _RoutedExpertsCapturerReal."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model config for testing."""
        class MockHfConfig:
            layers_block_type = ["moe", "moe", "moe", "moe"]  # 4 MoE layers
            num_experts_per_tok = 2

        class MockModelConfig:
            hf_text_config = MockHfConfig()

        return MockModelConfig()

    def test_init_creates_device_and_host_cache(self, mock_model_config):
        """Test that initialization creates both caches."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        assert capturer.device_cache is not None
        assert capturer.host_cache is not None
        assert capturer.num_hidden_layers == 4
        assert capturer.num_experts_per_tok == 2

    def test_init_skip_host_cache(self, mock_model_config):
        """Test skipping host cache for non-rank-0."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
            skip_host_cache=True,
        )

        assert capturer.device_cache is not None
        assert capturer.host_cache is None

    def test_capture_delegates_to_device_cache(self, mock_model_config):
        """Test that capture() delegates to device cache."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        topk_ids = torch.randint(0, 8, (32, 2), dtype=torch.int32)
        capturer.capture(layer_id=0, topk_ids=topk_ids)

        # Verify device cache has the data
        assert torch.equal(
            capturer.device_cache.buffer[:32, 0, :],
            topk_ids.to(torch.int16)
        )

    def test_sync_fwd_experts_buffer_DtoH(self, mock_model_config):
        """Test syncing device buffer to host cache."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        # Capture expert IDs for a batch
        batch_size = 16
        for layer_id in range(4):
            topk_ids = torch.randint(0, 8, (batch_size, 2), dtype=torch.int32)
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        # Positions for the tokens
        positions = torch.arange(batch_size)

        # Sync to host
        num_scheduled_tokens = {"req_1": batch_size}
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes=num_scheduled_tokens,
        )

        # Verify host cache has the data
        buf = capturer.host_cache.get_buffer("req_1")
        assert buf is not None
        assert buf.shape[0] == batch_size  # Buffer size matches positions
        assert buf.shape == (batch_size, 4, 2)

    def test_sync_with_multiple_requests(self, mock_model_config):
        """Test syncing with multiple requests in a batch."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        # Simulate batch with 2 requests
        req1_tokens = 10
        req2_tokens = 20
        total_tokens = req1_tokens + req2_tokens

        # Capture for all tokens
        for layer_id in range(4):
            topk_ids = torch.randint(0, 8, (total_tokens, 2), dtype=torch.int32)
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        # Positions: req1 gets positions 0-9, req2 gets positions 0-19
        positions = torch.cat([
            torch.arange(req1_tokens),
            torch.arange(req2_tokens),
        ])

        num_scheduled_tokens = {
            "req_1": req1_tokens,
            "req_2": req2_tokens,
        }

        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes=num_scheduled_tokens,
        )

        # Verify each request has correct buffer size
        buf1 = capturer.host_cache.get_buffer("req_1")
        buf2 = capturer.host_cache.get_buffer("req_2")
        assert buf1 is not None
        assert buf2 is not None
        assert buf1.shape[0] == req1_tokens
        assert buf2.shape[0] == req2_tokens

    def test_get_routed_experts(self, mock_model_config):
        """Test retrieving routed experts for a request."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        batch_size = 16

        # Capture and sync
        for layer_id in range(4):
            topk_ids = torch.randint(0, 8, (batch_size, 2), dtype=torch.int32)
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        positions = torch.arange(batch_size)
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes={"req_1": batch_size},
        )

        # Get routed experts
        result = capturer.get_routed_experts("req_1", seqlen=batch_size, free_slot=False)

        assert result is not None
        assert result.shape == (batch_size, 4, 2)

        # Without free_slot, should still be accessible
        assert capturer.host_cache.get_buffer("req_1") is not None

    def test_get_routed_experts_with_free(self, mock_model_config):
        """Test that get_routed_experts frees the buffer when requested."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        batch_size = 16

        # Capture and sync
        for layer_id in range(4):
            topk_ids = torch.randint(0, 8, (batch_size, 2), dtype=torch.int32)
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        positions = torch.arange(batch_size)
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes={"req_1": batch_size},
        )

        # Get routed experts with free
        result = capturer.get_routed_experts("req_1", seqlen=batch_size, free_slot=True)

        assert result is not None

        # Buffer should be freed
        assert capturer.host_cache.get_buffer("req_1") is None

    def test_get_routed_experts_nonexistent_request(self, mock_model_config):
        """Test getting routed experts for non-existent request."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=64,
            max_running_requests=16,
            num_fused_shared_experts=0,
            max_model_len=1024,
            device="cpu",
        )

        result = capturer.get_routed_experts("nonexistent", seqlen=10)
        assert result is None


class TestRoutedExpertsCapturerNoop:
    """Tests for _RoutedExpertsCapturerNoop."""

    def test_capture_does_nothing(self):
        """Test that noop capturer's capture does nothing."""
        capturer = _RoutedExpertsCapturerNoop()
        topk_ids = torch.randint(0, 8, (32, 2), dtype=torch.int32)

        # Should not raise
        capturer.capture(layer_id=0, topk_ids=topk_ids)

    def test_get_routed_experts_returns_none(self):
        """Test that noop capturer returns None."""
        capturer = _RoutedExpertsCapturerNoop()
        result = capturer.get_routed_experts("any_req", seqlen=10)
        assert result is None

    def test_sync_does_nothing(self):
        """Test that noop capturer's sync does nothing."""
        capturer = _RoutedExpertsCapturerNoop()

        # Should not raise
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=torch.arange(10),
            num_scheduled_tokes={"req_1": 10},
        )

    def test_get_host_cache_returns_none(self):
        """Test that noop capturer has no host cache."""
        capturer = _RoutedExpertsCapturerNoop()
        assert capturer.get_host_cache() is None


class TestRoutedExpertsCapturerFactory:
    """Tests for RoutedExpertsCapturer.create factory method."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model config for testing."""
        class MockHfConfig:
            layers_block_type = ["moe", "moe", "moe", "moe"]
            num_experts_per_tok = 2

        class MockModelConfig:
            hf_text_config = MockHfConfig()

        return MockModelConfig()

    def test_create_disabled_returns_noop(self, mock_model_config):
        """Test that disabled capturer returns noop."""
        capturer = RoutedExpertsCapturer.create(
            enable=False,
            model_config=mock_model_config,
            num_fused_shared_experts=0,
            num_batched_tokens=64,
            max_running_requests=16,
            max_model_len=1024,
            device="cpu",
        )

        assert isinstance(capturer, _RoutedExpertsCapturerNoop)

    def test_create_enabled_returns_real(self, mock_model_config):
        """Test that enabled capturer returns real implementation."""
        capturer = RoutedExpertsCapturer.create(
            enable=True,
            model_config=mock_model_config,
            num_fused_shared_experts=0,
            num_batched_tokens=64,
            max_running_requests=16,
            max_model_len=1024,
            device="cpu",
        )

        assert isinstance(capturer, _RoutedExpertsCapturerReal)


class TestGlobalCapturerManagement:
    """Tests for global capturer get/set functions."""

    def test_set_and_get_global_capturer(self):
        """Test setting and getting global capturer."""
        original = get_global_experts_capturer()

        try:
            new_capturer = _RoutedExpertsCapturerNoop()
            set_global_experts_capturer(new_capturer)

            assert get_global_experts_capturer() is new_capturer
        finally:
            # Restore original
            set_global_experts_capturer(original)


class TestEndToEndCapture:
    """End-to-end tests simulating real usage patterns."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model config for testing."""
        class MockHfConfig:
            layers_block_type = ["moe"] * 24  # 24 MoE layers
            num_experts_per_tok = 2

        class MockModelConfig:
            hf_text_config = MockHfConfig()

        return MockModelConfig()

    def test_full_request_lifecycle(self, mock_model_config):
        """Test capturing routing info for a full request lifecycle."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=512,
            max_running_requests=32,
            num_fused_shared_experts=0,
            max_model_len=2048,
            device="cpu",
        )

        req_id = "test_request"
        prompt_len = 100
        num_generated = 50
        total_tokens = prompt_len + num_generated

        # Simulate prefill: capture all prompt tokens at once
        expected_routing = {}
        for layer_id in range(24):
            topk_ids = torch.randint(0, 8, (prompt_len, 2), dtype=torch.int32)
            expected_routing[layer_id] = topk_ids.clone()
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        # Sync prefill
        positions = torch.arange(prompt_len)
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes={req_id: prompt_len},
        )

        # Simulate generation: capture one token at a time
        for gen_step in range(num_generated):
            for layer_id in range(24):
                topk_ids = torch.randint(0, 8, (1, 2), dtype=torch.int32)
                capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

            # Sync each generated token
            gen_position = torch.tensor([prompt_len + gen_step])
            capturer.sync_fwd_experts_buffer_DtoH(
                positions=gen_position,
                num_scheduled_tokes={req_id: 1},
            )

        # Verify final state
        buf = capturer.host_cache.get_buffer(req_id)
        assert buf is not None
        assert buf.shape[0] == total_tokens

        # Get final routing info
        result = capturer.get_routed_experts(req_id, seqlen=total_tokens, free_slot=True)
        assert result.shape == (total_tokens, 24, 2)

        # Buffer should be freed
        assert capturer.host_cache.get_buffer(req_id) is None

    def test_concurrent_requests(self, mock_model_config):
        """Test handling multiple concurrent requests."""
        capturer = _RoutedExpertsCapturerReal(
            model_config=mock_model_config,
            num_batched_tokens=512,
            max_running_requests=32,
            num_fused_shared_experts=0,
            max_model_len=2048,
            device="cpu",
        )

        # Simulate 3 requests with different lengths
        requests = {
            "req_a": {"prompt_len": 50, "gen_len": 20},
            "req_b": {"prompt_len": 100, "gen_len": 10},
            "req_c": {"prompt_len": 30, "gen_len": 40},
        }

        # Process prefills (batched)
        total_prefill = sum(r["prompt_len"] for r in requests.values())
        for layer_id in range(24):
            topk_ids = torch.randint(0, 8, (total_prefill, 2), dtype=torch.int32)
            capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

        # Build positions and schedule for prefill
        positions = []
        num_scheduled = {}
        for req_id, info in requests.items():
            positions.append(torch.arange(info["prompt_len"]))
            num_scheduled[req_id] = info["prompt_len"]

        positions = torch.cat(positions)
        capturer.sync_fwd_experts_buffer_DtoH(
            positions=positions,
            num_scheduled_tokes=num_scheduled,
        )

        # Verify initial state
        for req_id, info in requests.items():
            buf = capturer.host_cache.get_buffer(req_id)
            assert buf is not None
            assert buf.shape[0] == info["prompt_len"]

        # Simulate some generation steps
        for step in range(5):
            num_active = len(requests)
            for layer_id in range(24):
                topk_ids = torch.randint(0, 8, (num_active, 2), dtype=torch.int32)
                capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

            positions = []
            num_scheduled = {}
            for req_id, info in requests.items():
                gen_pos = info["prompt_len"] + step
                positions.append(torch.tensor([gen_pos]))
                num_scheduled[req_id] = 1

            positions = torch.cat(positions)
            capturer.sync_fwd_experts_buffer_DtoH(
                positions=positions,
                num_scheduled_tokes=num_scheduled,
            )

        # Verify all requests have correct buffer size
        for req_id, info in requests.items():
            expected_seqlen = info["prompt_len"] + 5  # prefill + 5 gen steps
            buf = capturer.host_cache.get_buffer(req_id)
            assert buf is not None
            assert buf.shape[0] == expected_seqlen

        # Finish one request and verify cleanup
        result = capturer.get_routed_experts("req_a", seqlen=55, free_slot=True)
        assert result is not None
        assert capturer.host_cache.get_buffer("req_a") is None

        # Other requests should still be available
        assert capturer.host_cache.get_buffer("req_b") is not None
        assert capturer.host_cache.get_buffer("req_c") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

