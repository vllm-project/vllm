# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import (
    MaxPendingTokensError,
    MaxTierPendingTokensError,
    QueueOverflowError,
    TooManyRequestsError,
)


class FakeModelConfig(ModelConfig):
    """Fake model config for testing."""

    def __init__(self):
        super().__init__(
            model="fake-model",
            tokenizer="fake-tokenizer",
            trust_remote_code=False,
            download_dir=None,
            use_np_cache=False,
        )


class SLOTestSetup:
    """Common setup for SLO integration tests."""

    def __init__(self, scheduler_config: SchedulerConfig):
        self.scheduler_config = scheduler_config
        self.vllm_config = VllmConfig(
            model_config=FakeModelConfig(),
            scheduler_config=scheduler_config,
        )
        self.sampling_params = SamplingParams(temperature=0.9, n=1)

    def create_async_llm(self, mock_output_processor_instance: Mock) -> AsyncLLM:
        """Create AsyncLLM instance with mocked dependencies."""
        with (
            patch("vllm.v1.engine.async_llm.EngineCoreClient") as mock_engine_client,
            patch("vllm.v1.engine.async_llm.OutputProcessor") as mock_output_proc,
        ):
            # Setup mocks
            mock_engine_core = AsyncMock()
            mock_engine_client.return_value = mock_engine_core
            mock_output_proc.return_value = mock_output_processor_instance

            # Create and return AsyncLLM instance
            return AsyncLLM(
                vllm_config=self.vllm_config,
                executor_class=Mock(),
                log_stats=False,
                start_engine_loop=False,
            )


@pytest.mark.asyncio
async def test_max_num_reqs_slo_validation():
    """Test end-to-end SLO validation for max_num_reqs."""

    # Create config with max_num_reqs limit
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024, is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 2

    # Setup test helpers
    setup = SLOTestSetup(scheduler_config)

    # Create mock output processor
    mock_output_processor_instance = Mock()
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 2
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 0

    # Create AsyncLLM instance
    async_llm = setup.create_async_llm(mock_output_processor_instance)

    # Test 1: Request with tier=1.0 should be rejected
    # (2/2 > 1.0 is false, but 2+1 > 2 is true)
    with pytest.raises(QueueOverflowError):
        await async_llm.add_request(
            request_id="test-request-1",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=1.0,
        )

    # Test 2: Request with tier=0.5 should be rejected (2/2 > 0.5 is true)
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 2

    with pytest.raises(TooManyRequestsError):
        await async_llm.add_request(
            request_id="test-request-2",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=0.5,
        )

    # Test 3: Request with tier=1.0 should be accepted when under limit
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 1

    # This should not raise an exception
    await async_llm.add_request(
        request_id="test-request-3",
        prompt="test prompt",
        params=setup.sampling_params,
        tier=1.0,
    )


@pytest.mark.asyncio
async def test_max_pending_context_tokens_slo_validation():
    """Test end-to-end SLO validation for max_pending_context_tokens."""

    # Create config with max_pending_context_tokens limit
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024, is_encoder_decoder=False
    )
    scheduler_config.max_pending_context_tokens = 1000

    # Setup test helpers
    setup = SLOTestSetup(scheduler_config)

    # Create mock output processor
    mock_output_processor_instance = Mock()
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 0
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 1000

    # Create AsyncLLM instance
    async_llm = setup.create_async_llm(mock_output_processor_instance)

    # Test 1: Request should be rejected when at token limit
    with pytest.raises(MaxPendingTokensError):
        await async_llm.add_request(
            request_id="test-request-1",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=1.0,
        )

    # Test 2: Request should be rejected when exceeding tier limit
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 800

    with pytest.raises(MaxTierPendingTokensError):
        await async_llm.add_request(
            request_id="test-request-2",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=0.7,  # 800/1000 > 0.7 should trigger rejection
        )

    # Test 3: Request should be accepted when under tier limit
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 600

    # This should not raise an exception
    await async_llm.add_request(
        request_id="test-request-3",
        prompt="test prompt",
        params=setup.sampling_params,
        tier=0.7,  # 600/1000 <= 0.7 should be accepted
    )


@pytest.mark.asyncio
async def test_slo_disabled_when_config_none():
    """Test that SLO validation is disabled when config values are None."""

    # Create config with SLO limits disabled (None)
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024, is_encoder_decoder=False
    )
    # max_num_reqs and max_pending_context_tokens are None by default

    # Setup test helpers
    setup = SLOTestSetup(scheduler_config)

    # Create mock output processor with high load (should be ignored)
    mock_output_processor_instance = Mock()
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 100
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 10000

    # Create AsyncLLM instance
    async_llm = setup.create_async_llm(mock_output_processor_instance)

    # Should not raise any SLO exceptions when limits are disabled
    await async_llm.add_request(
        request_id="test-request-1",
        prompt="test prompt",
        params=setup.sampling_params,
        tier=0.5,
    )


@pytest.mark.asyncio
async def test_tier_boundary_conditions():
    """Test tier boundary conditions."""

    # Create config with limits
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024, is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 10
    scheduler_config.max_pending_context_tokens = 1000

    # Setup test helpers
    setup = SLOTestSetup(scheduler_config)

    # Create mock output processor
    mock_output_processor_instance = Mock()
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 1
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 1

    # Create AsyncLLM instance
    async_llm = setup.create_async_llm(mock_output_processor_instance)

    # Test tier=0.0 (should always be rejected if any requests exist)
    with pytest.raises(TooManyRequestsError):
        await async_llm.add_request(
            request_id="test-request-1",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=0.0,  # Should be rejected immediately
        )

    # Test tier=1.0 (should only be rejected at hard limits)
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 10
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 1000

    with pytest.raises(QueueOverflowError):  # Hard limit, not tier-based
        await async_llm.add_request(
            request_id="test-request-2",
            prompt="test prompt",
            params=setup.sampling_params,
            tier=1.0,
        )


@pytest.mark.asyncio
async def test_default_tier_value():
    """Test that default tier value (1.0) is used when not specified."""

    # Create config with SLO limits
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024, is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 10

    # Setup test helpers
    setup = SLOTestSetup(scheduler_config)

    # Create mock output processor
    mock_output_processor_instance = Mock()
    mock_output_processor_instance.get_num_unfinished_requests.return_value = 9
    mock_output_processor_instance.get_num_pending_context_tokens.return_value = 0

    # Create AsyncLLM instance
    async_llm = setup.create_async_llm(mock_output_processor_instance)

    # Test that default tier=1.0 is used when not specified
    # This should be accepted since 9/10 <= 1.0 and 9+1 <= 10
    await async_llm.add_request(
        request_id="test-request-default-tier",
        prompt="test prompt",
        params=setup.sampling_params,
        # tier parameter not specified - should default to 1.0
    )
