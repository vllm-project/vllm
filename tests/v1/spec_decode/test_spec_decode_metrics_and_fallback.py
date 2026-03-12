# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculative Decoding: Acceptance Rate Metrics and Fallback Mechanisms Tests.

This test suite covers:
- SpecDecodingStats: Core metrics data structure
- SpecDecodingLogging: Logging aggregation and formatting
- SpecDecodingProm: Prometheus metrics integration
- Fallback mechanisms when spec decode is not configured
- Configuration validation and error handling
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingProm,
    SpecDecodingStats,
)


class TestSpecDecodingStats:
    """Test suite for SpecDecodingStats data structure."""

    def test_initialization_via_new(self):
        """Test SpecDecodingStats.new() creates properly initialized instance."""
        num_spec_tokens = 5
        stats = SpecDecodingStats.new(num_spec_tokens)

        assert stats.num_spec_tokens == num_spec_tokens
        assert stats.num_drafts == 0
        assert stats.num_draft_tokens == 0
        assert stats.num_accepted_tokens == 0
        assert len(stats.num_accepted_tokens_per_pos) == num_spec_tokens
        assert all(count == 0 for count in stats.num_accepted_tokens_per_pos)

    def test_initialization_with_single_spec_token(self):
        """Test initialization with minimum valid spec tokens (1)."""
        stats = SpecDecodingStats.new(num_spec_tokens=1)

        assert stats.num_spec_tokens == 1
        assert len(stats.num_accepted_tokens_per_pos) == 1

    def test_observe_draft_basic(self):
        """Test basic draft observation updates all counters correctly."""
        stats = SpecDecodingStats.new(num_spec_tokens=3)

        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)

        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 3
        assert stats.num_accepted_tokens == 2
        # First 2 positions should be incremented
        assert stats.num_accepted_tokens_per_pos[0] == 1
        assert stats.num_accepted_tokens_per_pos[1] == 1
        assert stats.num_accepted_tokens_per_pos[2] == 0

    def test_observe_draft_full_acceptance(self):
        """Test draft observation when all tokens are accepted."""
        stats = SpecDecodingStats.new(num_spec_tokens=4)

        stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=4)

        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 4
        assert stats.num_accepted_tokens == 4
        assert all(count == 1 for count in stats.num_accepted_tokens_per_pos)

    def test_observe_draft_zero_acceptance(self):
        """Test draft observation when no tokens are accepted."""
        stats = SpecDecodingStats.new(num_spec_tokens=3)

        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=0)

        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 3
        assert stats.num_accepted_tokens == 0
        assert all(count == 0 for count in stats.num_accepted_tokens_per_pos)

    def test_observe_draft_multiple_observations(self):
        """Test accumulation across multiple draft observations."""
        stats = SpecDecodingStats.new(num_spec_tokens=3)

        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=3)
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=1)

        assert stats.num_drafts == 3
        assert stats.num_draft_tokens == 9
        assert stats.num_accepted_tokens == 6
        # Position 0: accepted in all 3 drafts
        assert stats.num_accepted_tokens_per_pos[0] == 3
        # Position 1: accepted in drafts 1 and 2
        assert stats.num_accepted_tokens_per_pos[1] == 2
        # Position 2: accepted only in draft 2
        assert stats.num_accepted_tokens_per_pos[2] == 1

    def test_observe_draft_partial_draft(self):
        """Test when fewer tokens are drafted than num_spec_tokens."""
        stats = SpecDecodingStats.new(num_spec_tokens=5)

        # Draft only 3 tokens, accept 2
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)

        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 3
        assert stats.num_accepted_tokens == 2
        # Per-position tracking still uses first num_accepted positions
        assert stats.num_accepted_tokens_per_pos[0] == 1
        assert stats.num_accepted_tokens_per_pos[1] == 1
        assert stats.num_accepted_tokens_per_pos[2] == 0

    def test_observe_draft_assertion_on_overflow(self):
        """Test that accepting more than num_spec_tokens raises assertion."""
        stats = SpecDecodingStats.new(num_spec_tokens=3)

        with pytest.raises(AssertionError):
            stats.observe_draft(num_draft_tokens=5, num_accepted_tokens=4)

    def test_acceptance_rate_calculation(self):
        """Test acceptance rate calculation from accumulated stats."""
        stats = SpecDecodingStats.new(num_spec_tokens=5)

        observations = [
            (5, 4),  # 80%
            (5, 3),  # 60%
            (5, 5),  # 100%
            (5, 2),  # 40%
            (5, 4),  # 80%
        ]

        for num_draft, num_accepted in observations:
            stats.observe_draft(num_draft, num_accepted)

        # Calculate acceptance rate
        acceptance_rate = (
            stats.num_accepted_tokens / stats.num_draft_tokens
            if stats.num_draft_tokens > 0
            else 0
        )

        assert stats.num_drafts == 5
        assert stats.num_draft_tokens == 25
        assert stats.num_accepted_tokens == 18
        assert abs(acceptance_rate - 0.72) < 0.001

    def test_mean_acceptance_length_calculation(self):
        """Test mean acceptance length calculation (includes bonus token)."""
        stats = SpecDecodingStats.new(num_spec_tokens=5)

        stats.observe_draft(5, 4)
        stats.observe_draft(5, 3)
        stats.observe_draft(5, 5)

        # Mean acceptance length = 1 + (total_accepted / num_drafts)
        mean_acceptance_length = 1 + (stats.num_accepted_tokens / stats.num_drafts)

        assert stats.num_accepted_tokens == 12
        assert stats.num_drafts == 3
        assert abs(mean_acceptance_length - 5.0) < 0.001

    def test_per_position_acceptance_rates(self):
        """Test per-position acceptance rate tracking."""
        stats = SpecDecodingStats.new(num_spec_tokens=4)

        # Simulate drafts with decreasing acceptance
        stats.observe_draft(4, 4)  # All accepted
        stats.observe_draft(4, 3)  # First 3 accepted
        stats.observe_draft(4, 2)  # First 2 accepted
        stats.observe_draft(4, 1)  # Only first accepted

        # Verify per-position counts
        assert stats.num_accepted_tokens_per_pos[0] == 4  # 100%
        assert stats.num_accepted_tokens_per_pos[1] == 3  # 75%
        assert stats.num_accepted_tokens_per_pos[2] == 2  # 50%
        assert stats.num_accepted_tokens_per_pos[3] == 1  # 25%

        # Calculate per-position rates
        pos_rates = [
            count / stats.num_drafts for count in stats.num_accepted_tokens_per_pos
        ]

        assert pos_rates[0] == 1.0
        assert pos_rates[1] == 0.75
        assert pos_rates[2] == 0.50
        assert pos_rates[3] == 0.25

    def test_large_number_of_observations(self):
        """Test stats with many observations for numerical stability."""
        stats = SpecDecodingStats.new(num_spec_tokens=3)

        # Use a fixed pattern for predictable results
        num_observations = 9999  # Divisible by 3
        for i in range(num_observations):
            # Vary acceptance between 1-3
            num_accepted = (i % 3) + 1
            stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=num_accepted)

        assert stats.num_drafts == num_observations
        assert stats.num_draft_tokens == num_observations * 3
        # Pattern: 1, 2, 3, 1, 2, 3, ... (each appears num_observations/3 times)
        # Total: (1 + 2 + 3) * (num_observations / 3) = 6 * 3333 = 19998
        expected_accepted = (1 + 2 + 3) * (num_observations // 3)
        assert stats.num_accepted_tokens == expected_accepted


class TestSpecDecodingLogging:
    """Test suite for SpecDecodingLogging aggregation and logging."""

    def test_initialization(self):
        """Test SpecDecodingLogging initializes with empty state."""
        logger = SpecDecodingLogging()

        assert logger.num_drafts == []
        assert logger.num_draft_tokens == []
        assert logger.num_accepted_tokens == []
        assert logger.accepted_tokens_per_pos_lists == []

    def test_reset(self):
        """Test reset() clears all accumulated data."""
        logger = SpecDecodingLogging()

        # Add some data
        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 2)
        logger.observe(stats)

        # Reset
        logger.reset()

        assert logger.num_drafts == []
        assert logger.num_draft_tokens == []
        assert logger.num_accepted_tokens == []
        assert logger.accepted_tokens_per_pos_lists == []

    def test_observe_single_stats(self):
        """Test observe() correctly accumulates single stats object."""
        logger = SpecDecodingLogging()

        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 2)
        stats.observe_draft(3, 3)

        logger.observe(stats)

        assert logger.num_drafts == [2]
        assert logger.num_draft_tokens == [6]
        assert logger.num_accepted_tokens == [5]
        assert len(logger.accepted_tokens_per_pos_lists) == 1

    def test_observe_multiple_stats(self):
        """Test observe() correctly accumulates multiple stats objects."""
        logger = SpecDecodingLogging()

        # First stats
        stats1 = SpecDecodingStats.new(num_spec_tokens=3)
        stats1.observe_draft(3, 2)
        logger.observe(stats1)

        # Second stats
        stats2 = SpecDecodingStats.new(num_spec_tokens=3)
        stats2.observe_draft(3, 3)
        logger.observe(stats2)

        assert logger.num_drafts == [1, 1]
        assert logger.num_draft_tokens == [3, 3]
        assert logger.num_accepted_tokens == [2, 3]
        assert len(logger.accepted_tokens_per_pos_lists) == 2

    def test_log_empty_stats(self):
        """Test log() with no observations returns early without logging."""
        logger = SpecDecodingLogging()
        mock_log_fn = MagicMock()

        logger.log(log_fn=mock_log_fn)

        mock_log_fn.assert_not_called()

    def test_log_formats_correctly(self):
        """Test log() formats and outputs metrics correctly."""
        logger = SpecDecodingLogging()

        # Create stats with known values
        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 2)
        stats.observe_draft(3, 3)

        logger.observe(stats)
        mock_log_fn = MagicMock()

        logger.log(log_fn=mock_log_fn)

        # Verify log was called
        mock_log_fn.assert_called_once()
        call_args = mock_log_fn.call_args

        # Check the format string and arguments
        format_string = call_args[0][0]
        assert "Mean acceptance length" in format_string
        assert "Accepted throughput" in format_string
        assert "Drafted throughput" in format_string

    def test_log_calculates_acceptance_rate(self):
        """Test log() correctly calculates acceptance rate."""
        logger = SpecDecodingLogging()

        stats = SpecDecodingStats.new(num_spec_tokens=4)
        # 10 drafts of 4 tokens each, 30 accepted = 75% rate
        for _ in range(10):
            stats.observe_draft(4, 3)

        logger.observe(stats)
        mock_log_fn = MagicMock()

        logger.log(log_fn=mock_log_fn)

        # Extract the acceptance rate from the call
        call_args = mock_log_fn.call_args[0]
        # Last positional argument is the acceptance rate percentage
        acceptance_rate_percent = call_args[-1]
        assert abs(acceptance_rate_percent - 75.0) < 0.1

    def test_log_handles_zero_draft_tokens(self):
        """Test log() handles zero draft tokens (nan acceptance rate)."""
        logger = SpecDecodingLogging()

        # Create stats with zero draft tokens
        stats = SpecDecodingStats.new(num_spec_tokens=3)
        # Manually set to simulate edge case
        stats.num_drafts = 1
        stats.num_draft_tokens = 0
        stats.num_accepted_tokens = 0

        logger.observe(stats)
        mock_log_fn = MagicMock()

        logger.log(log_fn=mock_log_fn)

        # Should still log, but with nan acceptance rate
        mock_log_fn.assert_called_once()

    def test_log_resets_after_logging(self):
        """Test log() calls reset() after logging."""
        logger = SpecDecodingLogging()

        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 2)
        logger.observe(stats)

        mock_log_fn = MagicMock()
        logger.log(log_fn=mock_log_fn)

        # Data should be cleared after log
        assert logger.num_drafts == []
        assert logger.num_draft_tokens == []
        assert logger.num_accepted_tokens == []

    def test_log_per_position_rates(self):
        """Test log() correctly calculates per-position acceptance rates."""
        logger = SpecDecodingLogging()

        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 3)  # All accepted
        stats.observe_draft(3, 2)  # First 2 accepted
        stats.observe_draft(3, 1)  # First 1 accepted

        logger.observe(stats)
        mock_log_fn = MagicMock()

        logger.log(log_fn=mock_log_fn)

        # Verify the per-position string is in the output
        mock_log_fn.assert_called_once()
        format_string = mock_log_fn.call_args[0][0]
        assert "Per-position acceptance rate" in format_string


class TestSpecDecodingProm:
    """Test suite for SpecDecodingProm Prometheus metrics."""

    def test_initialization_without_spec_config(self):
        """Test initialization when speculative config is None."""
        # No need to mock when spec_config is None - no counters are created
        prom = SpecDecodingProm(
            speculative_config=None,
            labelnames=["model_name"],
            per_engine_labelvalues={0: ["test_model"]},
        )

        assert prom.spec_decoding_enabled is False

    def test_initialization_with_spec_config(self):
        """Test initialization with valid speculative config."""
        mock_config = MagicMock()
        mock_config.num_speculative_tokens = 3

        # Patch the _counter_cls class attribute directly
        mock_counter_cls = MagicMock()
        mock_counter_instance = MagicMock()
        mock_counter_cls.return_value = mock_counter_instance
        mock_counter_instance.labels.return_value = MagicMock()

        with patch.object(SpecDecodingProm, "_counter_cls", mock_counter_cls):
            prom = SpecDecodingProm(
                speculative_config=mock_config,
                labelnames=["model_name"],
                per_engine_labelvalues={0: ["test_model"]},
            )

            assert prom.spec_decoding_enabled is True

    def test_observe_when_disabled(self):
        """Test observe() does nothing when spec decode is disabled."""
        prom = SpecDecodingProm(
            speculative_config=None,
            labelnames=["model_name"],
            per_engine_labelvalues={0: ["test_model"]},
        )

        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(3, 2)

        # Should not raise any errors
        prom.observe(stats, engine_idx=0)

    def test_observe_when_enabled(self):
        """Test observe() increments counters when spec decode is enabled."""
        mock_config = MagicMock()
        mock_config.num_speculative_tokens = 3

        mock_counter_cls = MagicMock()
        mock_counter_instance = MagicMock()
        mock_counter_cls.return_value = mock_counter_instance
        mock_labeled_counter = MagicMock()
        mock_counter_instance.labels.return_value = mock_labeled_counter

        with patch.object(SpecDecodingProm, "_counter_cls", mock_counter_cls):
            prom = SpecDecodingProm(
                speculative_config=mock_config,
                labelnames=["model_name"],
                per_engine_labelvalues={0: ["test_model"]},
            )

            stats = SpecDecodingStats.new(num_spec_tokens=3)
            stats.observe_draft(3, 2)

            prom.observe(stats, engine_idx=0)

            # Verify counters were incremented (3 main counters + per-position)
            assert mock_labeled_counter.inc.call_count >= 3

    def test_counter_names(self):
        """Test that correct Prometheus counter names are used."""
        mock_config = MagicMock()
        mock_config.num_speculative_tokens = 2

        counter_names = []

        def capture_counter_name(*args, **kwargs):
            counter_names.append(kwargs.get("name"))
            mock_counter = MagicMock()
            mock_counter.labels.return_value = MagicMock()
            return mock_counter

        with patch.object(SpecDecodingProm, "_counter_cls", side_effect=capture_counter_name):
            SpecDecodingProm(
                speculative_config=mock_config,
                labelnames=["model_name"],
                per_engine_labelvalues={0: ["test_model"]},
            )

        assert "vllm:spec_decode_num_drafts" in counter_names
        assert "vllm:spec_decode_num_draft_tokens" in counter_names
        assert "vllm:spec_decode_num_accepted_tokens" in counter_names
        assert "vllm:spec_decode_num_accepted_tokens_per_pos" in counter_names

    def test_multi_engine_support(self):
        """Test that metrics work with multiple engines."""
        mock_config = MagicMock()
        mock_config.num_speculative_tokens = 2

        mock_counter_cls = MagicMock()
        mock_counter_instance = MagicMock()
        mock_counter_cls.return_value = mock_counter_instance

        mock_labeled_counters = {}
        call_count = [0]

        def create_labeled_counter(*args):
            key = (args, call_count[0])
            call_count[0] += 1
            mock_labeled_counters[key] = MagicMock()
            return mock_labeled_counters[key]

        mock_counter_instance.labels.side_effect = create_labeled_counter

        with patch.object(SpecDecodingProm, "_counter_cls", mock_counter_cls):
            prom = SpecDecodingProm(
                speculative_config=mock_config,
                labelnames=["model_name"],
                per_engine_labelvalues={
                    0: ["engine_0"],
                    1: ["engine_1"],
                },
            )

            # Both engines should have counters
            assert 0 in prom.counter_spec_decode_num_drafts
            assert 1 in prom.counter_spec_decode_num_drafts


class TestSpecDecodingFallback:
    """Test suite for speculative decoding fallback mechanisms."""

    @pytest.fixture
    def target_model(self):
        """Small target model for testing."""
        return "facebook/opt-125m"

    @pytest.fixture
    def test_prompts(self):
        """Generate test prompts for testing."""
        return [
            "Hello, my name is",
            "The capital of France is",
            "To be or not to be",
        ]

    def test_graceful_degradation_without_spec_config(
        self, target_model, test_prompts
    ):
        """Test that LLM works normally without speculative config."""
        from vllm import LLM, SamplingParams
        from vllm.distributed import cleanup_dist_env_and_memory

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=15,
            seed=42,
        )

        llm = LLM(
            model=target_model,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )

        try:
            outputs = llm.generate(test_prompts, sampling_params)

            assert len(outputs) == len(test_prompts)
            for i, output in enumerate(outputs):
                assert len(output.outputs[0].token_ids) > 0, (
                    f"No tokens generated for prompt {i}"
                )
                assert output.outputs[0].text.strip() != "", (
                    f"Empty text for prompt {i}"
                )
        finally:
            del llm
            cleanup_dist_env_and_memory()


class TestSpeculativeConfigValidation:
    """Test configuration validation and error handling."""

    @pytest.fixture
    def target_model(self):
        """Small target model for testing."""
        return "facebook/opt-125m"

    def test_invalid_num_speculative_tokens_zero(self, target_model):
        """Test that num_speculative_tokens=0 raises ValueError."""
        from vllm import LLM

        with pytest.raises(ValueError) as exc_info:
            LLM(
                model=target_model,
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": 0,
                    "prompt_lookup_max": 5,
                },
                enforce_eager=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "0" in error_msg or "speculative" in error_msg

    def test_invalid_num_speculative_tokens_negative(self, target_model):
        """Test that negative num_speculative_tokens raises ValueError."""
        from vllm import LLM

        with pytest.raises(ValueError) as exc_info:
            LLM(
                model=target_model,
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": -1,
                    "prompt_lookup_max": 5,
                },
                enforce_eager=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "speculative" in error_msg or "greater" in error_msg

    def test_invalid_tensor_parallel_size_arg_name(self, target_model):
        """Test that 'tensor_parallel_size' in spec config raises ValueError."""
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=target_model,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 5,
                "tensor_parallel_size": 1,  # Invalid - should be draft_tensor_parallel_size
            },
        )

        with pytest.raises(ValueError) as exc_info:
            engine_args.create_engine_config()

        error_msg = str(exc_info.value).lower()
        assert "tensor_parallel_size" in error_msg or "draft_tensor_parallel_size" in error_msg

    def test_ngram_defaults_lookup_params(self, target_model):
        """Test that ngram method defaults lookup params if not provided."""
        from vllm import LLM
        from vllm.distributed import cleanup_dist_env_and_memory

        # ngram defaults to prompt_lookup_min=5 and prompt_lookup_max=5
        # This should NOT raise an error
        llm = LLM(
            model=target_model,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3,
                # prompt_lookup_min and prompt_lookup_max will default to 5
            },
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )

        try:
            # Verify the config was created successfully
            spec_config = llm.llm_engine.vllm_config.speculative_config
            assert spec_config is not None
            assert spec_config.num_speculative_tokens == 3
        finally:
            del llm
            cleanup_dist_env_and_memory()

    def test_ngram_invalid_lookup_range(self, target_model):
        """Test that prompt_lookup_min > prompt_lookup_max raises ValueError."""
        from vllm import LLM

        with pytest.raises(ValueError) as exc_info:
            LLM(
                model=target_model,
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": 3,
                    "prompt_lookup_min": 10,
                    "prompt_lookup_max": 5,  # Invalid: min > max
                },
                enforce_eager=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "prompt_lookup" in error_msg or "must be" in error_msg


class TestBaselineFunctionality:
    """Test baseline (non-speculative) functionality as foundation for spec decode testing."""

    @pytest.fixture
    def target_model(self):
        """Small target model for testing."""
        return "facebook/opt-125m"

    def test_determinism_with_seed(self, target_model):
        """Test that outputs are deterministic with fixed seed."""
        from vllm import LLM, SamplingParams
        from vllm.distributed import cleanup_dist_env_and_memory

        prompt = "Hello, how are you doing today?"
        sampling_params = SamplingParams(
            temperature=0.8,  # Non-zero to test seeding
            max_tokens=20,
            seed=12345,
        )

        # First run
        llm1 = LLM(
            model=target_model,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )
        output1 = llm1.generate([prompt], sampling_params)[0]
        text1 = output1.outputs[0].text
        del llm1
        cleanup_dist_env_and_memory()

        # Second run with same seed
        llm2 = LLM(
            model=target_model,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )
        output2 = llm2.generate([prompt], sampling_params)[0]
        text2 = output2.outputs[0].text
        del llm2
        cleanup_dist_env_and_memory()

        assert text1 == text2, (
            f"Non-deterministic output:\nRun 1: {text1}\nRun 2: {text2}"
        )

    def test_batch_processing_varying_lengths(self, target_model):
        """Test batch processing with varying prompt lengths."""
        from vllm import LLM, SamplingParams
        from vllm.distributed import cleanup_dist_env_and_memory

        prompts = [
            "Short",
            "This is a medium length prompt",
            "This is a much longer prompt with many more words to test "
            "how the system handles varying input lengths in a batch",
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=15,
            seed=42,
        )

        llm = LLM(
            model=target_model,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )

        try:
            outputs = llm.generate(prompts, sampling_params)

            assert len(outputs) == len(prompts)
            for i, output in enumerate(outputs):
                assert len(output.outputs[0].token_ids) > 0, (
                    f"No tokens for prompt {i}"
                )
                assert output.outputs[0].text.strip() != "", (
                    f"Empty output for prompt {i}"
                )
        finally:
            del llm
            cleanup_dist_env_and_memory()

    def test_multiple_temperatures_single_llm(self, target_model):
        """Test different temperatures with a single LLM instance (efficient)."""
        from vllm import LLM, SamplingParams
        from vllm.distributed import cleanup_dist_env_and_memory

        prompt = "The meaning of life is"
        temperatures = [0.0, 0.5, 1.0]

        llm = LLM(
            model=target_model,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )

        try:
            for temp in temperatures:
                sampling_params = SamplingParams(
                    temperature=temp,
                    max_tokens=15,
                    seed=42,
                )

                outputs = llm.generate([prompt], sampling_params)

                assert len(outputs) == 1
                assert len(outputs[0].outputs[0].token_ids) > 0, (
                    f"No tokens generated at temperature {temp}"
                )
        finally:
            del llm
            cleanup_dist_env_and_memory()


class TestIntegrationMetricsCollection:
    """Integration tests for metrics collection during actual inference."""

    @pytest.fixture
    def target_model(self):
        """Small target model for testing."""
        return "facebook/opt-125m"

    def test_stats_accumulation_pattern(self):
        """Test the pattern of accumulating stats across multiple iterations."""
        # Simulate scheduler behavior
        num_spec_tokens = 3
        total_stats = SpecDecodingStats.new(num_spec_tokens)
        logging = SpecDecodingLogging()

        # Simulate multiple scheduler steps
        for step in range(5):
            # Per-step stats
            step_stats = SpecDecodingStats.new(num_spec_tokens)

            # Simulate multiple requests in a step
            for _ in range(3):
                accepted = np.random.randint(0, num_spec_tokens + 1)
                step_stats.observe_draft(num_spec_tokens, accepted)

            # Accumulate into total
            total_stats.num_drafts += step_stats.num_drafts
            total_stats.num_draft_tokens += step_stats.num_draft_tokens
            total_stats.num_accepted_tokens += step_stats.num_accepted_tokens
            for i in range(num_spec_tokens):
                total_stats.num_accepted_tokens_per_pos[i] += (
                    step_stats.num_accepted_tokens_per_pos[i]
                )

            # Log per-step stats
            logging.observe(step_stats)

        # Verify accumulation
        assert total_stats.num_drafts == 15  # 5 steps * 3 requests
        assert total_stats.num_draft_tokens == 45  # 15 * 3 tokens

        # Verify logging observed all steps
        assert len(logging.num_drafts) == 5

    def test_metrics_reset_behavior(self):
        """Test that metrics reset correctly for windowed reporting."""
        logging = SpecDecodingLogging()

        # First window
        stats1 = SpecDecodingStats.new(num_spec_tokens=3)
        stats1.observe_draft(3, 2)
        logging.observe(stats1)

        mock_log = MagicMock()
        logging.log(log_fn=mock_log)

        # After log, should be reset
        assert len(logging.num_drafts) == 0

        # Second window
        stats2 = SpecDecodingStats.new(num_spec_tokens=3)
        stats2.observe_draft(3, 3)
        logging.observe(stats2)

        # Should only have second window data
        assert len(logging.num_drafts) == 1
        assert logging.num_accepted_tokens[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
