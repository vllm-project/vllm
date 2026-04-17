# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import (RepetitionDetectionParams, SamplingParams,
                                 StructuredOutputsParams)
from vllm.v1.core.sched.utils import check_sequence_repetition, check_stop
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# ============================================================================
# UNIT TESTS - check_sequence_repetition function
# ============================================================================


class TestCheckSequenceRepetition:
    """Unit tests for the check_sequence_repetition function"""

    def test_simple_repetition_detected(self):
        """Test detection of simple repetitive patterns"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_repetition_below_min_count(self):
        """Test that pattern below min_count is not detected"""
        token_ids = [1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_two_token_pattern(self):
        """Test detection of 2-token patterns"""
        token_ids = [1, 2, 1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=4,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_no_repetition_varied_sequence(self):
        """Test that non-repetitive sequences are not flagged"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=2,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_partial_repetition_not_detected(self):
        """Test that incomplete repetitions are not detected"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 4]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_empty_token_list(self):
        """Test with empty token list"""
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=2,
        )
        assert not check_sequence_repetition([], params)

    def test_detection_disabled_max_size_zero(self):
        """Test that zero max_pattern_size disables detection"""
        token_ids = [1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams()
        assert not check_sequence_repetition(token_ids, params)

    def test_invalid_min_count(self):
        """Test that min_count < 2 returns False"""
        token_ids = [1, 2, 1, 2]
        params = RepetitionDetectionParams()
        assert not check_sequence_repetition(token_ids, params)

    def test_repetition_at_end_of_sequence(self):
        """Test detection when repetition occurs at the end"""
        token_ids = [1, 2, 3, 4, 5, 6, 5, 6, 5, 6]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_large_pattern_many_repetitions(self):
        """Test large pattern repeated many times"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 5
        params = RepetitionDetectionParams(
            max_pattern_size=10,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)


# ============================================================================
# INTEGRATION TESTS - check_stop with repetition detection
# ============================================================================


class TestRepetitionDetectionIntegration:
    """Integration tests for repetition detection in check_stop"""

    def test_basic_repetition_stops_generation(self):
        """Test that repetition is detected and stops generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_detection_disabled_no_stop(self):
        """Test that disabled detection doesn't stop generation"""
        params = SamplingParams(
            max_tokens=100,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_repetition_respects_min_tokens(self):
        """Test that repetition detection respects min_tokens"""
        params = SamplingParams(
            min_tokens=10,
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_no_repetition_continues_generation(self):
        """Test that non-repetitive tokens don't stop generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 30, 40, 50, 60])
        assert not check_stop(request, max_model_len=1024)

    def test_pattern_at_size_boundary(self):
        """Test detection at exact pattern size boundary"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=3,
                min_pattern_size=3,
                min_count=2,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 30, 10, 20, 30])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_multiple_pattern_sizes_checked(self):
        """Test that function checks pattern sizes in range"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_eos_takes_precedence_over_repetition(self):
        """Test that EOS token stops before repetition check"""
        params = SamplingParams(
            max_tokens=100,
            stop_token_ids=[999],
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 999])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_STOPPED

    def test_min_pattern_size_filters_small_patterns(self):
        """Test that min_pattern_size filters out smaller patterns"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=3,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_high_repetition_threshold(self):
        """Test that high min_count requires many repetitions"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=5,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)


# ============================================================================
# STRUCTURED OUTPUT + REPETITION DETECTION COMPOSITION TESTS
# https://github.com/vllm-project/vllm/issues/40080
# ============================================================================


class TestStructuredOutputRepetitionDetection:
    """Tests that structured output requests auto-enable repetition detection
    and that the detection correctly terminates degenerate loops.

    Validates the mitigation for:
    https://github.com/vllm-project/vllm/issues/40080
    """

    _JSON_SCHEMA = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }

    def test_auto_enabled_for_json_schema(self):
        """SamplingParams with json structured_outputs should auto-populate
        repetition_detection when not explicitly set."""
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
        )
        assert params.repetition_detection is not None
        assert params.repetition_detection.max_pattern_size == 20
        assert params.repetition_detection.min_pattern_size == 3
        assert params.repetition_detection.min_count == 4

    def test_auto_enabled_for_json_object(self):
        """json_object mode also uses grammar constraints."""
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )
        assert params.repetition_detection is not None

    def test_not_auto_enabled_for_choice(self):
        """choice mode does not use grammar-constrained decoding."""
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(
                choice=["positive", "negative"],
            ),
        )
        assert params.repetition_detection is None

    def test_not_auto_enabled_for_regex(self):
        """regex mode does not use grammar bitmask in the same way."""
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(regex=r"\d+"),
        )
        assert params.repetition_detection is None

    def test_explicit_detection_not_overridden(self):
        """User-supplied repetition_detection must not be overwritten."""
        user_params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=3,
        )
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
            repetition_detection=user_params,
        )
        assert params.repetition_detection.max_pattern_size == 5
        assert params.repetition_detection.min_count == 3

    def test_explicit_disabled_detection_not_overridden(self):
        """User can explicitly disable by passing all-zero params."""
        disabled = RepetitionDetectionParams(
            max_pattern_size=0, min_pattern_size=0, min_count=0)
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
            repetition_detection=disabled,
        )
        # All zeros = disabled; should NOT be overridden
        assert params.repetition_detection.max_pattern_size == 0

    def test_no_auto_enable_without_structured_output(self):
        """Without structured_outputs, repetition_detection stays None."""
        params = SamplingParams(max_tokens=100)
        assert params.repetition_detection is None

    def test_detection_stops_loops_with_structured_output(self):
        """Verify repetition detection terminates loops during structured
        output generation (the Gemma 4 scenario)."""
        params = SamplingParams(
            max_tokens=2000,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        # Simulate: 3-gram [10, 20, 30] repeated 4 times = 12 tokens
        request.append_output_token_ids(
            [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30])
        assert check_stop(request, max_model_len=4096)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_non_repeating_structured_output_continues(self):
        """Normal (non-repeating) structured output must not be stopped."""
        params = SamplingParams(
            max_tokens=100,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 30, 40, 50, 60])
        assert not check_stop(request, max_model_len=4096)

    def test_repeated_json_elements_not_false_positive(self):
        """Valid JSON with repeated array elements must not be stopped.
        Simulates [{"k":"v"},{"k":"v"},{"k":"v"}] where each object
        shares some tokens but the overall pattern is below min_count."""
        params = SamplingParams(
            max_tokens=2000,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        # 3 similar objects with shared prefix but varying content
        # Pattern: [A,B,C,D,E,  A,B,C,D,F,  A,B,C,D,G] — not a repeat
        tokens = [10, 20, 30, 40, 50,
                  10, 20, 30, 40, 60,
                  10, 20, 30, 40, 70]
        request.append_output_token_ids(tokens)
        assert not check_stop(request, max_model_len=4096)

    def test_large_ngram_json_repetition_detected(self):
        """Simulate a large N-gram repetition resembling a JSON field
        being repeated (the exact Gemma 4 failure mode)."""
        params = SamplingParams(
            max_tokens=2000,
            structured_outputs=StructuredOutputsParams(
                json=self._JSON_SCHEMA,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        # 10-gram pattern repeated 4 times = 40 tokens
        pattern = list(range(100, 110))  # [100..109]
        request.append_output_token_ids(pattern * 4)
        assert check_stop(request, max_model_len=4096)
        assert request.status == RequestStatus.FINISHED_REPETITION
