# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import SamplingParams
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
        assert check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )

    def test_repetition_below_min_count(self):
        """Test that pattern below min_count is not detected"""
        token_ids = [1, 2, 3, 1, 2, 3]
        assert not check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )

    def test_two_token_pattern(self):
        """Test detection of 2-token patterns"""
        token_ids = [1, 2, 1, 2, 1, 2, 1, 2]
        assert check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=4,
        )

    def test_no_repetition_varied_sequence(self):
        """Test that non-repetitive sequences are not flagged"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert not check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=2,
        )

    def test_partial_repetition_not_detected(self):
        """Test that incomplete repetitions are not detected"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 4]
        assert not check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )

    def test_empty_token_list(self):
        """Test with empty token list"""
        assert not check_sequence_repetition(
            [],
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=2,
        )

    def test_detection_disabled_max_size_zero(self):
        """Test that zero max_pattern_size disables detection"""
        token_ids = [1, 2, 1, 2, 1, 2]
        assert not check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=0,
            min_repetition_pattern_size=2,
            repetition_min_count=2,
        )

    def test_invalid_min_count(self):
        """Test that min_count < 2 returns False"""
        token_ids = [1, 2, 1, 2]
        assert not check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=1,
        )

    def test_repetition_at_end_of_sequence(self):
        """Test detection when repetition occurs at the end"""
        token_ids = [1, 2, 3, 4, 5, 6, 5, 6, 5, 6]
        assert check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )

    def test_large_pattern_many_repetitions(self):
        """Test large pattern repeated many times"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 5
        assert check_sequence_repetition(
            token_ids,
            max_repetition_pattern_size=10,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )


# ============================================================================
# INTEGRATION TESTS - check_stop with repetition detection
# ============================================================================


class TestRepetitionDetectionIntegration:
    """Integration tests for repetition detection in check_stop"""

    def test_basic_repetition_stops_generation(self):
        """Test that repetition is detected and stops generation"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_detection_disabled_no_stop(self):
        """Test that disabled detection doesn't stop generation"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=0,
            min_repetition_pattern_size=0,
            repetition_min_count=0,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_repetition_respects_min_tokens(self):
        """Test that repetition detection respects min_tokens"""
        params = SamplingParams(
            min_tokens=10,
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_no_repetition_continues_generation(self):
        """Test that non-repetitive tokens don't stop generation"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 30, 40, 50, 60])
        assert not check_stop(request, max_model_len=1024)

    def test_pattern_at_size_boundary(self):
        """Test detection at exact pattern size boundary"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=3,
            min_repetition_pattern_size=3,
            repetition_min_count=2,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 30, 10, 20, 30])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_multiple_pattern_sizes_checked(self):
        """Test that function checks pattern sizes in range"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_eos_takes_precedence_over_repetition(self):
        """Test that EOS token stops before repetition check"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=999,
        )
        request.append_output_token_ids([10, 20, 10, 20, 999])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_STOPPED

    def test_min_pattern_size_filters_small_patterns(self):
        """Test that min_pattern_size filters out smaller patterns"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=3,
            repetition_min_count=3,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_high_repetition_threshold(self):
        """Test that high min_count requires many repetitions"""
        params = SamplingParams(
            max_tokens=100,
            max_repetition_pattern_size=5,
            min_repetition_pattern_size=2,
            repetition_min_count=5,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
            eos_token_id=0,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)
