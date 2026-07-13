# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
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

    def test_occurrence_mode_detects_non_consecutive_ngram_count(self):
        """Occurrence mode stops when the tail N-gram reaches min_count."""
        repeated_ngram = list(range(16))
        token_ids = []
        for i in range(5):
            token_ids.extend(repeated_ngram)
            token_ids.append(1000 + i)
        token_ids.extend(repeated_ngram)
        params = RepetitionDetectionParams(
            max_pattern_size=16,
            min_pattern_size=16,
            min_count=6,
            mode="occurrence",
        )
        assert check_sequence_repetition(token_ids, params)

    def test_occurrence_mode_ignores_count_below_threshold(self):
        """Occurrence mode preserves output before the sixth 16-gram."""
        repeated_ngram = list(range(16))
        token_ids = []
        for i in range(4):
            token_ids.extend(repeated_ngram)
            token_ids.append(1000 + i)
        token_ids.extend(repeated_ngram)
        params = RepetitionDetectionParams(
            max_pattern_size=16,
            min_pattern_size=16,
            min_count=6,
            mode="occurrence",
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_occurrence_mode_allows_overlapping_ngrams(self):
        """Occurrence mode uses sliding N-grams, matching rollout fallback."""
        token_ids = [7] * 21
        params = RepetitionDetectionParams(
            max_pattern_size=16,
            min_pattern_size=16,
            min_count=6,
            mode="occurrence",
        )
        assert check_sequence_repetition(token_ids, params)

    def test_occurrence_rules_detect_short_ngram_with_higher_threshold(self):
        """Occurrence rules allow short N-grams only at severe repeat counts."""
        ngram = [11, 12, 13, 14]
        params = RepetitionDetectionParams(
            mode="occurrence",
            occurrence_rules=[(4, 20)],
        )

        assert not check_sequence_repetition(ngram * 19, params)
        assert check_sequence_repetition(ngram * 20, params)

    def test_occurrence_rules_keep_eight_gram_below_threshold(self):
        """The rollout-calibrated 8-gram rule fires on the tenth occurrence."""
        ngram = list(range(8))
        params = RepetitionDetectionParams(
            mode="occurrence",
            occurrence_rules=[(8, 10)],
        )

        assert not check_sequence_repetition(ngram * 9, params)
        assert check_sequence_repetition(ngram * 10, params)

    def test_occurrence_rules_require_occurrence_mode(self):
        with pytest.raises(ValueError):
            RepetitionDetectionParams(
                max_pattern_size=8,
                min_pattern_size=8,
                min_count=10,
                mode="consecutive",
                occurrence_rules=[(8, 10)],
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

    def test_occurrence_mode_stops_generation_incrementally(self):
        """Occurrence mode uses request-local counts in check_stop."""
        params = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=16,
                min_pattern_size=16,
                min_count=6,
                mode="occurrence",
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        repeated_ngram = list(range(16))

        for i in range(5):
            request.append_output_token_ids(repeated_ngram + [1000 + i])
            assert not check_stop(request, max_model_len=1024)

        request.append_output_token_ids(repeated_ngram)
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"
        assert (
            request.repetition_ngram_next_start[16]
            == len(request.output_token_ids) - 16 + 1
        )

    def test_occurrence_rules_stop_generation_incrementally(self):
        """Occurrence rules share request-local counts across decode steps."""
        params = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                mode="occurrence",
                occurrence_rules=[(8, 10), (4, 20)],
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        repeated_ngram = list(range(8))

        for _ in range(9):
            request.append_output_token_ids(repeated_ngram)
            assert not check_stop(request, max_model_len=1024)

        request.append_output_token_ids(repeated_ngram)
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"
        assert (
            request.repetition_ngram_next_start[8]
            == len(request.output_token_ids) - 8 + 1
        )
