# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.sampling_params import (
    _MAX_ABSOLUTE_MIN_COUNT,
    _MAX_ABSOLUTE_PATTERN_SIZE,
    RepetitionDetectionParams,
    SamplingParams,
    VLLMValidationError,
)
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
# CAP / VALIDATION TESTS - server-side limits on repetition detection
# ============================================================================


def _mock_model_config(max_rep_cap: int = 2048) -> MagicMock:
    """Minimal mock satisfying SamplingParams.verify() for cap tests."""
    cfg = MagicMock()
    cfg.max_logprobs = 20
    cfg.get_vocab_size.return_value = 32000
    cfg.max_repetition_detection_pattern_size = max_rep_cap
    cfg.is_diffusion = False
    cfg.logits_processors = None
    return cfg


class TestRepetitionDetectionCaps:
    """Tests for server-side caps on repetition detection parameters."""

    def test_max_pattern_size_exceeds_server_cap(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=4096,
                min_pattern_size=1,
                min_count=2,
            ),
        )
        with pytest.raises(VLLMValidationError, match="exceeds the server"):
            params._validate_repetition_detection(_mock_model_config(max_rep_cap=2048))

    def test_max_pattern_size_within_server_cap(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=1024,
                min_pattern_size=1,
                min_count=2,
            ),
        )
        params._validate_repetition_detection(_mock_model_config(max_rep_cap=2048))

    def test_server_disabled_repetition_detection(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=1,
                min_count=2,
            ),
        )
        with pytest.raises(VLLMValidationError, match="disabled"):
            params._validate_repetition_detection(_mock_model_config(max_rep_cap=0))

    def test_server_uncapped_repetition_detection(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=60000,
                min_pattern_size=1,
                min_count=2,
            ),
        )
        params._validate_repetition_detection(_mock_model_config(max_rep_cap=-1))

    def test_disabled_detection_skips_validation(self):
        params = SamplingParams(max_tokens=100)
        params._validate_repetition_detection(_mock_model_config(max_rep_cap=0))

    def test_scan_work_budget_exceeded(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=2048,
                min_pattern_size=1,
                min_count=100,
            ),
        )
        with pytest.raises(VLLMValidationError, match="scan work"):
            params._validate_repetition_detection(_mock_model_config(max_rep_cap=2048))

    def test_scan_work_budget_ok(self):
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=2048,
                min_pattern_size=1,
                min_count=5,
            ),
        )
        params._validate_repetition_detection(_mock_model_config(max_rep_cap=2048))


class TestRepetitionDetectionAbsoluteCaps:
    """Tests for hard absolute limits in RepetitionDetectionParams."""

    def test_absolute_hard_cap_max_pattern_size(self):
        with pytest.raises(ValueError, match="max_pattern_size must be"):
            RepetitionDetectionParams(
                max_pattern_size=_MAX_ABSOLUTE_PATTERN_SIZE + 1,
                min_pattern_size=1,
                min_count=2,
            )

    def test_absolute_hard_cap_min_count(self):
        with pytest.raises(ValueError, match="min_count must be"):
            RepetitionDetectionParams(
                max_pattern_size=10,
                min_pattern_size=1,
                min_count=_MAX_ABSOLUTE_MIN_COUNT + 1,
            )

    def test_values_at_absolute_limit_accepted(self):
        params = RepetitionDetectionParams(
            max_pattern_size=_MAX_ABSOLUTE_PATTERN_SIZE,
            min_pattern_size=1,
            min_count=_MAX_ABSOLUTE_MIN_COUNT,
        )
        assert params.max_pattern_size == _MAX_ABSOLUTE_PATTERN_SIZE
        assert params.min_count == _MAX_ABSOLUTE_MIN_COUNT

    def test_normal_small_window_still_works(self):
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_scheduler_defense_rejects_oversized_window(self):
        token_ids = list(range(200000))
        params = RepetitionDetectionParams(
            max_pattern_size=_MAX_ABSOLUTE_PATTERN_SIZE,
            min_pattern_size=1,
            min_count=_MAX_ABSOLUTE_MIN_COUNT,
        )
        assert not check_sequence_repetition(token_ids, params)
