# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import (
    RollingHashState,
    check_sequence_repetition,
    check_stop,
)
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
# ROLLING-HASH ALGORITHM
# ============================================================================


class TestRollingHash:
    """Tests for the rolling_hash algorithm and its incremental state."""

    def test_matches_naive_on_shared_inputs(self):
        """rolling_hash and naive must agree whenever both are enabled."""
        cases = [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3], 5, 1, 3, True),
            ([1, 2, 3, 1, 2, 3], 5, 1, 3, False),
            ([1, 2] * 10, 5, 1, 4, True),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9], 5, 2, 2, False),
            ([7] * 20, 5, 1, 5, True),
        ]
        for tokens, mx, mn, k, expected in cases:
            naive = RepetitionDetectionParams(
                max_pattern_size=mx, min_pattern_size=mn, min_count=k
            )
            rh = RepetitionDetectionParams(
                max_pattern_size=mx,
                min_pattern_size=mn,
                min_count=k,
                algorithm="rolling_hash",
            )
            assert check_sequence_repetition(tokens, naive) is expected
            assert check_sequence_repetition(tokens, rh) is expected

    def test_unbounded_detects_long_pattern(self):
        """max_pattern_size=0 in rolling_hash mode catches long patterns
        that the bounded naive scan would miss."""
        pattern = list(range(50))  # length 50, beyond a typical naive cap
        token_ids = pattern * 4
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        assert check_sequence_repetition(token_ids, params)

    def test_state_matches_recompute(self):
        """Incremental state must produce the same result as a one-shot
        recompute, step by step."""
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=1,
            min_count=3,
            algorithm="rolling_hash",
        )
        state = RollingHashState()
        token_ids: list[int] = []
        for tok in [1, 5, 9, 2, 7, 3, 4, 7, 3, 4, 7, 3, 4]:
            token_ids.append(tok)
            inc = check_sequence_repetition(token_ids, params, state=state)
            recomp = check_sequence_repetition(token_ids, params)
            assert inc == recomp
            assert state.n == len(token_ids)

    def test_state_handles_truncation(self):
        """Speculative-decode rollback: when token_ids shrinks, state
        drops trailing hashes so the next append is correct."""
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        state = RollingHashState()
        check_sequence_repetition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], params, state=state)
        assert state.n == 10
        # Roll back 4 tokens then re-extend with a different tail.
        replayed = [1, 2, 3, 4, 5, 6, 99, 99, 99, 99, 99, 99]
        check_sequence_repetition(replayed[:6], params, state=state)
        assert state.n == 6
        inc = check_sequence_repetition(replayed, params, state=state)
        recomp = check_sequence_repetition(replayed, params)
        assert inc == recomp
        assert inc  # all-99 tail triggers detection

    def test_check_stop_attaches_state(self):
        """check_stop lazily allocates RollingHashState on first call,
        only for algorithm='rolling_hash'."""
        rh_params = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=0,
                min_pattern_size=2,
                min_count=3,
                algorithm="rolling_hash",
            ),
        )
        rh_request = Request(
            request_id="rh",
            prompt_token_ids=[1, 2, 3],
            sampling_params=rh_params,
            pooling_params=None,
        )
        assert rh_request.repetition_hash_state is None
        rh_request.append_output_token_ids([10, 11, 12])
        assert not check_stop(rh_request, max_model_len=2048)
        assert isinstance(rh_request.repetition_hash_state, RollingHashState)

        naive_params = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=4, min_pattern_size=2, min_count=3
            ),
        )
        naive_request = Request(
            request_id="naive",
            prompt_token_ids=[1],
            sampling_params=naive_params,
            pooling_params=None,
        )
        naive_request.append_output_token_ids([5, 6, 7])
        assert not check_stop(naive_request, max_model_len=2048)
        assert naive_request.repetition_hash_state is None


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
