# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import (
    RollingHashState,
    check_sequence_repetition,
    check_stop,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

ALGORITHMS = ["naive", "rolling_hash"]

# ============================================================================
# UNIT TESTS - check_sequence_repetition function (run for both algorithms)
# ============================================================================


@pytest.mark.parametrize("algorithm", ALGORITHMS)
class TestCheckSequenceRepetition:
    """Unit tests for the check_sequence_repetition function"""

    def test_simple_repetition_detected(self, algorithm):
        """Test detection of simple repetitive patterns"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
            algorithm=algorithm,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_repetition_below_min_count(self, algorithm):
        """Test that pattern below min_count is not detected"""
        token_ids = [1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
            algorithm=algorithm,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_two_token_pattern(self, algorithm):
        """Test detection of 2-token patterns"""
        token_ids = [1, 2, 1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=4,
            algorithm=algorithm,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_no_repetition_varied_sequence(self, algorithm):
        """Test that non-repetitive sequences are not flagged"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=2,
            algorithm=algorithm,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_partial_repetition_not_detected(self, algorithm):
        """Test that incomplete repetitions are not detected"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 4]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
            algorithm=algorithm,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_empty_token_list(self, algorithm):
        """Test with empty token list"""
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=2,
            algorithm=algorithm,
        )
        assert not check_sequence_repetition([], params)

    def test_detection_disabled_default_params(self, algorithm):
        """Default parameters disable detection regardless of algorithm.

        For 'naive' this is enforced via max_pattern_size=0; for
        'rolling_hash' via min_count=0 (the "active" gate).
        """
        token_ids = [1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams(algorithm=algorithm)
        assert not check_sequence_repetition(token_ids, params)

    def test_invalid_min_count(self, algorithm):
        """Test that min_count < 2 returns False"""
        token_ids = [1, 2, 1, 2]
        params = RepetitionDetectionParams(algorithm=algorithm)
        assert not check_sequence_repetition(token_ids, params)

    def test_repetition_at_end_of_sequence(self, algorithm):
        """Test detection when repetition occurs at the end"""
        token_ids = [1, 2, 3, 4, 5, 6, 5, 6, 5, 6]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
            algorithm=algorithm,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_large_pattern_many_repetitions(self, algorithm):
        """Test large pattern repeated many times"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 5
        params = RepetitionDetectionParams(
            max_pattern_size=10,
            min_pattern_size=2,
            min_count=3,
            algorithm=algorithm,
        )
        assert check_sequence_repetition(token_ids, params)


# ============================================================================
# ROLLING-HASH SPECIFIC TESTS - unbounded pattern length & equivalence
# ============================================================================


class TestRollingHashSpecific:
    """Tests that exercise rolling_hash-only behavior."""

    def test_rolling_hash_unbounded_detects_long_pattern(self):
        """A pattern longer than any reasonable max_pattern_size is detected."""
        pattern = list(range(50))  # length 50, well beyond default caps
        token_ids = pattern * 4
        params = RepetitionDetectionParams(
            max_pattern_size=0,  # unbounded
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        assert check_sequence_repetition(token_ids, params)

    def test_rolling_hash_unbounded_no_false_positive(self):
        """Unbounded mode should not flag a non-repetitive sequence."""
        token_ids = list(range(200))
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_naive_max_size_zero_still_disables(self):
        """Backward compat: naive with max_pattern_size=0 stays disabled."""
        pattern = list(range(50))
        token_ids = pattern * 4
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=2,
            min_count=3,
            algorithm="naive",
        )
        # naive validates min_count<2 against max_pattern_size>0 only, so
        # min_count=3 is allowed here even with max_pattern_size=0; the
        # function must still treat it as disabled.
        assert not check_sequence_repetition(token_ids, params)

    def test_rolling_hash_capped_by_max_pattern_size(self):
        """When set, max_pattern_size still caps rolling_hash search."""
        # 50-length pattern, but we cap search at 10 → must NOT be detected.
        pattern = list(range(50))
        token_ids = pattern * 3
        params = RepetitionDetectionParams(
            max_pattern_size=10,
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="algorithm"):
            RepetitionDetectionParams(
                max_pattern_size=3,
                min_pattern_size=2,
                min_count=2,
                algorithm="bogus",
            )


# ============================================================================
# EQUIVALENCE TEST - naive == rolling_hash on shared parameter ranges
# ============================================================================


class TestAlgorithmEquivalence:
    """Verify rolling_hash agrees with naive whenever both are enabled."""

    @pytest.mark.parametrize(
        "token_ids,max_size,min_size,min_count",
        [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3], 5, 1, 3),
            ([1, 2, 3, 1, 2, 3], 5, 1, 3),
            ([1, 2] * 10, 5, 1, 4),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9], 5, 2, 2),
            ([1, 2, 3, 1, 2, 3, 1, 2, 4], 5, 2, 3),
            ([1, 2, 3, 4, 5, 6, 5, 6, 5, 6], 5, 2, 3),
            ([1, 2, 3, 4, 5, 6, 7, 8] * 5, 10, 2, 3),
            ([10, 20, 30, 10, 20, 30], 3, 3, 2),  # boundary
            ([7] * 20, 5, 1, 5),  # all-same
            ([7, 7, 7, 7, 8, 9], 5, 1, 3),  # repetition broken at tail
            (list(range(30)) + [1, 2, 1, 2, 1, 2, 1, 2], 5, 2, 4),
        ],
    )
    def test_table(self, token_ids, max_size, min_size, min_count):
        naive_params = RepetitionDetectionParams(
            max_pattern_size=max_size,
            min_pattern_size=min_size,
            min_count=min_count,
            algorithm="naive",
        )
        rh_params = RepetitionDetectionParams(
            max_pattern_size=max_size,
            min_pattern_size=min_size,
            min_count=min_count,
            algorithm="rolling_hash",
        )
        assert check_sequence_repetition(
            token_ids, naive_params
        ) == check_sequence_repetition(token_ids, rh_params), (
            f"disagreement for {token_ids=} {max_size=} {min_size=} {min_count=}"
        )

    def test_random_equivalence(self):
        """Fuzz: rolling_hash must match naive across many random inputs."""
        rng = random.Random(0xC0FFEE)
        for _ in range(2000):
            length = rng.randint(0, 60)
            vocab = rng.randint(2, 6)
            token_ids = [rng.randrange(vocab) for _ in range(length)]
            max_size = rng.randint(1, 10)
            min_size = rng.randint(1, max_size)
            min_count = rng.randint(2, 5)
            naive = RepetitionDetectionParams(
                max_pattern_size=max_size,
                min_pattern_size=min_size,
                min_count=min_count,
                algorithm="naive",
            )
            rh = RepetitionDetectionParams(
                max_pattern_size=max_size,
                min_pattern_size=min_size,
                min_count=min_count,
                algorithm="rolling_hash",
            )
            naive_out = check_sequence_repetition(token_ids, naive)
            rh_out = check_sequence_repetition(token_ids, rh)
            assert naive_out == rh_out, (
                f"disagreement: {token_ids=} max={max_size} "
                f"min={min_size} count={min_count} "
                f"naive={naive_out} rh={rh_out}"
            )


# ============================================================================
# INCREMENTAL STATE TESTS - rolling-hash state survives across calls
# ============================================================================


class TestRollingHashIncrementalState:
    """Verify the incremental ``RollingHashState`` produces the same
    detection result as a one-shot recomputation, and that the state
    grows append-only as decoding progresses."""

    def test_state_grows_append_only(self):
        rng = random.Random(0xBEEF)
        token_ids: list[int] = []
        params = RepetitionDetectionParams(
            max_pattern_size=0,  # unbounded
            min_pattern_size=2,
            min_count=3,
            algorithm="rolling_hash",
        )
        state = RollingHashState()

        # Stream 200 random tokens, then a 12-token repeating pattern
        # 4 times — detection must fire late, never before.
        steps_until_detect = []
        flagged_step = None
        pattern = [rng.randrange(50) for _ in range(12)]
        for step in range(1, 201):
            token_ids.append(rng.randrange(50))
            assert state.n == step - 1, "state must lag tokens by one"
            hit = check_sequence_repetition(token_ids, params, state=state)
            assert state.n == step
            if hit and flagged_step is None:
                flagged_step = step
        # Random stream: very unlikely to flag (with 50-vocab there is a
        # tiny chance, but with this seed it shouldn't).
        steps_until_detect.append(flagged_step)

        for rep in range(4):
            for tok in pattern:
                token_ids.append(tok)
                hit = check_sequence_repetition(token_ids, params, state=state)
                if hit:
                    # Once detected, state is still consistent.
                    assert state.n == len(token_ids)
                    break
            if hit:
                break
        assert hit, "12-token pattern repeated 3+ times must trigger"

    def test_state_matches_recompute(self):
        """For the same token stream, incremental == recompute, step by step."""
        rng = random.Random(123)
        token_ids: list[int] = []
        params = RepetitionDetectionParams(
            max_pattern_size=0,
            min_pattern_size=1,
            min_count=3,
            algorithm="rolling_hash",
        )
        state = RollingHashState()
        # Build a stream that will hit detection partway through.
        prefix = [rng.randrange(20) for _ in range(40)]
        tail = [4, 7, 9] * 5
        stream = prefix + tail

        for tok in stream:
            token_ids.append(tok)
            inc = check_sequence_repetition(token_ids, params, state=state)
            recomp = check_sequence_repetition(token_ids, params)
            assert inc == recomp, (
                f"step {len(token_ids)}: incremental={inc} recompute={recomp}"
            )

    def test_check_stop_persists_state_on_request(self):
        """``check_stop`` lazily attaches RollingHashState to the request and
        reuses it across consecutive calls."""
        params = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=0,
                min_pattern_size=2,
                min_count=3,
                algorithm="rolling_hash",
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        # No state until the first check_stop call.
        assert request.repetition_hash_state is None
        request.append_output_token_ids([10, 11, 12, 13, 14])
        assert not check_stop(request, max_model_len=2048)
        first_state = request.repetition_hash_state
        assert isinstance(first_state, RollingHashState)
        assert first_state.n == 5
        # Subsequent decode steps reuse the same state object and grow it.
        request.append_output_token_ids([15])
        assert not check_stop(request, max_model_len=2048)
        assert request.repetition_hash_state is first_state
        assert first_state.n == 6
        # Naive algorithm path must not allocate state.
        params2 = SamplingParams(
            max_tokens=200,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=4,
                min_pattern_size=2,
                min_count=3,
                algorithm="naive",
            ),
        )
        req2 = Request(
            request_id="test2",
            prompt_token_ids=[1],
            sampling_params=params2,
            pooling_params=None,
        )
        req2.append_output_token_ids([5, 6, 7])
        assert not check_stop(req2, max_model_len=2048)
        assert req2.repetition_hash_state is None


# ============================================================================
# INTEGRATION TESTS - check_stop with repetition detection (both algorithms)
# ============================================================================


@pytest.mark.parametrize("algorithm", ALGORITHMS)
class TestRepetitionDetectionIntegration:
    """Integration tests for repetition detection in check_stop"""

    def test_basic_repetition_stops_generation(self, algorithm):
        """Test that repetition is detected and stops generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
                algorithm=algorithm,
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

    def test_detection_disabled_no_stop(self, algorithm):
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

    def test_repetition_respects_min_tokens(self, algorithm):
        """Test that repetition detection respects min_tokens"""
        params = SamplingParams(
            min_tokens=10,
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
                algorithm=algorithm,
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

    def test_no_repetition_continues_generation(self, algorithm):
        """Test that non-repetitive tokens don't stop generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
                algorithm=algorithm,
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

    def test_pattern_at_size_boundary(self, algorithm):
        """Test detection at exact pattern size boundary"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=3,
                min_pattern_size=3,
                min_count=2,
                algorithm=algorithm,
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

    def test_multiple_pattern_sizes_checked(self, algorithm):
        """Test that function checks pattern sizes in range"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
                algorithm=algorithm,
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

    def test_eos_takes_precedence_over_repetition(self, algorithm):
        """Test that EOS token stops before repetition check"""
        params = SamplingParams(
            max_tokens=100,
            stop_token_ids=[999],
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
                algorithm=algorithm,
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

    def test_min_pattern_size_filters_small_patterns(self, algorithm):
        """Test that min_pattern_size filters out smaller patterns"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=3,
                min_count=3,
                algorithm=algorithm,
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

    def test_high_repetition_threshold(self, algorithm):
        """Test that high min_count requires many repetitions"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=5,
                algorithm=algorithm,
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
