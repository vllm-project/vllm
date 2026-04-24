# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import _has_hit_token_repetition_limit
from vllm.v1.request import Request, RequestStatus


def _make_request(
    output_token_ids: list[int],
    request_id: str = "test",
) -> Request:
    req = Request(
        request_id=request_id,
        prompt_token_ids=[0],
        sampling_params=SamplingParams(max_tokens=1000),
        pooling_params=None,
    )
    if output_token_ids:
        req.append_output_token_ids(output_token_ids)
    req.status = RequestStatus.RUNNING
    return req


DETECTION_CASES = [
    pytest.param(
        [1, 2, 3, 4, 4, 4],
        3,
        1,
        True,
        id="single_token_repeated",
    ),
    pytest.param(
        [1, 2, 3, 4, 4, 4],
        3,
        2,
        True,
        id="single_token_detected_with_max_seq_2",
    ),
    pytest.param(
        [1, 2, 1, 2, 1, 2],
        3,
        2,
        True,
        id="pair_detected",
    ),
    pytest.param(
        [4, 4, 4],
        3,
        1,
        True,
        id="exact_count",
    ),
    pytest.param(
        [9, 4, 4, 4],
        3,
        1,
        True,
        id="prefix_then_exact_count",
    ),
    pytest.param(
        [1, 2, 1, 2, 1, 2],
        3,
        10,
        True,
        id="max_seq_clamped_to_available",
    ),
    pytest.param(
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        3,
        3,
        True,
        id="triplet_repetition",
    ),
    pytest.param(
        [7] * 100,
        50,
        1,
        True,
        id="long_single_token_run",
    ),
    pytest.param(
        list(range(100)) + [42] * 20,
        20,
        1,
        True,
        id="repetition_only_in_tail",
    ),
    pytest.param(
        [5, 3, 7],
        1,
        1,
        True,
        id="limit_1_always_triggers",
    ),
    pytest.param(
        [1, 2, 3],
        1,
        5,
        True,
        id="limit_1_with_larger_max_seq",
    ),
    pytest.param(
        [1, 2, 3, 5, 5],
        2,
        1,
        True,
        id="limit_2_single_token",
    ),
    pytest.param(
        [1, 2, 1, 2],
        2,
        2,
        True,
        id="limit_2_pair",
    ),
    pytest.param(
        [1, 2, 3, 4] * 3,
        3,
        4,
        True,
        id="period_k4",
    ),
    pytest.param(
        [5, 5, 5, 5, 5, 5],
        3,
        5,
        True,
        id="identical_tokens_detected_at_k1",
    ),
    pytest.param(
        [42] * 100,
        100,
        1,
        True,
        id="large_limit_exact",
    ),
    pytest.param(
        list(range(50)) + [7] * 5,
        5,
        1,
        True,
        id="late_bootstrap_match",
    ),
]

NO_DETECTION_CASES = [
    pytest.param(
        [1, 2, 3, 4, 5, 6],
        3,
        1,
        False,
        id="no_repetition_k1",
    ),
    pytest.param(
        [1, 2, 1, 2, 1, 2],
        3,
        1,
        False,
        id="pair_not_detected_max_seq_1",
    ),
    pytest.param(
        [1, 2, 3, 4, 5, 6],
        3,
        2,
        False,
        id="no_repetition_k2",
    ),
    pytest.param(
        [4, 4],
        3,
        1,
        False,
        id="too_few_tokens",
    ),
    pytest.param(
        [1, 2, 3, 1, 2, 3, 1, 2, 4],
        3,
        3,
        False,
        id="almost_repeating_last_token_differs",
    ),
    pytest.param(
        [],
        3,
        1,
        False,
        id="empty_tokens",
    ),
    pytest.param(
        [1],
        3,
        1,
        False,
        id="single_token",
    ),
    pytest.param(
        [1, 2, 3, 4, 5],
        2,
        1,
        False,
        id="limit_2_no_match",
    ),
    pytest.param(
        [1, 2, 3, 4] * 2,
        3,
        4,
        False,
        id="period_k4_not_enough_reps",
    ),
    pytest.param(
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        3,
        2,
        False,
        id="max_seq_too_small_for_pattern",
    ),
    pytest.param(
        [5, 5, 5, 5, 1, 2, 3],
        3,
        1,
        False,
        id="repetition_in_middle_not_tail",
    ),
    pytest.param(
        list(range(50)),
        3,
        10,
        False,
        id="all_unique_large_max_seq",
    ),
    pytest.param(
        [42] * 99,
        100,
        1,
        False,
        id="large_limit_one_short",
    ),
    pytest.param(
        [1, 2, 1, 2],
        3,
        2,
        False,
        id="one_short_of_pair_detection",
    ),
    pytest.param(
        list(range(50)) + [7, 7, 7, 7, 8],
        5,
        1,
        False,
        id="late_bootstrap_no_match",
    ),
]


class TestBootstrapDetection:
    """Single-call tests where all tokens are provided upfront."""

    @pytest.mark.parametrize(
        "tokens, limit, max_seq, expected",
        DETECTION_CASES + NO_DETECTION_CASES,
    )
    def test_detection(self, tokens, limit, max_seq, expected):
        r = _make_request(tokens)
        result = _has_hit_token_repetition_limit(r, limit, max_seq)
        assert result == expected


class TestIncrementalDetection:
    """Tests that simulate the real scheduler loop:
    append one token, check, repeat."""

    def test_single_token_repeat(self):
        req = _make_request([1, 2, 3])
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(4)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(4)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(4)
        assert _has_hit_token_repetition_limit(req, 3, 1)

    def test_streak_reset(self):
        """Streak from one token breaks, new token repeats."""
        req = _make_request([1, 2, 3, 4, 4])
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert _has_hit_token_repetition_limit(req, 3, 1)

    def test_pair_pattern(self):
        """Pair seq_len becomes trackable as max_possible grows."""
        req = _make_request([1])
        assert not _has_hit_token_repetition_limit(req, 3, 2)
        for tok in [2, 1, 2, 1]:
            req.append_output_token_ids(tok)
            assert not _has_hit_token_repetition_limit(req, 3, 2)
        req.append_output_token_ids(2)
        assert _has_hit_token_repetition_limit(req, 3, 2)

    def test_triplet_pattern(self):
        """max_possible grows through 1 -> 2 -> 3."""
        req = _make_request([1])
        assert not _has_hit_token_repetition_limit(req, 3, 3)
        for tok in [2, 3, 1, 2, 3, 1, 2]:
            req.append_output_token_ids(tok)
            assert not _has_hit_token_repetition_limit(req, 3, 3)
        req.append_output_token_ids(3)
        assert _has_hit_token_repetition_limit(req, 3, 3)

    def test_pattern_break_then_new_pattern(self):
        req = _make_request([1, 1])
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(2)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(2)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(2)
        assert _has_hit_token_repetition_limit(req, 3, 1)

    def test_no_false_positive_alternating(self):
        """Alternating tokens never trigger at k=1."""
        req = _make_request([1])
        for _ in range(50):
            req.append_output_token_ids(2)
            assert not _has_hit_token_repetition_limit(req, 3, 1)
            req.append_output_token_ids(1)
            assert not _has_hit_token_repetition_limit(req, 3, 1)

    def test_speculative_batch_completes_pattern(self):
        """Speculative batch processed per-token, matching
        the real scheduler loop."""
        req = _make_request([1, 2, 3])
        assert not _has_hit_token_repetition_limit(req, 3, 2)
        speculative_tokens = [1, 2, 1, 2, 1, 2]
        for tok in speculative_tokens[:5]:
            req.append_output_token_ids(tok)
            assert not _has_hit_token_repetition_limit(req, 3, 2)
        req.append_output_token_ids(speculative_tokens[5])
        assert _has_hit_token_repetition_limit(req, 3, 2)

    def test_speculative_early_stop(self):
        """Repetition detected mid-batch; scheduler would
        trim remaining speculative tokens."""
        req = _make_request([1, 2])
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert not _has_hit_token_repetition_limit(req, 3, 1)
        req.append_output_token_ids(5)
        assert _has_hit_token_repetition_limit(req, 3, 1)


class TestCheckStopIntegration:
    """Tests that check_stop() respects the repetition env vars."""

    def test_disabled_by_default(self, monkeypatch):
        import vllm.envs

        monkeypatch.setattr(vllm.envs, "VLLM_REPETITION_LIMIT", 0)
        monkeypatch.setattr(vllm.envs, "VLLM_REPETITION_MAX_SEQUENCE_LENGTH", 0)

        from vllm.v1.core.sched.utils import check_stop

        req = _make_request([1, 2, 4, 4, 4], "req-disabled")
        result = check_stop(req, max_model_len=4096)
        assert not result
        assert req.status == RequestStatus.RUNNING

    def test_enabled_triggers_error(self, monkeypatch):
        import vllm.envs

        monkeypatch.setattr(vllm.envs, "VLLM_REPETITION_LIMIT", 3)
        monkeypatch.setattr(vllm.envs, "VLLM_REPETITION_MAX_SEQUENCE_LENGTH", 1)

        from vllm.v1.core.sched.utils import check_stop

        req = _make_request([1, 2, 4, 4, 4], "req-enabled")
        result = check_stop(req, max_model_len=4096)
        assert result
        assert req.status == RequestStatus.FINISHED_REPETITION
