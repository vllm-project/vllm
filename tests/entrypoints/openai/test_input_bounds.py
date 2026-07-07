# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for input size bounds on SamplingParams (GHSA-4fv6-24cq-h5pw,
GHSA-m9fh-2f8w-jhw9, GHSA-rp22-rc8r-qjmh)."""

from unittest.mock import patch

import pytest

from vllm.sampling_params import SamplingParams


class TestLogitBiasBounds:
    """logit_bias size is rejected in _verify_args when env var is set."""

    def test_logit_bias_within_limit(self):
        with patch("vllm.envs.VLLM_MAX_LOGIT_BIAS_SIZE", 10):
            params = SamplingParams(logit_bias={i: 1.0 for i in range(10)})
            assert len(params.logit_bias) == 10

    def test_logit_bias_exceeds_limit(self):
        with (
            patch("vllm.envs.VLLM_MAX_LOGIT_BIAS_SIZE", 5),
            pytest.raises(ValueError, match="logit_bias has 10 entries"),
        ):
            SamplingParams(logit_bias={i: 1.0 for i in range(10)})

    def test_logit_bias_no_limit_when_zero(self):
        with patch("vllm.envs.VLLM_MAX_LOGIT_BIAS_SIZE", 0):
            params = SamplingParams(logit_bias={i: 1.0 for i in range(50000)})
            assert len(params.logit_bias) == 50000


class TestStopTokenIdsBounds:
    """stop_token_ids size is rejected in _verify_args."""

    def test_stop_token_ids_within_limit(self):
        with patch("vllm.envs.VLLM_MAX_STOP_TOKEN_IDS", 128):
            params = SamplingParams(stop_token_ids=list(range(100)))
            assert len(params.stop_token_ids) == 100

    def test_stop_token_ids_exceeds_limit(self):
        with (
            patch("vllm.envs.VLLM_MAX_STOP_TOKEN_IDS", 10),
            pytest.raises(ValueError, match="stop_token_ids has 20"),
        ):
            SamplingParams(stop_token_ids=list(range(20)))

    def test_stop_token_ids_configurable(self):
        with patch("vllm.envs.VLLM_MAX_STOP_TOKEN_IDS", 500):
            params = SamplingParams(stop_token_ids=list(range(400)))
            assert len(params.stop_token_ids) == 400


class TestBadWordsBounds:
    """bad_words count and per-entry length are rejected in _verify_args."""

    def test_bad_words_within_limit(self):
        with patch("vllm.envs.VLLM_MAX_BAD_WORDS", 100):
            params = SamplingParams(bad_words=["word"] * 50)
            assert len(params.bad_words) == 50

    def test_bad_words_exceeds_count_limit(self):
        with (
            patch("vllm.envs.VLLM_MAX_BAD_WORDS", 5),
            pytest.raises(ValueError, match="bad_words has 10 entries"),
        ):
            SamplingParams(bad_words=["word"] * 10)

    def test_bad_word_exceeds_length_limit(self):
        with (
            patch("vllm.envs.VLLM_MAX_BAD_WORD_LENGTH", 10),
            pytest.raises(ValueError, match="exceeds the maximum of 10"),
        ):
            SamplingParams(bad_words=["a" * 100])

    def test_bad_words_empty_string_rejected(self):
        with pytest.raises(ValueError, match="cannot contain an empty string"):
            SamplingParams(bad_words=[""])

    def test_bad_words_configurable(self):
        with patch("vllm.envs.VLLM_MAX_BAD_WORDS", 5000):
            params = SamplingParams(bad_words=["word"] * 3000)
            assert len(params.bad_words) == 3000
