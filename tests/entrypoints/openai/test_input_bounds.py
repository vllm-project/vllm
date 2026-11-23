# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for input size bounds on SamplingParams (GHSA-4fv6-24cq-h5pw,
GHSA-m9fh-2f8w-jhw9, GHSA-rp22-rc8r-qjmh)."""

from unittest.mock import patch

import pytest

from vllm.exceptions import VLLMValidationError
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
            pytest.raises(VLLMValidationError, match="logit_bias has 10 entries"),
        ):
            SamplingParams(logit_bias={i: 1.0 for i in range(10)})

    def test_logit_bias_no_limit_when_zero(self):
        with patch("vllm.envs.VLLM_MAX_LOGIT_BIAS_SIZE", 0):
            params = SamplingParams(logit_bias={i: 1.0 for i in range(10)})
            assert len(params.logit_bias) == 10


class TestBadWordsBounds:
    """Individual bad_words entries are bounded before tokenization."""

    def test_bad_word_exceeds_length_limit(self):
        with (
            patch("vllm.envs.VLLM_MAX_BAD_WORD_LENGTH", 10),
            pytest.raises(VLLMValidationError, match="exceeds the maximum of 10"),
        ):
            SamplingParams(bad_words=["a" * 11])

    def test_bad_word_at_length_limit(self):
        with patch("vllm.envs.VLLM_MAX_BAD_WORD_LENGTH", 4):
            params = SamplingParams(bad_words=["word"])
            assert params.bad_words == ["word"]
