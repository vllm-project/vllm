# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that config fields reject negative values where they are not valid.

Covers the field_validator guards added for max_logprobs (ModelConfig) and
long_prefill_token_threshold (SchedulerConfig) per issue #43985.
"""
import pytest
from pydantic import ValidationError

from vllm.config.model import ModelConfig
from vllm.config.scheduler import SchedulerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scheduler(**kwargs) -> SchedulerConfig:
    """Construct a minimal SchedulerConfig for validation testing."""
    defaults = dict(max_model_len=4096, is_encoder_decoder=False)
    defaults.update(kwargs)
    return SchedulerConfig(**defaults)


# ---------------------------------------------------------------------------
# ModelConfig.max_logprobs
# ---------------------------------------------------------------------------

class TestMaxLogprobs:
    def test_positive_accepted(self):
        mc = ModelConfig(model="facebook/opt-125m", max_logprobs=5)
        assert mc.max_logprobs == 5

    def test_zero_accepted(self):
        mc = ModelConfig(model="facebook/opt-125m", max_logprobs=0)
        assert mc.max_logprobs == 0

    def test_minus_one_accepted(self):
        # -1 is the sentinel meaning "derive from vocab size"
        mc = ModelConfig(model="facebook/opt-125m", max_logprobs=-1)
        assert mc.max_logprobs == -1

    def test_default_accepted(self):
        mc = ModelConfig(model="facebook/opt-125m")
        assert mc.max_logprobs == 20

    @pytest.mark.parametrize("bad", [-2, -5, -100])
    def test_negative_rejected(self, bad):
        with pytest.raises(ValidationError, match="max_logprobs"):
            ModelConfig(model="facebook/opt-125m", max_logprobs=bad)


# ---------------------------------------------------------------------------
# SchedulerConfig.long_prefill_token_threshold
# ---------------------------------------------------------------------------

class TestLongPrefillTokenThreshold:
    def test_zero_accepted(self):
        sc = make_scheduler(long_prefill_token_threshold=0)
        assert sc.long_prefill_token_threshold == 0

    def test_positive_accepted(self):
        sc = make_scheduler(long_prefill_token_threshold=512)
        assert sc.long_prefill_token_threshold == 512

    def test_default_accepted(self):
        sc = make_scheduler()
        # default is 0 (off)
        assert sc.long_prefill_token_threshold == 0

    @pytest.mark.parametrize("bad", [-1, -5, -1000])
    def test_negative_rejected(self, bad):
        with pytest.raises(ValidationError, match="long_prefill_token_threshold"):
            make_scheduler(long_prefill_token_threshold=bad)
