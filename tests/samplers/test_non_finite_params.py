# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that non-finite float values (NaN, Inf) are rejected by
SamplingParams validation, preventing them from propagating to GPU kernels.

Addresses advisory GHSA-7h4p-rffg-7823.
"""

import math

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


class TestNonFiniteTemperature:
    """Verify that NaN and Infinity temperature values are rejected."""

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), math.nan, math.inf],
        ids=["nan", "inf", "-inf", "math.nan", "math.inf"],
    )
    def test_non_finite_temperature_rejected(self, value: float):
        with pytest.raises(VLLMValidationError, match="temperature"):
            SamplingParams(temperature=value)

    def test_finite_temperature_accepted(self):
        SamplingParams(temperature=0.0)
        SamplingParams(temperature=0.5)
        SamplingParams(temperature=1.0)
        SamplingParams(temperature=2.0)


class TestNonFiniteRepetitionPenalty:
    """Verify that NaN and Infinity repetition_penalty values are rejected."""

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), math.nan, math.inf],
        ids=["nan", "inf", "-inf", "math.nan", "math.inf"],
    )
    def test_non_finite_repetition_penalty_rejected(self, value: float):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=value)

    def test_finite_repetition_penalty_accepted(self):
        SamplingParams(repetition_penalty=0.5)
        SamplingParams(repetition_penalty=1.0)
        SamplingParams(repetition_penalty=2.0)
