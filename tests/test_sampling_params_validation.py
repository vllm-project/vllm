# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``SamplingParams`` argument validation.

These tests cover pure-Python argument validation in
``vllm.sampling_params.SamplingParams._verify_args`` and surrounding
``__post_init__`` logic. They do not require a GPU or model.
"""

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


class TestTemperatureValidation:
    """temperature must lie in [0, 2] per the OpenAI spec.

    The Rust frontend enforces the same range via
    ``#[validate(range(min = 0.0, max = 2.0))]``; this test pins the
    Python frontend to the same contract so the two frontends agree.
    """

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
    def test_temperature_in_range_is_accepted(self, temperature: float) -> None:
        params = SamplingParams(temperature=temperature)
        assert params.temperature == temperature

    @pytest.mark.parametrize("temperature", [2.001, 3.0, 100.0, 1e9])
    def test_temperature_above_max_is_rejected(self, temperature: float) -> None:
        with pytest.raises(VLLMValidationError, match="temperature must be in"):
            SamplingParams(temperature=temperature)

    @pytest.mark.parametrize("temperature", [-0.1, -1.0, -100.0])
    def test_temperature_below_zero_is_rejected(self, temperature: float) -> None:
        with pytest.raises(
            VLLMValidationError, match="temperature must be non-negative"
        ):
            SamplingParams(temperature=temperature)
