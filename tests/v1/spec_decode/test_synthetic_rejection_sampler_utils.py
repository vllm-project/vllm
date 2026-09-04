# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates


def test_unconditional_to_conditional_rates_basic():
    # c_0 = p_0; c_i = p_i / p_{i-1}
    assert unconditional_to_conditional_rates([0.9, 0.5, 0.2]) == pytest.approx(
        [0.9, 0.5 / 0.9, 0.2 / 0.5]
    )


def test_unconditional_to_conditional_rates_handles_zero():
    # After a zero, subsequent conditional rates are clamped to 0 (the chain
    # has already terminated in the kernel, so these values are unused).
    assert unconditional_to_conditional_rates([1.0, 0.6, 0.0, 0.0]) == pytest.approx(
        [1.0, 0.6, 0.0, 0.0]
    )


def test_unconditional_to_conditional_rates_all_ones():
    assert unconditional_to_conditional_rates([1.0, 1.0, 1.0]) == pytest.approx(
        [1.0, 1.0, 1.0]
    )


@pytest.mark.parametrize(
    "length,n,expected",
    [
        (2.6, 3, [1.0, 0.6, 0.0]),
        (1.0, 3, [0.0, 0.0, 0.0]),
        (4.0, 3, [1.0, 1.0, 1.0]),
        (2.0, 3, [1.0, 0.0, 0.0]),
        (3.5, 4, [1.0, 1.0, 0.5, 0.0]),
    ],
)
def test_acceptance_length_to_rates(length, n, expected):
    assert SpeculativeConfig._acceptance_length_to_rates(length, n) == pytest.approx(
        expected
    )


def test_resolve_length_produces_minvariance_schedule():
    assert SpeculativeConfig._resolve_synthetic_acceptance_rates(
        3, None, 2.6
    ) == pytest.approx([1.0, 0.6, 0.0])
