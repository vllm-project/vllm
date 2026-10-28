# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import SpeculativeConfig


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asd_risk_budget", -1.0),
        ("asd_risk_budget", float("nan")),
        ("asd_risk_budget", float("inf")),
        ("asd_max_regret_per_value", -0.1),
        ("asd_max_regret_per_value", float("inf")),
        ("asd_max_relaxed_tokens_per_block", -1),
    ],
)
def test_asd_controls_reject_invalid_bounds(field, value):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=4,
            rejection_sample_method="approximate",
            **{field: value},
        )
