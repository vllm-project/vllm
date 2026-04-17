# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.worker.gpu.spec_decode.synthetic_rejection_sampler_utils import (
    compute_synthetic_rejection_sampler_params,
)

NUM_SPECULATIVE_STEPS = [1, 2, 3, 4, 5, 7, 10]
ACCEPTANCE_RATES = [i / 100 for i in range(0, 100)]


@pytest.mark.parametrize("num_speculative_steps", NUM_SPECULATIVE_STEPS)
def test_compute_synthetic_rejection_sampler_params(num_speculative_steps: int):
    """Test that the base acceptance rate and decay factor generated for
    synthetic rejection sampling have a mean joint acceptance probability
    that matches the desired acceptance rate."""
    tol = 1e-9
    for desired_acceptance_rate in ACCEPTANCE_RATES:
        base_rate, decay_factor = compute_synthetic_rejection_sampler_params(
            desired_acceptance_rate, num_speculative_steps, tol=tol
        )

        # Compute the mean of joint acceptance probabilities across
        # all speculative positions.
        joint_prob = 1.0
        mean_joint = 0.0
        for i in range(num_speculative_steps):
            joint_prob *= base_rate * decay_factor**i
            mean_joint += joint_prob
        mean_joint /= num_speculative_steps

        assert abs(desired_acceptance_rate - mean_joint) < 10 * tol
        assert base_rate <= 1.0
