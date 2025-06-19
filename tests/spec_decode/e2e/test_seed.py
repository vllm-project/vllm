# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .conftest import run_equality_correctness_test

# main model
MAIN_MODEL = "JackFram/llama-68m"

# speculative model
SPEC_MODEL = "JackFram/llama-160m"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # speculative config
        "speculative_config": {
            "model": "JackFram/llama-160m",
            "num_speculative_tokens": 3,
        },
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{"seed": 1}])
@pytest.mark.parametrize("test_llm_kwargs", [{"seed": 5}])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("temperature", [0.1, 1.0])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        20,
    ])
def test_seeded_consistency(vllm_runner, common_llm_kwargs,
                            per_test_common_llm_kwargs, baseline_llm_kwargs,
                            test_llm_kwargs, batch_size: int,
                            temperature: float, output_len: int):
    """Verify outputs are consistent across multiple runs with same seed
    """
    run_equality_correctness_test(
        vllm_runner,
        common_llm_kwargs,
        per_test_common_llm_kwargs,
        baseline_llm_kwargs,
        test_llm_kwargs,
        batch_size,
        max_output_len=output_len,
        temperature=temperature,
        disable_seed=False,
    )

    # Ensure this same test does fail if we _don't_ include per-request seeds
    with pytest.raises(AssertionError):
        run_equality_correctness_test(
            vllm_runner,
            common_llm_kwargs,
            per_test_common_llm_kwargs,
            baseline_llm_kwargs,
            test_llm_kwargs,
            batch_size,
            max_output_len=output_len,
            temperature=temperature,
            disable_seed=True,
        )
