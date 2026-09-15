# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    HYBRID_MODELS,
    MAX_NUM_SEQS,
    SSM_MODELS,
)

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_fail_upon_inc_requests_and_finished_requests_lt_available_blocks(
    vllm_runner,
    example_prompts,
    model: str,
) -> None:
    """
    This test is for verifying that the hybrid inner state management doesn't
    collapse in case where the number of incoming requests and
    finished_requests_ids is larger than the maximum mamba block capacity.

    This could generally happen due to the fact that hybrid does support
    statelessness mechanism where it can clean up new incoming requests in
    a single step.
    """
    try:
        with vllm_runner(
            model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
        ) as vllm_model:
            vllm_model.generate_greedy([example_prompts[0]] * 100, 10)
    except ValueError:
        pytest.fail(
            "Hybrid inner state wasn't cleaned up properly between"
            "steps finished requests registered unnecessarily "
        )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_state_cleanup(
    vllm_runner,
    example_prompts,
    model: str,
) -> None:
    """
    This test is for verifying that the Hybrid state is cleaned up between
    steps.

    If it's not cleaned, an error would be expected.
    """
    try:
        with vllm_runner(
            model, max_num_seqs=MAX_NUM_SEQS, enable_chunked_prefill=True
        ) as vllm_model:
            for _ in range(10):
                vllm_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail(
            "Hybrid inner state wasn't cleaned up between states, "
            "could be related to finished_requests_ids"
        )
