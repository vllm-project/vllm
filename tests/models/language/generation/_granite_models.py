# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, non-collected helper for the granite model test slices.

This module holds the body of the original ``test_granite.py::test_models``
and the per-model minimum transformers versions. The leading underscore
keeps it out of pytest's ``test_*.py`` collection;
``tests/tools/test_language_layout.py`` proves it is never collected on
its own.
"""

from ...utils import check_logprobs_close, check_transformers_version

# model -> minimum transformers version, or None if unconstrained
MODELS = {
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b": None,
    "ibm/PowerMoE-3b": None,
    "ibm-granite/granite-swash-2b": "5.15.1",
    "ibm-granite/granite-swash-3b-a600m": "5.15.1",
}


def _test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    check_transformers_version(model, min_transformers_version=MODELS[model])

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
