# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared BF16 baselines for the quantization comparison tests.

Several tests here quantize the same model and compare against an
unquantized run of it. Computing that reference once per session avoids
booting an identical engine for every quantization scheme under test.

Only the outputs are cached; each engine is still torn down as usual, so
tests keep the isolation ``VllmRunner.__exit__`` provides.
"""

import pytest


@pytest.fixture(scope="session")
def quant_baseline_logprobs(vllm_runner):
    """Greedy logprobs for an unquantized model, computed once per config."""
    cache: dict = {}

    def baseline(
        model: str,
        prompts: list[str],
        *,
        max_model_len: int,
        max_tokens: int,
        num_logprobs: int,
    ):
        key = (model, max_model_len, max_tokens, num_logprobs, tuple(prompts))
        if key not in cache:
            with vllm_runner(
                model,
                max_model_len=max_model_len,
                enforce_eager=True,
            ) as vllm_model:
                cache[key] = vllm_model.generate_greedy_logprobs(
                    prompts, max_tokens, num_logprobs
                )
        return cache[key]

    return baseline
