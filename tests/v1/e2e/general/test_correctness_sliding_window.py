# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from ....utils import check_answers, prep_prompts


@dataclass
class TestConfig:
    sliding_window: int
    ln_range: tuple[int, int]


model_config = {
    "bigcode/starcoder2-3b": TestConfig(4096, (800, 1100)),
    "google/gemma-3-1b-it": TestConfig(4096, (400, 800)),
}


@pytest.mark.parametrize(
    "model",
    [
        "bigcode/starcoder2-3b",  # sliding window only
        "google/gemma-3-1b-it",  # sliding window + full attention
    ],
)
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("disable_hybrid_kv_cache_manager", [True, False])
def test_sliding_window_retrieval(
    model, batch_size, seed, disable_hybrid_kv_cache_manager, vllm_runner
):
    """
    The test does a bunch of assignments "x1 = 10\nx2 = 33\n..." and then
    asks for value of one of them (which is outside the sliding window).
    If we tell it upfront which we are going to be looking for, then
    it answers correctly (mostly).
    """
    # NOTE: For ROCm, we have to enforce eager mode to use custom kernel
    # implementation of GELU with tanh approximation, as PyTorch's native
    # implementation is currently unstable with torch.compile and produces garbage.
    enforce_eager = current_platform.is_rocm()

    test_config = model_config[model]

    with vllm_runner(
        model,
        max_model_len=None,
        enable_chunked_prefill=None,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
        enforce_eager=enforce_eager,
    ) as runner:
        llm = runner.get_llm()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

        prompts, answer, indices = prep_prompts(
            batch_size, ln_range=test_config.ln_range
        )

        check_length(prompts, llm, test_config.sliding_window)

        # Fresh generation
        responses = llm.generate(prompts, sampling_params)
        check_answers(
            indices,
            answer,
            [response.outputs[0].text for response in responses],
            accept_rate=1.0,
        )

        # Re-generate with the same prompts to test prefix caching
        responses = llm.generate(prompts, sampling_params)
        check_answers(
            indices,
            answer,
            [response.outputs[0].text for response in responses],
            accept_rate=1.0,
        )


@pytest.mark.parametrize("model", ["google/gemma-3-1b-it"])
def test_hybrid_kv_cache_manager_output_equivalence(model, vllm_runner):
    """Disabling the hybrid KV cache manager must not change what the model
    computes: it promotes the sliding-window layers to full-attention *storage*
    only, so the global layers must keep attending globally.

    Guards against the group's KV cache spec, which then covers windowed and
    global layers alike, imposing its window on the global layers too.
    """
    prompts, _, _ = prep_prompts(2, ln_range=model_config[model].ln_range)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    enforce_eager = current_platform.is_rocm()

    outputs = []
    for disable_hybrid in (False, True):
        with vllm_runner(
            model,
            max_model_len=None,
            enable_chunked_prefill=None,
            disable_hybrid_kv_cache_manager=disable_hybrid,
            enforce_eager=enforce_eager,
        ) as runner:
            responses = runner.get_llm().generate(prompts, sampling_params)
            outputs.append([response.outputs[0].text for response in responses])

    assert outputs[0] == outputs[1]


def check_length(prompts: list[str], llm: LLM, sliding_window: int):
    """
    Check if the prompt length is valid, i.e., longer than the sliding window
    size and shorter than the model's max length.

    Args:
        prompts: list of prompts
        llm: LLM object
        sliding_window: Sliding window size
    """
    tokenizer = llm.get_tokenizer()
    max_model_len = llm.llm_engine.model_config.max_model_len
    assert any(len(tokenizer.encode(prompt)) > sliding_window for prompt in prompts), (
        "Prompt is too short for test"
    )
    assert all(len(tokenizer.encode(prompt)) <= max_model_len for prompt in prompts), (
        "Prompt is too long for test"
    )
