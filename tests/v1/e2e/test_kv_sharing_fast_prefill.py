# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.platforms import current_platform

from ...utils import check_answers, fork_new_process_for_each_test, prep_prompts

# global seed
SEED = 42


@pytest.fixture
def test_prompts():
    """
    Adapted from tests/v1/e2e/test_spec_decode.py
    """
    prompt_types = ["repeat", "sentence"]
    # Setting higher num prompts increases the chance of numerics mismatch
    # due to matrix multiplication numerics depending on batch dimension
    num_prompts = 10
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""please repeat the word '{word}' 10 times."""
        elif kind == "sentence":
            prompt = f"""please give a ten-word sentence that
            uses the word {word} at least once."""
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append(prompt)

    return prompts


use_fork_for_test = (
    fork_new_process_for_each_test if not current_platform.is_rocm() else lambda x: x
)


@use_fork_for_test
@pytest.mark.parametrize("kv_sharing_fast_prefill", [False, True])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_kv_sharing_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    kv_sharing_fast_prefill: bool,
    enforce_eager: bool,
):
    if not enforce_eager and current_platform.is_rocm():
        # Relevant context: https://github.com/vllm-project/vllm/pull/29244
        pytest.skip(
            "ROCm: torch.compile produces incorrect output for gemma-3n's GELU "
            "with tanh approximation. Use enforce_eager=True instead."
        )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        # This allows vLLM compilation backend to handle allocating and
        # managing buffers for cudagraph
        cudagraph_copy_inputs=True,
        mode=CompilationMode.VLLM_COMPILE
        if not enforce_eager
        else CompilationMode.NONE,
    )
    batch_size = 10

    with monkeypatch.context() as m:
        # Make scheduling deterministic for reproducibility
        if current_platform.is_rocm():
            # Use spawn to prevent cuda re-initialization error
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        else:
            m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        prompts, answer, indices = prep_prompts(batch_size)

        llm = LLM(
            model="google/gemma-3n-E2B-it",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            seed=SEED,
            kv_sharing_fast_prefill=kv_sharing_fast_prefill,
        )
        responses = llm.generate(prompts, sampling_params)
        check_answers(
            indices,
            answer,
            [response.outputs[0].text for response in responses],
            accept_rate=1.0,
        )
