# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def create_test_prompts() -> list[str]:
    return [
        "A robot may not injure a human being",
        "To be or not to be,",
        "What is the meaning of life?",
    ]


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=64, ignore_eos=False)


def check_outputs(ref_outputs, spec_outputs):
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))


@pytest.mark.parametrize(
    "model_setup",
    [
        (
            "eagle",
            "eagle618/deepseek-v3-random",
            "eagle618/eagle-deepseek-v3-random",
            4,
        ),
        pytest.param(
            "deepseek_mtp",
            "eagle618/deepseek-v3-random",
            None,
            2,
            marks=pytest.mark.skip(reason="Skipping for CI test time savings"),
        ),
        pytest.param(
            "qwen3_next_mtp",
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            None,
            2,
            marks=pytest.mark.skip(reason="Skipping for CI test time savings"),
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues"),
        ),
    ],
    ids=["deepseek_eagle", "deepseek_mtp", "qwen3_next_mtp", "llama4_eagle"],
)
def test_eplb_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_MLA_DISABLE", "1")

        method, model_name, spec_model_name, tp_size = model_setup
        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 4,
                "max_model_len": 2048,
            },
            max_model_len=2048,
            enable_expert_parallel=True,
            num_redundant_experts=tp_size,
            eplb_window_size=8,
            eplb_step_interval=32,
            eplb_log_balancedness=True,
            enable_eplb=True,
        )
        test_prompts = create_test_prompts()
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        ref_llm = LLM(
            model=model_name, max_model_len=2048, tensor_parallel_size=tp_size
        )
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        check_outputs(ref_outputs, spec_outputs)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
