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
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.mark.parametrize(
    "model_setup",
    [
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 4),
    ],
    ids=["llama4"],
)
def test_eplb_model(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, int],
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_MLA_DISABLE", "1")

        model_name, tp_size = model_setup
        test_prompts = create_test_prompts()
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            max_model_len=2048,
            enable_expert_parallel=True,
            num_redundant_experts=tp_size,
            eplb_window_size=4,
            eplb_step_interval=16,
            eplb_log_balancedness=True,
            enable_eplb=True,
            load_format="dummy",
            gpu_memory_utilization=0.95,
        )
        test_prompts = create_test_prompts()
        llm.generate(test_prompts, sampling_config)
        del llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "model_setup",
    [
        (
            "eagle",
            "eagle618/deepseek-v3-random",
            "eagle618/eagle-deepseek-v3-random",
            4,
        ),
        ("deepseek_mtp", "eagle618/deepseek-v3-random", None, 4),
        ("qwen3_next_mtp", "Qwen/Qwen3-Next-80B-A3B-Instruct", None, 4),
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
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 1,
                "max_model_len": 2048,
            },
            max_model_len=2048,
            enable_expert_parallel=True,
            num_redundant_experts=tp_size,
            eplb_window_size=1000,
            eplb_step_interval=3000,
            eplb_log_balancedness=True,
            enable_eplb=True,
            load_format="dummy",
        )
        test_prompts = create_test_prompts()
        llm.generate(test_prompts, sampling_config)
        del llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
