# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EPLBConfig


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.mark.parametrize(
    "model_setup",
    [
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", 4),
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

        model_name, tp_size = model_setup
        test_prompts = ["This is a prompt which has more than 10 tokens."]

        llm_args = dict(
            model=model_name,
            tensor_parallel_size=tp_size,
            max_model_len=2048,
            enable_expert_parallel=True,
            num_redundant_experts=tp_size,
            eplb_window_size=8,
            eplb_step_interval=10,
            eplb_log_balancedness=True,
            enable_eplb=True,
            load_format="dummy",
            gpu_memory_utilization=0.95,
        )

        # Save EPLB statistics to disk
        eplb_config_save = EPLBConfig(save_dir="/tmp")
        llm = LLM(eplb_config=eplb_config_save, **llm_args)
        llm.generate(test_prompts, sampling_config)
        del llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        # Load EPLB statistics from disk
        eplb_config_load = EPLBConfig(
            load_path="/tmp/global_expert_load_window_i0.safetensors"
        )
        llm = LLM(eplb_config=eplb_config_load, **llm_args)
        llm.generate(test_prompts, sampling_config)
        del llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
