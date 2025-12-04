# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest
import torch

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EPLBConfig


@pytest.mark.parametrize(
    "model_setup",
    [
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", 4),
    ],
)
def test_eplb_model(
    model_setup: tuple[str, int],
):
    model_name, tp_size = model_setup
    test_prompt = ["This is a prompt which has more than 10 tokens."]

    llm_args = dict(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_model_len=2048,
        enable_expert_parallel=True,
        enable_eplb=True,
        load_format="dummy",
        gpu_memory_utilization=0.95,
    )

    # Save EPLB statistics to disk
    eplb_config_save = EPLBConfig(
        window_size=8, step_interval=10, save_load_window=True, save_dir="/tmp"
    )
    llm = LLM(eplb_config=eplb_config_save, **llm_args)
    llm.generate(test_prompt)
    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Load EPLB statistics from disk
    eplb_config_load = EPLBConfig(
        load_initial_load_window=True,
        load_path="/tmp/global_expert_load_window_i0.safetensors",
        use_async=True,
    )
    llm = LLM(eplb_config=eplb_config_load, **llm_args)
    llm.generate(test_prompt)
    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()
