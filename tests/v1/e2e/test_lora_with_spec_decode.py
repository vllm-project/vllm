# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script contains:
1. test lora with speculative decoding for batch inference
"""

import random

import numpy as np
import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

LORA_TEST_PROMPT_MAP: dict[str, str] = {}

LORA_TEST_PROMPT_MAP["premjatin/qwen-linear-algebra-coder"] = """
### INSTRUCTION:
You are an AI assistant that generates Python code to solve linear
algebra problems.

### PROBLEM:
Find the eigenvalues and eigenvectors of the following 3x3 matrix:
[[3, 2, 0],
 [2, 3, 0],
 [0, 0, 2]]

### OUTPUT FORMAT (STRICT):
Numbers should be represented as integers only.

### PYTHON SOLUTION:
"""

SEED = 42


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize(
    "model_setup",
    [
        (
            "eagle3",
            "Qwen/Qwen3-1.7B",
            "AngelSlim/Qwen3-1.7B_eagle3",
            "premjatin/qwen-linear-algebra-coder",
            1,
        )
    ],
)
def test_batch_inference_correctness(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str, str, int],
):
    """
    Compare the outputs of a LLM with only Lora and a LLM with both SD and Lora.
    Should be the same and no failure when doing batch inference.
    model_setup: (method, model_name, spec_model_name, lora_path, tp_size)
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Disable randomness
        m.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        method, model_name, spec_model_name, lora_path, tp_size = model_setup

        # without speculative decoding
        ref_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            max_model_len=2048,
            max_num_seqs=4,
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=16,
        )

        prompts = [LORA_TEST_PROMPT_MAP[lora_path]] * 100
        lora_request = LoRARequest("adapter", 1, lora_path)
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, seed=SEED, max_tokens=128
        )

        ref_outputs = ref_llm.generate(
            prompts, sampling_params, lora_request=lora_request
        )
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        lora_spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 3,
                "max_model_len": 2048,
            },
            max_model_len=2048,
            max_num_seqs=4,
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=16,
        )

        lora_spec_outputs = lora_spec_llm.generate(
            prompts, sampling_params, lora_request=lora_request
        )

        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, lora_spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 90% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        print(f"match ratio: {matches}/{len(ref_outputs)}")
        assert matches > int(0.90 * len(ref_outputs))
        del lora_spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
