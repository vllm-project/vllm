# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script contains:
1. test lora with speculative decoding for batch inference
"""

import pytest
import torch

from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from ..utils import assert_request_outputs_match

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
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)
def test_batch_inference_correctness(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str, str, int],
    vllm_runner,
):
    """
    Compare the outputs of a LLM with only Lora and a LLM with both SD and Lora.
    Should be the same and no failure when doing batch inference.
    model_setup: (method, model_name, spec_model_name, lora_path, tp_size)
    """
    with monkeypatch.context() as m:
        # Disable randomness
        if current_platform.is_cuda():
            m.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        m.setenv("VLLM_BATCH_INVARIANT", "1")
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        set_random_seed(SEED)
        m.setattr(torch.backends.cudnn, "benchmark", False)
        m.setattr(torch.backends.cudnn, "deterministic", True)

        method, model_name, spec_model_name, lora_path, tp_size = model_setup

        prompts = [LORA_TEST_PROMPT_MAP[lora_path]] * 100
        lora_request = LoRARequest("adapter", 1, lora_path)
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, seed=SEED, max_tokens=128
        )

        # without speculative decoding
        with vllm_runner(
            model_name,
            block_size=None,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            max_model_len=2048,
            max_num_seqs=4,
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=16,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
        ) as ref_runner:
            ref_outputs = ref_runner.llm.generate(
                prompts, sampling_params, lora_request=lora_request
            )

        with vllm_runner(
            model_name,
            block_size=None,
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
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
        ) as spec_runner:
            lora_spec_outputs = spec_runner.llm.generate(
                prompts, sampling_params, lora_request=lora_request
            )

        # Under greedy verification, the spec-decode output should equal the
        # non-spec output (modulo FP noise from the target's verify-path matmul
        # running at seqlen num_speculative_tokens+1 vs 1). 90% leaves slack.
        assert_request_outputs_match(
            ref_outputs,
            lora_spec_outputs,
            required_matches=int(0.90 * len(ref_outputs)) + 1,
            context=(
                f"LoRA + {method}: target={model_name}, draft={spec_model_name}, "
                f"adapter={lora_path}"
            ),
        )
