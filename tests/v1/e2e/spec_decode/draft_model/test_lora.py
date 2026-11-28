# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This script contains:
1. test lora with speculative decoding for batch inference
"""

import pytest
import torch

import vllm.envs as envs
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from ..utils import assert_request_outputs_match

LORA_TEST_PROMPT_MAP: dict[str, str] = {}

LORA_TEST_PROMPT_MAP["vllm/qwen-linear-algebra-coder"] = """
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

LORA_TEST_PROMPT_MAP["vllm-project/qwen35-4b-text-only-sql-lora"] = (
    "Write a SQL query for the given database.\n"
    "Schema:\n"
    "Tables:\n"
    "  - stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)\n"
    "  - singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age,"
    " Is_male)\n"
    "  - concert(concert_ID, Concert_Name, Theme, Stadium_ID, Year)\n"
    "  - singer_in_concert(concert_ID, Singer_ID)\n\n"
    "Question:\n"
    "What is the average, minimum, and maximum age of all singers from France?"
)

SEED = 42


@pytest.mark.parametrize(
    "model_setup",
    [
        (
            "eagle3",
            "Qwen/Qwen3-1.7B",
            "AngelSlim/Qwen3-1.7B_eagle3",
            "vllm/qwen-linear-algebra-coder",
            1,
        ),
        # Native MTP: the drafter is the target's own MTP module. The adapter
        # has no MTP weights, so only the target runs with LoRA.
        (
            "mtp",
            "Qwen/Qwen3.5-4B",
            None,
            "vllm-project/qwen35-4b-text-only-sql-lora",
            1,
        ),
    ],
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)
def test_batch_inference_correctness(
    monkeypatch: pytest.MonkeyPatch,
    model_setup: tuple[str, str, str | None, str, int],
    vllm_runner,
):
    """Compare the outputs of a LLM with only Lora and a LLM with both SD and Lora.
    Should be the same and no failure when doing batch inference.
    model_setup: (method, model_name, spec_model_name, lora_path, tp_size)
    """
    method, model_name, spec_model_name, lora_path, tp_size = model_setup
    with monkeypatch.context() as m:
        # Batch-invariant mode does not support the GDN (linear attention)
        # layers of Qwen3.5, so that case relies on the match threshold below.
        if "Qwen3.5" not in model_name:
            m.setenv("VLLM_BATCH_INVARIANT", "1")
            m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        set_random_seed(SEED)
        m.setattr(torch.backends.cudnn, "benchmark", False)
        m.setattr(torch.backends.cudnn, "deterministic", True)

        prompts = [LORA_TEST_PROMPT_MAP[lora_path]] * 100
        # Qwen3.5 is a VLM; skip the vision encoder for this text-only test.
        extra_kwargs: dict = {}
        if "Qwen3.5" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        speculative_config = {
            "method": method,
            "num_speculative_tokens": 3,
            "max_model_len": 2048,
        }
        if spec_model_name is not None:
            speculative_config["model"] = spec_model_name
        lora_request = LoRARequest("adapter", 1, lora_path)
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, seed=SEED, max_tokens=128
        )

        # without speculative decoding
        with vllm_runner(
            model_name,
            block_size=None,
            tensor_parallel_size=tp_size,
            max_model_len=2048,
            max_num_seqs=4,
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=16,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        ) as ref_runner:
            ref_outputs = ref_runner.llm.generate(
                prompts, sampling_params, lora_request=lora_request
            )

        with vllm_runner(
            model_name,
            block_size=None,
            tensor_parallel_size=tp_size,
            speculative_config=speculative_config,
            max_model_len=2048,
            max_num_seqs=4,
            enable_lora=True,
            max_loras=1,
            max_cpu_loras=1,
            max_lora_rank=16,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
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
