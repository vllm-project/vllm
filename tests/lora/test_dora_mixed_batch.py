# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.lora.utils import convert_dora_checkpoint_to_lora
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DORA_ADAPTER_NAME = "qwen25-dora"
LORA_ADAPTER_NAME = "qwen25-lora-from-dora"


@pytest.fixture(scope="module")
def qwen25_05b_lora_files(tmp_path_factory, qwen25_05b_dora_files):
    lora_dir = tmp_path_factory.mktemp("qwen25_05b_lora_from_dora")
    return convert_dora_checkpoint_to_lora(qwen25_05b_dora_files, lora_dir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DoRA forward support is CUDA-only.",
)
def test_mixed_batch_base_lora_and_dora_requests(
    qwen25_05b_dora_files,
    qwen25_05b_lora_files,
):
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2,
        max_cpu_loras=2,
        max_num_seqs=8,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )
    prompts = [
        "The capital of France is",
        "A doctor can help with",
        "The capital of France is",
    ]
    lora_requests = [
        None,
        LoRARequest(LORA_ADAPTER_NAME, 1, qwen25_05b_lora_files),
        LoRARequest(DORA_ADAPTER_NAME, 2, qwen25_05b_dora_files),
    ]

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=4),
        lora_request=lora_requests,
        use_tqdm=False,
    )

    assert [output.lora_request for output in outputs] == lora_requests
    assert all(output.outputs for output in outputs)
