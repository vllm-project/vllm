# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"


def do_sample(
    llm: vllm.LLM, lora_path: str, lora_id: int, prompts: list[str]
) -> list[str]:
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("tp_size", [2])
def test_nemotron_fp8_lora(nemotron_lora_files, tp_size, monkeypatch):
    """Test LoRA with Nemotron Nano FP8 MoE model."""
    if (
        torch.cuda.device_count() < tp_size
        and tp_size > 1
        and current_platform.is_cuda_alike()
    ):
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")

    prompts_and_answers = [
        ("What is the capital of France?", "Paris"),
    ]

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=2,
        max_lora_rank=64,
        max_model_len=1024,
        quantization="modelopt",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )

    output = do_sample(
        llm, nemotron_lora_files, lora_id=1, prompts=[p[0] for p in prompts_and_answers]
    )

    # Check outputs are coherent
    for i, (_, answer) in enumerate(prompts_and_answers):
        assert answer in output[i], f"Expected '{answer}' in output: {output[i]}"
