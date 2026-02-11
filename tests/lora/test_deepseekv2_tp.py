# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE To avoid overloading the CI pipeline, this test script will
# not be triggered on CI and is primarily intended for local testing
# and verification.

import vllm
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"

PROMPT_TEMPLATE = "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\nUser: {context}\n\nAssistant:"  # noqa: E501


def generate_and_test(llm: vllm.LLM, lora_path: str, lora_id: int):
    prompts = [
        PROMPT_TEMPLATE.format(context="Who are you?"),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # return generated_texts
    expected_lora_output = [
        "I am \u5f20\u5b50\u8c6a, an AI assistant developed by \u9648\u58eb\u680b.",  # noqa: E501
    ]
    for i in range(len(expected_lora_output)):
        assert generated_texts[i].startswith(expected_lora_output[i])


def test_deepseekv2_lora(deepseekv2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )
    generate_and_test(llm, deepseekv2_lora_files, 1)


def test_deepseekv2(deepseekv2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
    )
    generate_and_test(llm, deepseekv2_lora_files, 1)


@multi_gpu_test(num_gpus=2)
def test_deepseekv2_tp2(deepseekv2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
    )
    generate_and_test(llm, deepseekv2_lora_files, 2)


@multi_gpu_test(num_gpus=4)
def test_deepseekv2_tp4(deepseekv2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=4,
    )
    generate_and_test(llm, deepseekv2_lora_files, 2)
