# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    if os.path.isdir("local_storage"):
        shutil.rmtree("local_storage")


def test_integration():

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig.from_cli(
            '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both", '
            '"kv_connector_extra_config": '
            '{"shared_storage_path": "local_storage"}}'))

    context = "Hi " * 1000
    context2 = "Hey " * 500
    prompts = [
        context + "Hello, my name is",
        context + "The capital of France is",
        context2 + "Your name is",
        context2 + "The capital of China is",
    ]

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
        print(f"Saved {len(new_prompts)} prompts to output.txt")

    del llm

    # Read prompts from output.txt
    prompts = []
    try:
        with open("output.txt") as f:
            for line in f:
                prompts.append(line.strip())
        print(f"Loaded {len(prompts)} prompts from output.txt")
    except FileNotFoundError:
        print("Error: output.txt file not found")
        exit(-1)

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    decode_llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig.from_cli(
            '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both",'
            '"kv_connector_extra_config": {"shared_storage_path": "local_storage"}}'  # noqa: E501
        ))

    # 2nd generation (decode instance)
    outputs = decode_llm.generate(prompts, sampling_params)

    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert len(generated_text) > 5
