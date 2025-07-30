# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


@pytest.fixture
def env_setup():
    """Set up required environment variables"""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.fixture
def input_prompts():
    """Create test prompts"""
    context = "Hi " * 10  # Reduced size for testing
    context2 = "Hey " * 10
    context3 = "Hello " * 10
    context4 = "How " * 10
    return [
        context + "Hello, my name is",
        context2 + "The capital of France is",
        context3 + "Your name is",
        context4 + "The capital of China is",
    ]


@pytest.fixture
def llm_instance():
    """Create LLM instance with test configuration"""
    return LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="CPUConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={},
        ),
        load_format="dummy",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        block_size=64,
    )


def test_llm_generation(env_setup, input_prompts, llm_instance, tmp_path):
    """Test LLM generation and output saving"""
    # Configure sampling parameters
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Generate outputs
    outputs = llm_instance.generate(input_prompts, sampling_params)

    # Verify outputs
    assert len(outputs) == len(
        input_prompts), "Number of outputs should match number of prompts"

    # Process outputs
    new_prompts = []
    for output in outputs:
        assert hasattr(output, 'prompt'), "Output should have prompt attribute"
        assert hasattr(output,
                       'outputs'), "Output should have outputs attribute"
        assert len(output.outputs) > 0, "Output should have generated text"

        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)

    # Test file writing
    output_file = tmp_path / "output.txt"
    with open(output_file, "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")

    # Verify file contents
    assert output_file.exists(), "Output file should be created"
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == len(
            input_prompts), "File should contain all prompts"
        for line in lines:
            assert line.strip(), "Lines should not be empty"
