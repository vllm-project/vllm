# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether vllm correctly load and run gptq_v2 format checkpoints.

Run `pytest tests/quantization/test_gptq_v2.py --forked`.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod

# Asymmetrically quantized models, stored in GPTQ v1/v2 format
# MODELS = [(model_id, checkpoint_format, enable_thinking)]
MODELS = [
    ('BitDistiller/Qwen-8B-w2g64-gptq', 'gptq_v2', True),
    ('BitDistiller/Llama-3.1-8B-Instruct-w2g64-gptq', 'gptq_v2', False),
]


@pytest.mark.parametrize("model_id, checkpoint_format, enable_thinking",
                         MODELS)
def test_model_load(vllm_runner, model_id, checkpoint_format, enable_thinking,
                    monkeypatch):

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Only check the default GPTQ linear method.
    # TODO: add test for other kernels, e.g., Marlin.
    linear_method_cls = GPTQLinearMethod

    with vllm_runner(model_id, dtype=torch.float16, max_model_len=512) as llm:

        def check_model(model_id):
            for name, submodule in model_id.named_modules():
                # Could check more modules if necessary
                if name == 'model_id.layers.0.self_attn.qkv_proj':
                    assert isinstance(submodule.quant_method,
                                      linear_method_cls)

                    config = submodule.quant_method.quant_config
                    assert config.checkpoint_format == checkpoint_format
                    if checkpoint_format == 'gptq_v2':
                        assert submodule.quant_method.use_gptq_gemm_v2

                    # Just break since currently we only check 1 module
                    break

        # Check if gptq_v2 format is correctly loaded
        llm.apply_model(check_model)


@pytest.mark.parametrize("model_id, checkpoint_format, enable_thinking",
                         MODELS)
def test_model_inference(vllm_runner, model_id, checkpoint_format,
                         enable_thinking):

    # Prepare prompt to test the model_id's generation result.
    prompt = "What is the meaning of life?"
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": prompt
    }]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=
        enable_thinking,  # This will only apply to thinking models
    )

    with vllm_runner(model_id, dtype=torch.float16, max_model_len=512) as llm:
        # Generate a response to verify inference correctness
        output = llm.generate_greedy([text], max_tokens=128)

    # Make sure the output exists
    assert output
    assert output[0][1]

    def has_normal_char_distribution(text, min_len):
        # Response too short
        if len(text) < min_len:
            return False

        # Basic ratio checks
        letters = sum(c.isalpha() for c in text)
        spaces = sum(c.isspace() for c in text)
        total = len(text)

        letter_ratio = letters / total
        space_ratio = spaces / total

        # Normal text should be mostly letters with reasonable spacing
        # Some magic numbers, could be adjusted
        return (0.5 <= letter_ratio <= 0.9 and 0.01 <= space_ratio <= 0.3)

    # Apply some simple checks for giberish output
    assert has_normal_char_distribution(output[0][1], 5)
    # Also print the output text to check
    print(f"{output[0][1]}")
