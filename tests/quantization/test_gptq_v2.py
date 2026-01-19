# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether vllm correctly load and run gptq_v2 format checkpoints.

Run `pytest tests/quantization/test_gptq_v2.py --forked`.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod

# A dummy small model quantized by GPTQModel, stored in GPTQ v2 format
MODELS = ["XXXXyu/Qwen3-1.7B-w2g64-gptq_v2"]

# Generate multiple sequences for testing, because an 1.7B 2-bit model
# cannot always generate normal texts.
N_SEQ = 5


@pytest.mark.parametrize("model_id", MODELS)
def test_model_load(vllm_runner, model_id, monkeypatch):
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Only check the default GPTQ linear method (used for 2/3-bit models).
    # 4/8-bit linear methods like Marlin already support gptq_v2.
    linear_method_cls = GPTQLinearMethod

    with vllm_runner(model_id, dtype=torch.float16, max_model_len=512) as llm:

        def check_model(model_id):
            for name, submodule in model_id.named_modules():
                # Could check more modules if necessary
                if name == "model_id.layers.0.self_attn.qkv_proj":
                    assert isinstance(submodule.quant_method, linear_method_cls)

                    config = submodule.quant_method.quant_config
                    assert config.checkpoint_format == "gptq_v2"
                    assert submodule.quant_method.use_v2_format

                    # Just break since currently we only check 1 module
                    break

        # Check if gptq_v2 format is correctly loaded
        llm.apply_model(check_model)


@pytest.mark.parametrize("model_id", MODELS)
def test_model_inference(vllm_runner, model_id):
    # Prepare prompt to test the model's generation result.
    prompt = "What is the meaning of life?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # If thinking model, set it to false
    )
    sampling_params = SamplingParams(
        n=N_SEQ,
        max_tokens=128,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        presence_penalty=2,
    )

    with vllm_runner(model_id, dtype=torch.float16, max_model_len=512) as llm:
        # Generate a response to verify inference correctness
        output = llm.generate(text, sampling_params)

    # Make sure the output exists
    assert output
    assert output[0][1]
    assert len(output[0][1]) == N_SEQ

    def has_normal_char_distribution(texts, min_len):
        for text in texts:
            # Response too short
            if len(text) < min_len:
                return False

            # Basic ratio checks
            letters = sum(c.isalpha() for c in text)
            spaces = sum(c.isspace() for c in text)
            total = len(text)

            letter_ratio = letters / total
            space_ratio = spaces / total

            # At least 1 normal text should exist within output sequences
            # Normal text should be mostly letters with reasonable spacing
            # Some magic numbers, could be adjusted
            if 0.5 <= letter_ratio <= 0.9 and 0.01 <= space_ratio <= 0.3:
                return True
        # No sequence contains normal text, output might be broken
        return False

    # Apply some simple checks for giberish output
    # Print the output sequences if failed
    assert has_normal_char_distribution(output[0][1], 5), output[0][1]
