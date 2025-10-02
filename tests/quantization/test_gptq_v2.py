# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq_v2 format checkpoints are correctly handled.

Run `pytest tests/quantization/test_gptq_v2.py --forked`.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod

# Asymmetrically quantized model, stored in GPTQ v2 format
MODELS = ['BitDistiller/Qwen-8B-w2g64-gptq']  # TODO: add more maybe


@pytest.mark.parametrize("model", MODELS)
def test_gptq_v2(vllm_runner, model, monkeypatch):

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Currently we only check the default GPTQ linear method.
    # TODO: add test for other kernels, e.g., Marlin.
    linear_method_cls = GPTQLinearMethod

    # Prepare prompt to test the model's generation result.
    prompt = "What is the meaning of life?"
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": prompt
    }]
    tokenizer = AutoTokenizer.from_pretrained(model)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # This will only apply to thinking models
    )

    with vllm_runner(model, dtype=torch.float16, max_model_len=1024) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == 'model.layers.0.self_attn.qkv_proj':  # for each?
                    assert isinstance(submodule.quant_method,
                                      linear_method_cls)
                    config = submodule.quant_method.quant_config
                    assert config.checkpoint_format == 'gptq_v2'
                    assert submodule.use_gptq_gemm_v2 == True

        # Check if gptq_v2 is detected
        llm.apply_model(check_model)
        # Generate a response to verify inference correctness
        output = llm.generate_greedy([text])
    assert output
    print(f"{output[0][1]}")
