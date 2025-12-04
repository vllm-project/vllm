# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode, DynamicShapesType
from vllm.tokenizers import get_tokenizer
from vllm.utils.torch_utils import is_torch_equal_or_newer


def get_test_models():
    """Get list of models to test based on PyTorch version"""
    # TODO "Qwen/Qwen3-4B-Instruct-2507" fails Fix issue and support it.
    return ["gpt2", "Qwen/Qwen2-7B-Instruct", "meta-llama/Llama-3.1-8B"]


@pytest.mark.parametrize("model_name", get_test_models())
@pytest.mark.parametrize(
    "shapes_type",
    [
        DynamicShapesType.BACKED,
        DynamicShapesType.UNBACKED,
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
    ],
)
@pytest.mark.parametrize("use_aot_compile", ["0"])
@pytest.mark.parametrize("use_bytecode_hook", [True, False])
@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
def test_dynamic_shapes_compilation(
    monkeypatch, model_name, shapes_type, use_aot_compile, use_bytecode_hook
):
    """Test that all dynamic shapes types compile successfully"""
    print(
        f"\nTesting model: {model_name} with {shapes_type.name}, "
        f"AOT compile: {use_aot_compile}, "
        f"Bytecode hook: {use_bytecode_hook}"
    )
    if use_bytecode_hook and shapes_type == DynamicShapesType.UNBACKED:
        pytest.skip("UNBACKED dynamic shapes require VLLM_USE_BYTECODE_HOOK=0")

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", use_aot_compile)
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    prompt = "Hello, my name is"

    print(f"Testing {shapes_type.name} dynamic shapes...")

    # Initialize the model with specific dynamic shapes configuration
    model = LLM(
        model=model_name,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "dynamic_shapes_config": {
                "type": shapes_type.value,
            },
        },
    )

    output = model.generate(prompt)
    result = output[0].outputs[0].text
    # Example of setting the sampling parameters
    tokenizer = get_tokenizer(model_name)
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)
    allowed_ids = list(set(yes_tokens + no_tokens))
    sampling_params = SamplingParams(
        max_tokens=1, temperature=0, allowed_token_ids=allowed_ids
    )

    output = model.generate(
        "answer with yes or no is " + result + " rubbish for prompt " + prompt + "?",
        sampling_params=sampling_params,
    )
    result = output[0].outputs[0].text
    assert result == "yes"

    # Clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU memory cleared")
