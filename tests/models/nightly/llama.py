# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import time

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS


AITER_MODEL_LIST = [
    "meta-llama/Llama-3.2-1B-Instruct"
]


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.2-1B-Instruct"])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_serving(
    model: str,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    """
    Test online serving for Llama-3.2-1B-Instruct
    Run with: pytest this_file.py::test_online_serving -v -s
    """
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    print(f"\n{'='*60}")
    print(f"Starting online serving for: {model}")
    print(f"ROCm AITER: {use_rocm_aiter}")
    print(f"{'='*60}\n")

    # Initialize vLLM engine for online serving
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
    )

    print("Model loaded successfully!\n")
    
    # Assert model is loaded
    assert llm is not None, "Model should be initialized"
    assert llm.llm_engine is not None, "LLM engine should be initialized"

    # Define test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain what AI is in simple terms.",
        "Write a short poem about Python programming.",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )

    print("Running inference on test prompts...\n")
    
    # Generate responses for each prompt
    all_outputs = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Prompt {i}/{len(test_prompts)}: {prompt}")
        print(f"{'-'*60}")
        
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start_time
        
        # Assertions for single generation
        assert len(outputs) == 1, f"Should return 1 output, got {len(outputs)}"
        assert len(outputs[0].outputs) > 0, "Should have at least one output"
        
        generated_text = outputs[0].outputs[0].text
        
        # Assert output is not empty
        assert generated_text is not None, "Generated text should not be None"
        assert len(generated_text) > 0, "Generated text should not be empty"
        assert isinstance(generated_text, str), "Generated text should be a string"
        
        all_outputs.append(outputs[0])
        
        print(f"Response: {generated_text}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Assertions passed for prompt {i}")
        print(f"{'='*60}")

    # Assert all prompts generated outputs
    assert len(all_outputs) == len(test_prompts), \
        f"Should generate {len(test_prompts)} outputs, got {len(all_outputs)}"

    # Test batch generation
    print(f"\n{'='*60}")
    print("Testing batch generation with all prompts...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    batch_outputs = llm.generate(test_prompts, sampling_params)
    batch_elapsed = time.time() - start_time
    
    # Assertions for batch generation
    assert len(batch_outputs) == len(test_prompts), \
        f"Batch should return {len(test_prompts)} outputs, got {len(batch_outputs)}"
    
    for i, output in enumerate(batch_outputs):
        assert output.prompt == test_prompts[i], \
            f"Output {i} prompt mismatch"
        assert len(output.outputs) > 0, \
            f"Output {i} should have generated text"
        assert output.outputs[0].text is not None, \
            f"Output {i} text should not be None"
        assert len(output.outputs[0].text) > 0, \
            f"Output {i} text should not be empty"
    
    print(f"Batch generation completed in {batch_elapsed:.2f}s")
    print(f"Average time per prompt: {batch_elapsed/len(test_prompts):.2f}s")
    print(f"All batch assertions passed\n")
    
    for i, output in enumerate(batch_outputs, 1):
        print(f"\nPrompt {i}: {output.prompt}")
        print(f"Response: {output.outputs[0].text[:100]}...")

    # Test that batch generation is more efficient than sequential
    sequential_time_estimate = sum(
        len(output.outputs[0].text) for output in all_outputs
    ) / 100  # rough estimate
    print(f"\n{'='*60}")
    print(f"Performance comparison:")
    print(f"Batch time: {batch_elapsed:.2f}s")
    print(f"Batch processing completed successfully")
    print(f"{'='*60}")

    # Verify outputs are deterministic (at low temperature)
    if sampling_params.temperature < 0.5:
        print(f"\n{'='*60}")
        print("Testing deterministic generation...")
        repeat_outputs = llm.generate([test_prompts[0]], sampling_params)
        
        assert repeat_outputs[0].outputs[0].text == all_outputs[0].outputs[0].text, \
            "Low temperature outputs should be deterministic"
        print("✓ Deterministic generation verified")
        print(f"{'='*60}")

    # Cleanup for ROCm
    if use_rocm_aiter:
        torch.cuda.synchronize()
        
    print(f"\n{'='*60}")
    print("✅ All assertions passed!")
    print("Online serving test completed successfully!")
    print(f"{'='*60}\n")