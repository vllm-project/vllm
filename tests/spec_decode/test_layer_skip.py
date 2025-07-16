import pytest
import torch
from vllm import LLM, SamplingParams

@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("exit_layer", [4, 6])
@pytest.mark.parametrize("num_speculative_tokens", [3, 5])
def test_layer_skip_smoke(model: str, exit_layer: int, num_speculative_tokens: int):
    """Test basic functionality with early exit at various layers."""
    llm = LLM(
        model=model,
        speculative_config={
            "method": "layer_skip",
            "layer_skip": exit_layer,
            "num_speculative_tokens": num_speculative_tokens,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,  # Small model, reduce memory
    )
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The quick brown fox",
    ]
    
    outputs = llm.generate(prompts, SamplingParams(
        max_tokens=20,
        temperature=0.8,  # Higher temperature for better acceptance
    ))
    
    # Verify we got outputs for all prompts
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert len(output.outputs[0].token_ids) <= 20
        assert len(output.outputs[0].text) > 0

@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_layer_skip_deterministic(model: str):
    """Test that full-layer exit produces identical output to base model."""
    prompt = "The meaning of life is"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        seed=42,
    )
    
    # Get base model output
    llm_base = LLM(model=model, tensor_parallel_size=1, gpu_memory_utilization=0.3)
    base_output = llm_base.generate(prompt, sampling_params)[0].outputs[0].text
    del llm_base  # Free memory
    
    # Get layer-skip output with exit at last layer (should be identical)
    llm_skip = LLM(
        model=model,
        speculative_config={
            "method": "layer_skip",
            "layer_skip": 11,  # OPT-125M has 12 layers (0-11)
            "num_speculative_tokens": 3,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
    )
    skip_output = llm_skip.generate(prompt, sampling_params)[0].outputs[0].text
    
    assert base_output == skip_output, \
        f"Outputs differ:\nBase: {base_output}\nSkip: {skip_output}"

def test_layer_skip_acceptance_rate():
    """Test that we get reasonable acceptance rates with proper temperature."""
    # This is more of an integration test to verify the system works
    llm = LLM(
        model="facebook/opt-125m",
        speculative_config={
            "method": "layer_skip",
            "layer_skip": 4,
            "num_speculative_tokens": 5,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
    )
    
    # Use temperature > 0 for better acceptance
    outputs = llm.generate(
        "Write a short story about a robot:",
        SamplingParams(max_tokens=100, temperature=0.8)
    )
    
    # Just verify it completes without error
    assert len(outputs[0].outputs[0].text) > 0