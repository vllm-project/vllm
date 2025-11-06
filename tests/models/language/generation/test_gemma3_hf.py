# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma3 HF text-only generation tests.

Tests HuggingFace safetensors models via vLLM runner to ensure correct
loading and inference for both 1B and 4B Gemma3 models.
"""

# Test prompts
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def test_gemma3_1b_hf_text_only(vllm_runner):
    """Test Gemma3 1B HF text-only generation."""
    with vllm_runner(
        "google/gemma-3-1b-it",
        max_model_len=1024,
        tensor_parallel_size=1,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(PROMPTS, max_tokens=32)

    # Verify outputs are generated
    for i in range(len(PROMPTS)):
        output_ids, output_str = vllm_outputs[i]
        assert len(output_ids) > 0, f"No tokens generated for prompt {i}"
        assert len(output_str) > 0, f"Empty output string for prompt {i}"
        assert output_str.strip(), f"Output is only whitespace for prompt {i}"


def test_gemma3_4b_hf_text_only(vllm_runner):
    """Test Gemma3 4B HF text-only generation."""
    with vllm_runner(
        "google/gemma-3-4b-it",
        max_model_len=1024,
        tensor_parallel_size=1,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(PROMPTS, max_tokens=32)

    # Verify outputs are generated
    for i in range(len(PROMPTS)):
        output_ids, output_str = vllm_outputs[i]
        assert len(output_ids) > 0, f"No tokens generated for prompt {i}"
        assert len(output_str) > 0, f"Empty output string for prompt {i}"
        assert output_str.strip(), f"Output is only whitespace for prompt {i}"
