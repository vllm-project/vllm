# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma3 GGUF text-only generation tests.

Tests GGUF quantized models via vLLM runner to ensure correct loading
and inference for both 1B and 4B Gemma3 models.
"""

from huggingface_hub import hf_hub_download

# Test prompts
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def test_gemma3_1b_gguf_text_only(vllm_runner):
    """Test Gemma3 1B GGUF text-only generation."""
    # Download GGUF model
    gguf_file = hf_hub_download(
        repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
        filename="gemma-3-1b-it-q4_0.gguf",
    )

    with vllm_runner(
        gguf_file,
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


def test_gemma3_4b_gguf_text_only(vllm_runner):
    """Test Gemma3 4B GGUF text-only generation."""
    # Download GGUF model
    gguf_file = hf_hub_download(
        repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
        filename="gemma-3-4b-it-q4_0.gguf",
    )

    with vllm_runner(
        gguf_file,
        tokenizer_name="google/gemma-3-4b-it",  # Required for processor loading
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
