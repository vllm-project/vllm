# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma3 GGUF multimodal generation tests.

Tests GGUF quantized multimodal model via vLLM runner to ensure correct
loading and inference for the 4B Gemma3 model with mmproj support.
"""

from huggingface_hub import hf_hub_download

from vllm.assets.image import ImageAsset

# Multimodal test prompt
PROMPT = "<start_of_image> Describe this image in detail:"


def test_gemma3_4b_gguf_multimodal(vllm_runner):
    """Test Gemma3 4B GGUF multimodal generation."""
    # Download GGUF model
    gguf_file = hf_hub_download(
        repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
        filename="gemma-3-4b-it-q4_0.gguf",
    )

    # Download mmproj file
    mmproj_file = hf_hub_download(
        repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
        filename="mmproj-model-f16-4B.gguf",
    )

    # Get test image
    image = ImageAsset("stop_sign").pil_image

    with vllm_runner(
        gguf_file,
        tokenizer_name="google/gemma-3-4b-it",
        max_model_len=4096,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(
            [PROMPT],
            max_tokens=128,
            images=[image],
        )

    # Verify output is generated
    output_ids, output_str = vllm_outputs[0]
    assert len(output_ids) > 0, "No tokens generated"
    assert len(output_str) > 0, "Empty output string"
    assert output_str.strip(), "Output is only whitespace"

    # Verify output contains image-related content
    # (basic sanity check that model processed the image)
    output_lower = output_str.lower()
    # Check for common image description indicators
    image_keywords = [
        "image",
        "picture",
        "photo",
        "sign",
        "red",
        "stop",
        "shows",
        "depicts",
    ]
    assert any(word in output_lower for word in image_keywords), (
        f"Output doesn't appear to describe the image: {output_str}"
    )
