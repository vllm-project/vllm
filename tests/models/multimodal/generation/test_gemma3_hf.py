# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma3 HF multimodal generation tests.

Tests HuggingFace safetensors multimodal model via vLLM runner to ensure
correct loading and inference for the 4B Gemma3 model.
"""

from vllm.assets.image import ImageAsset

# Multimodal test prompt
PROMPT = "Describe this image in detail:"


def test_gemma3_4b_hf_multimodal(vllm_runner):
    """Test Gemma3 4B HF multimodal generation."""
    # Get test image
    image = ImageAsset("stop_sign").pil_image

    with vllm_runner(
        "google/gemma-3-4b-it",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
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
