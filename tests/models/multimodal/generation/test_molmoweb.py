# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MolmoWeb models (allenai/MolmoWeb-8B, allenai/MolmoWeb-4B).

MolmoWeb is a multimodal web agent built on the Molmo2 architecture
(Molmo2ForConditionalGeneration). These tests verify that MolmoWeb models
work correctly through vLLM's existing Molmo2 handler.
"""

import pytest

from vllm import LLM, SamplingParams

MODELS = [
    "allenai/MolmoWeb-4B",
    "allenai/MolmoWeb-8B",
]


@pytest.mark.parametrize("model", MODELS)
def test_molmoweb_single_image(image_assets, model: str):
    """Test MolmoWeb with a single image input.

    Verifies that MolmoWeb models correctly load and generate text
    given an image through vLLM's Molmo2 handler.
    """
    images = [asset.pil_image.convert("RGB") for asset in image_assets]

    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        seed=42,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    for image in images:
        outputs = llm.generate(
            {
                "prompt": "Describe this image briefly.",
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
        )

        assert len(outputs) == 1
        generated_text = outputs[0].outputs[0].text
        assert len(generated_text) > 10, (
            f"Generated text is too short: {generated_text!r}"
        )


@pytest.mark.parametrize("model", MODELS)
def test_molmoweb_multi_image(image_assets, model: str):
    """Test MolmoWeb with multiple images in a single request."""
    images = [asset.pil_image.convert("RGB") for asset in image_assets]

    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        seed=42,
        limit_mm_per_prompt={"image": len(images)},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    outputs = llm.generate(
        {
            "prompt": "Describe what you see in these images.",
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )

    assert len(outputs) == 1
    generated_text = outputs[0].outputs[0].text
    assert len(generated_text) > 10, (
        f"Generated text is too short: {generated_text!r}"
    )
