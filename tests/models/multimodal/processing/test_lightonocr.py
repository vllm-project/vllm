# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LightOnOCR precomputed image embeddings."""

import pytest
import torch
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

_LIGHTON_MODEL_ID = "lightonai/LightOnOCR-1B-1025"
_MODEL_CONFIG_KWARGS = {"config_format": "hf"}


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("tensor_input", [False, True])
@pytest.mark.parametrize("prompt_as_tokens", [False, True])
def test_lightonocr_image_embeddings_preserve_lengths_and_values(
    tensor_input: bool, prompt_as_tokens: bool
):
    """Reserve each image's feature count and pass projected embeddings through."""
    ctx = build_model_context(
        _LIGHTON_MODEL_ID,
        limit_mm_per_prompt={"image": 2},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    ctx.model_config.get_multimodal_config().enable_mm_embeds = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    hidden_size = ctx.model_config.get_hidden_size()
    feature_sizes = [3, 3] if tensor_input else [3, 7]
    embeddings = [
        torch.full((size, hidden_size), float(i))
        for i, size in enumerate(feature_sizes)
    ]
    image_input = torch.stack(embeddings) if tensor_input else embeddings
    prompt = " ".join([hf_processor.image_token] * len(embeddings))
    if prompt_as_tokens:
        prompt = ctx.get_tokenizer().encode(prompt)

    items = processor.info.parse_mm_data({"image": image_input})
    processed = processor(prompt, mm_items=items)

    assert processed["prompt_token_ids"].count(hf_processor.image_token_id) == sum(
        feature_sizes
    )
    assert [p.length for p in processed["mm_placeholders"]["image"]] == feature_sizes
    for item, expected in zip(processed["mm_kwargs"]["image"], embeddings):
        assert item is not None
        torch.testing.assert_close(item.get_data()["image_embeds"], expected)


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("longest_edge", [448, 896])
@pytest.mark.parametrize("prompt_as_tokens", [False, True])
def test_resize_kwargs_match_image_features(longest_edge, prompt_as_tokens):
    """Resized images reserve one placeholder per projected vision feature."""
    images = [Image.new("RGB", (1600, 1200)), Image.new("RGB", (300, 800))]
    kwargs = {"size": {"longest_edge": longest_edge}}
    results = []
    for on_engine in (False, True):
        ctx = build_model_context(
            _LIGHTON_MODEL_ID,
            model_config_kwargs=_MODEL_CONFIG_KWARGS,
            mm_processor_kwargs=kwargs if on_engine else None,
            limit_mm_per_prompt={"image": 2},
            mm_processor_cache_gb=0.01,
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        prompt = processor.dummy_inputs.get_dummy_text({"image": 2})
        if prompt_as_tokens:
            prompt = ctx.get_tokenizer().encode(prompt)
        config = ctx.model_config.hf_config
        stride = config.vision_config.patch_size * config.spatial_merge_size
        for _ in range(2):
            result = processor(
                prompt,
                mm_items=processor.info.parse_mm_data({"image": images}),
                hf_processor_mm_kwargs={} if on_engine else kwargs,
            )
            for placeholder, item in zip(
                result["mm_placeholders"]["image"], result["mm_kwargs"]["image"]
            ):
                assert item is not None
                pixels = item["pixel_values"].data
                assert isinstance(pixels, torch.Tensor)
                height, width = pixels.shape[-2:]
                assert placeholder.length == ((height + stride - 1) // stride) * (
                    (width + stride - 1) // stride
                )
        results.append(result["prompt_token_ids"])
    assert results[0] == results[1]
