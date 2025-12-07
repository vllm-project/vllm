# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mllama's multimodal preprocessing and profiling."""

import pytest
from torch import prod
from transformers import Llama4Config

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.profiling import MultiModalProfiler

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["meta-llama/Llama-Guard-4-12B"])
@pytest.mark.parametrize("max_model_len", [4096, 8192, 25600, 131072])
def test_profiling(model_id: str, max_model_len: int):
    model_config_kwargs = {
        "max_model_len": max_model_len,
    }
    mm_counts = {"image": 1}
    ctx = build_model_context(
        model_id,
        model_config_kwargs=model_config_kwargs,
        limit_mm_per_prompt=mm_counts,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(ctx.renderer_config)
    profiler = MultiModalProfiler(processor)

    decoder_dummy_data = profiler.get_decoder_dummy_data(
        max_model_len,
        mm_counts=mm_counts,
    )
    dummy_mm_data = processor.dummy_inputs.get_dummy_processor_inputs(
        max_model_len,
        mm_counts=mm_counts,
    )

    hf_config = ctx.get_hf_config(Llama4Config)

    mm_data = processor.apply(
        prompt=dummy_mm_data.prompt,
        mm_data=dummy_mm_data.mm_data,
        hf_processor_mm_kwargs=dict(),
    )["mm_kwargs"].get_data()

    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size
    downsample_ratio = int(
        round(1.0 / (hf_config.vision_config.pixel_shuffle_ratio**2))
    )
    tokens_per_patch = ((image_size // patch_size) ** 2) // downsample_ratio
    chunks_per_image = prod(mm_data["patches_per_image"])
    total_num_patches = chunks_per_image * tokens_per_patch
    num_tiles = (
        mm_data["aspect_ratios"][0][0] * mm_data["aspect_ratios"][0][1]
    )  # x-y separator tokens
    total_tokens = (
        total_num_patches.item() + num_tiles.item() + 3
    )  # image start, image, image end

    profiled_tokens = profiler.get_mm_max_contiguous_tokens(
        max_model_len,
        mm_counts=mm_counts,
    )

    assert total_tokens == profiled_tokens["image"]
    assert total_tokens == sum(
        placeholder.length
        for placeholder in decoder_dummy_data.multi_modal_placeholders["image"]
    )
