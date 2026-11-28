# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

from vllm.model_executor.models.llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
)
from vllm.model_executor.models.vision import get_vision_encoder_info
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache

from ...utils import build_model_context


class _StubModel:
    """Carries only the attributes the token-count method reads from
    `self`, so the real methods can be exercised without constructing
    the full `nn.Module` (vision tower, language model, etc.)."""

    tower_tokens_per_frame: int
    pooled_grid_length: int


get_mm_lora_token_counts = (
    LlavaNextVideoForConditionalGeneration.get_mm_lora_token_counts
)


@pytest.mark.parametrize("model_id", ["llava-hf/LLaVA-NeXT-Video-7B-hf"])
def test_num_mm_tokens_match_real_config(model_id):
    """The stored values must match what `__init__` derives from
    the real HF config, and the LoRA token counts must preserve the
    frame-level scaling exactly."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"video": 1})
    config = ctx.model_config.hf_config

    vision_encoder_info = get_vision_encoder_info(config)
    patch_grid_length = vision_encoder_info.get_patch_grid_length()
    pooled_grid_length = math.ceil(patch_grid_length / config.spatial_pool_stride)
    image_size = vision_encoder_info.get_image_size()

    stub = _StubModel()
    stub.tower_tokens_per_frame = vision_encoder_info.get_num_image_tokens(
        image_width=image_size, image_height=image_size
    )
    stub.pooled_grid_length = pooled_grid_length

    for num_frames in (1, 2, 8, 16, 32):
        num_video_tokens = num_frames * pooled_grid_length**2

        encoder_tokens, connector_tokens = get_mm_lora_token_counts(
            stub,
            modality="video",
            mm_kwargs=None,
            num_mm_embeds=num_video_tokens,
        )
        assert encoder_tokens == num_frames * (patch_grid_length**2 + 1)  # + CLS
        assert connector_tokens == num_video_tokens


@pytest.mark.parametrize(
    ("tower_tokens_per_frame", "pooled_grid_length", "num_frames"),
    [
        (577, 12, 1),  # llava-hf/LLaVA-NeXT-Video-7B-hf: 24 ** 2 + CLS, stride 2
        (577, 12, 16),
        (729, 14, 5),  # non-power-of-2 pooled grid (ceil rounding), no CLS
        (256, 8, 6),
    ],
)
def test_num_mm_tokens_roundtrip(
    tower_tokens_per_frame, pooled_grid_length, num_frames
):
    stub = _StubModel()
    stub.tower_tokens_per_frame = tower_tokens_per_frame
    stub.pooled_grid_length = pooled_grid_length

    num_video_tokens = num_frames * pooled_grid_length**2

    encoder_tokens, connector_tokens = get_mm_lora_token_counts(
        stub,
        modality="video",
        mm_kwargs=None,
        num_mm_embeds=num_video_tokens,
    )
    assert encoder_tokens == num_frames * tower_tokens_per_frame
    assert connector_tokens == num_video_tokens


def test_num_mm_tokens_zero():
    stub = _StubModel()
    stub.tower_tokens_per_frame = 577
    stub.pooled_grid_length = 12

    assert get_mm_lora_token_counts(
        stub,
        modality="video",
        mm_kwargs=None,
        num_mm_embeds=0,
    ) == (0, 0)


@pytest.mark.parametrize(
    "model_id",
    [
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    ],
)
@pytest.mark.parametrize("num_frames", [None, 4, 8])
@pytest.mark.parametrize("kwargs_on_init", [False, True])
@pytest.mark.parametrize("token_prompt", [False, True])
def test_video_placeholders_follow_sampled_frames(
    model_id, num_frames, kwargs_on_init, token_prompt
):
    """Sampling must reserve one embedding slot per produced video feature."""
    kwargs = (
        {"do_sample_frames": True, "num_frames": num_frames}
        if num_frames is not None
        else {}
    )
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"video": 1},
        mm_processor_cache_gb=1,
        mm_processor_kwargs=kwargs if kwargs_on_init else None,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    frames = np.zeros((12, 224, 320, 3), dtype=np.uint8)
    prompt = hf_processor.video_token
    reference = hf_processor(
        text=prompt, videos=[frames], return_tensors="pt", **kwargs
    )
    expected_ids = reference["input_ids"][0].tolist()
    video_token_id = ctx.model_config.hf_config.video_token_index
    expected_tokens = expected_ids.count(video_token_id)
    expected_frames = num_frames if num_frames is not None else len(frames)
    assert reference["pixel_values_videos"].shape[1] == expected_frames
    if token_prompt:
        prompt = ctx.tokenizer.encode(prompt)
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    for _ in range(2):
        processed = processor(
            prompt,
            mm_items=processor.info.parse_mm_data({"video": [frames]}),
            hf_processor_mm_kwargs={} if kwargs_on_init else kwargs,
            cache=cache,
        )
        pixels = processed["mm_kwargs"].get_data()["pixel_values_videos"][0]
        assert isinstance(pixels, torch.Tensor)
        assert pixels.shape[0] == expected_frames
        assert (
            processed["mm_placeholders"]["video"][0].get_num_embeds() == expected_tokens
        )
        assert processed["prompt_token_ids"] == expected_ids
