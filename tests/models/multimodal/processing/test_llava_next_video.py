# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from vllm.model_executor.models.llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
)
from vllm.model_executor.models.vision import get_vision_encoder_info

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
