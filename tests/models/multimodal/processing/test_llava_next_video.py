# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import CLIPVisionConfig, LlavaNextVideoConfig, SiglipVisionConfig

from vllm.model_executor.models.llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
)

from ...utils import build_model_context


class _StubModel:
    """Carries only the attributes the token-count method reads from
    `self`, so the real methods can be exercised without constructing
    the full `nn.Module` (vision tower, language model, etc.)."""

    config: LlavaNextVideoConfig


get_mm_lora_token_counts = (
    LlavaNextVideoForConditionalGeneration.get_mm_lora_token_counts
)


@pytest.mark.parametrize("model_id", ["llava-hf/LLaVA-NeXT-Video-7B-hf"])
def test_num_mm_tokens_match_real_config(model_id):
    """The LoRA token counts derived from the real HF config must preserve
    the frame-level scaling exactly: 12 * 12 pooled tokens per frame for the
    connector, and 24 * 24 patches plus the CLS token for the CLIP tower."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"video": 1})

    stub = _StubModel()
    stub.config = ctx.model_config.hf_config

    for num_frames in (1, 2, 8, 16, 32):
        num_video_tokens = num_frames * 144

        encoder_tokens, connector_tokens = get_mm_lora_token_counts(
            stub,
            modality="video",
            mm_kwargs=None,
            num_mm_embeds=num_video_tokens,
        )
        assert encoder_tokens == num_frames * 577
        assert connector_tokens == num_video_tokens


@pytest.mark.parametrize(
    (
        "vision_config",
        "tower_tokens_per_frame",
        "pooled_tokens_per_frame",
        "num_frames",
    ),
    [
        # llava-hf/LLaVA-NeXT-Video-7B-hf: 24 * 24 patches + CLS, pooled to 12 * 12
        (CLIPVisionConfig(image_size=336, patch_size=14), 577, 144, 1),
        (CLIPVisionConfig(image_size=336, patch_size=14), 577, 144, 16),
        # non-power-of-2 pooled grid (ceil rounding), no CLS
        (SiglipVisionConfig(image_size=384, patch_size=14), 729, 196, 5),
        (CLIPVisionConfig(image_size=224, patch_size=14), 257, 64, 6),
    ],
)
def test_num_mm_tokens_roundtrip(
    vision_config, tower_tokens_per_frame, pooled_tokens_per_frame, num_frames
):
    stub = _StubModel()
    stub.config = LlavaNextVideoConfig(
        vision_config=vision_config, spatial_pool_stride=2
    )

    num_video_tokens = num_frames * pooled_tokens_per_frame

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
    stub.config = LlavaNextVideoConfig(
        vision_config=CLIPVisionConfig(image_size=336, patch_size=14)
    )

    assert get_mm_lora_token_counts(
        stub,
        modality="video",
        mm_kwargs=None,
        num_mm_embeds=0,
    ) == (0, 0)
