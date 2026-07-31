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
    """Carries only the two attributes the token helpers read from
    `self`, so the real methods can be exercised without constructing
    the full `nn.Module` (vision tower, language model, etc.)."""

    patch_grid_length: int
    pooled_grid_length: int


get_num_mm_encoder_tokens = (
    LlavaNextVideoForConditionalGeneration.get_num_mm_encoder_tokens
)
get_num_mm_connector_tokens = (
    LlavaNextVideoForConditionalGeneration.get_num_mm_connector_tokens
)


@pytest.mark.parametrize("model_id", ["llava-hf/LLaVA-NeXT-Video-7B-hf"])
def test_num_mm_tokens_match_real_config(model_id):
    """The stored grid lengths must match what `__init__` derives from
    the real HF config, and the two helpers must invert each other's
    frame-level scaling exactly."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"video": 1})
    config = ctx.model_config.hf_config

    vision_encoder_info = get_vision_encoder_info(config)
    patch_grid_length = vision_encoder_info.get_patch_grid_length()
    pooled_grid_length = math.ceil(patch_grid_length / config.spatial_pool_stride)

    stub = _StubModel()
    stub.patch_grid_length = patch_grid_length
    stub.pooled_grid_length = pooled_grid_length

    for num_frames in (1, 2, 8, 16, 32):
        num_video_tokens = num_frames * pooled_grid_length**2

        encoder_tokens = get_num_mm_encoder_tokens(stub, num_video_tokens)
        assert encoder_tokens == num_frames * patch_grid_length**2

        connector_tokens = get_num_mm_connector_tokens(stub, encoder_tokens)
        assert connector_tokens == num_video_tokens


@pytest.mark.parametrize(
    ("patch_grid_length", "pooled_grid_length", "num_frames"),
    [
        (24, 12, 1),  # llava-hf/LLaVA-NeXT-Video-7B-hf: 336 / 14, stride 2
        (24, 12, 16),
        (27, 14, 5),  # non-power-of-2 pooled grid (ceil rounding)
        (16, 8, 6),
    ],
)
def test_num_mm_tokens_roundtrip(patch_grid_length, pooled_grid_length, num_frames):
    stub = _StubModel()
    stub.patch_grid_length = patch_grid_length
    stub.pooled_grid_length = pooled_grid_length

    num_video_tokens = num_frames * pooled_grid_length**2

    encoder_tokens = get_num_mm_encoder_tokens(stub, num_video_tokens)
    assert encoder_tokens == num_frames * patch_grid_length**2

    connector_tokens = get_num_mm_connector_tokens(stub, encoder_tokens)
    assert connector_tokens == num_video_tokens


def test_num_mm_tokens_zero():
    stub = _StubModel()
    stub.patch_grid_length = 24
    stub.pooled_grid_length = 12

    assert get_num_mm_encoder_tokens(stub, 0) == 0
    assert get_num_mm_connector_tokens(stub, 0) == 0
