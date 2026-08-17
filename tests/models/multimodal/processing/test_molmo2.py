# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.multimodal import VideoDummyOptions
from vllm.model_executor.models.molmo2 import (
    Molmo2DummyInputsBuilder,
    build_flat_image_bool_length,
)


def test_build_flat_image_bool_length_matches_molmoweb_processor_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[14, 14, 14, 23]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=True,
        use_single_crop_col_tokens=None,
        use_single_crop_start_token=False,
    )

    assert num_image_tokens.tolist() == [550]
    assert len(image_tokens) == 550
    assert image_tokens[0].item() == hf_config.image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 28


def test_build_flat_image_bool_length_respects_disabled_col_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[2, 3, 5, 7]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=False,
        use_single_crop_col_tokens=False,
        use_single_crop_start_token=True,
    )

    assert num_image_tokens.tolist() == [45]
    assert len(image_tokens) == 45
    assert image_tokens[0].item() == hf_config.low_res_image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 0


@pytest.mark.parametrize(
    ("num_frames_override", "expected_frames"),
    [(1, 2), (2, 2), (3, 3)],
)
def test_dummy_video_num_frames_override_honors_min_of_two(
    num_frames_override, expected_frames
):
    """A ``num_frames`` override below 2 must be ignored (the model needs at
    least 2 frames), matching the "cannot be less than 2, will be ignored"
    warning."""
    builder = object.__new__(Molmo2DummyInputsBuilder)
    builder.info = SimpleNamespace(
        get_hf_processor=lambda: SimpleNamespace(
            video_processor=SimpleNamespace(size={"width": 64, "height": 64})
        ),
        get_num_frames_with_most_features=lambda seq_len, mm_counts: 16,
        get_image_size_with_most_features=lambda: (64, 64),
    )

    data = builder.get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 0, "video": 1},
        mm_options={"video": VideoDummyOptions(num_frames=num_frames_override)},
    )

    video, _metadata = data["video"][0]
    assert video.shape[0] == expected_frames
