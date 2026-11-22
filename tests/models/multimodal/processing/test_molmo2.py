# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm.config.multimodal import MultiModalDummyOptions, VideoDummyOptions
from vllm.model_executor.models.molmo2 import (
    Molmo2DummyInputsBuilder,
    Molmo2VisionBackbone,
    build_flat_image_bool_length,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

pytestmark = pytest.mark.skip_global_cleanup


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
    """A `num_frames` override below 2 must be ignored (the model needs at
    least 2 frames), matching the "cannot be less than 2, will be ignored"
    warning."""
    builder = object.__new__(Molmo2DummyInputsBuilder)
    builder.info = SimpleNamespace(
        get_hf_processor=lambda: SimpleNamespace(
            video_processor=SimpleNamespace(size={"width": 64, "height": 64})
        ),
        get_num_frames_with_most_features=lambda seq_len, num_videos: 16,
        get_image_size_with_most_features=lambda: (64, 64),
    )

    data = builder.get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 0, "video": 1},
        mm_options=MultiModalDummyOptions(
            video=VideoDummyOptions(num_frames=num_frames_override)
        ),
    )

    videos = data["video"]
    assert isinstance(videos, list)
    video, _metadata = videos[0]
    assert video.shape[0] == expected_frames


@pytest.mark.parametrize("masked_average", [False, True])
def test_vision_pooling_means_preserve_batch_indices_and_drop_invalid_groups(
    masked_average,
):
    """Isolate pooling preparation from the vision encoder and learned layers."""
    image_features = (
        torch.tensor([2, 4, 6, 8, 10, 20, 30, 40]).float().reshape(2, 2, 2, 1)
    )
    token_pooling = torch.tensor(
        [
            [[0, 3, 3, -1], [-1, -1, -1, -1]],
            [[0, 2, 2, -1], [1, 3, 2, 0]],
        ],
        dtype=torch.int32,
    )
    expected_query = torch.tensor(
        [6 if masked_average else 4.5, 0, 70 / 3 if masked_average else 17.5, 25]
    ).reshape(4, 1, 1)

    def pool(query, to_pool, *, attn_mask):
        expected_patches = (
            torch.tensor(
                [[2, 8, 8, 0], [0, 0, 0, 0], [10, 30, 30, 0], [20, 40, 30, 10]]
            )
            .float()
            .unsqueeze(-1)
        )
        torch.testing.assert_close(to_pool, expected_patches)
        torch.testing.assert_close(query, expected_query)
        if masked_average:
            torch.testing.assert_close(
                attn_mask, (token_pooling >= 0).reshape(4, 1, 1, 4)
            )
        else:
            assert attn_mask is None
        return query

    backbone = SimpleNamespace(
        device=image_features.device,
        dtype=image_features.dtype,
        adapter_config=SimpleNamespace(pooling_attention_mask=masked_average),
        encode_image=lambda images: images,
        image_pooling_2d=pool,
        image_projector=torch.nn.Identity(),
    )
    actual = Molmo2VisionBackbone.forward(backbone, image_features, token_pooling)
    torch.testing.assert_close(actual, expected_query[[0, 2, 3], 0])


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("max_crops", [1, 2])
@pytest.mark.parametrize("prompt_as_tokens", [False, True])
def test_crop_kwargs_match_processed_image_tokens(max_crops, prompt_as_tokens):
    """Crop counts and placeholders follow the actual processor output."""
    images = [Image.new("RGB", (1600, 1200)), Image.new("RGB", (400, 300))]
    kwargs = {"max_crops": max_crops}
    results = []
    for on_engine in (False, True):
        ctx = build_model_context(
            "allenai/Molmo2-8B",
            mm_processor_kwargs=kwargs if on_engine else None,
            limit_mm_per_prompt={"image": 2},
            mm_processor_cache_gb=0.01,
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        prompt = processor.dummy_inputs.get_dummy_text({"image": 2})
        if prompt_as_tokens:
            prompt = ctx.get_tokenizer().encode(prompt)
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
                image_tokens = item["image_tokens"].data
                pixels = item["pixel_values"].data
                assert isinstance(image_tokens, torch.Tensor)
                assert isinstance(pixels, torch.Tensor)
                expected = image_tokens.tolist()
                assert placeholder.length == len(expected)
                start = placeholder.offset
                assert (
                    result["prompt_token_ids"][start : start + len(expected)]
                    == expected
                )
                assert pixels.shape[0] <= max_crops + 1
        results.append(result["prompt_token_ids"])
    assert results[0] == results[1]
