# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.model_executor.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
from vllm.model_executor.models.interfaces import supports_multimodal_pruning
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import compute_retained_tokens_count

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MODEL_ID = "nvidia/Cosmos3-Edge"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
LOCAL_MODEL_PATH = os.getenv("COSMOS3_EDGE_MODEL_PATH")


def _make_processor(*, video_pruning_rate: float | None = None):
    kwargs = {}
    if video_pruning_rate is not None:
        kwargs["video_pruning_rate"] = video_pruning_rate

    if LOCAL_MODEL_PATH:
        model_config = ModelConfig(
            LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2, "video": 1},
            **kwargs,
        )
    else:
        ctx = build_model_context(
            MODEL_ID,
            limit_mm_per_prompt={"image": 2, "video": 1},
            model_config_kwargs=kwargs,
        )
        model_config = ctx.model_config

    return MULTIMODAL_REGISTRY.create_processor(model_config)


@pytest.fixture(scope="module")
def processor():
    return _make_processor()


def _assert_image_outputs(processor, processed, num_images: int) -> None:
    mm_data = processed["mm_kwargs"].get_data()
    grid_thw = mm_data["image_grid_thw"]
    pixel_values = mm_data["pixel_values"]

    assert grid_thw.shape == (num_images, 3)
    assert pixel_values.shape[0] == int(grid_thw.prod(dim=-1).sum())

    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size
    expected_tokens = (grid_thw.prod(dim=-1) // merge_size**2).tolist()
    image_placeholders = processed["mm_placeholders"]["image"]

    assert len(image_placeholders) == num_images
    assert [placeholder.length for placeholder in image_placeholders] == (
        expected_tokens
    )

    image_token_id = processor.info.get_hf_processor().image_token_id
    assert processed["prompt_token_ids"].count(image_token_id) == sum(expected_tokens)


def _assert_video_outputs(processor, processed) -> None:
    mm_data = processed["mm_kwargs"].get_data()
    grid_thw = mm_data["video_grid_thw"]
    pixel_values = mm_data["pixel_values_videos"]

    assert grid_thw.shape == (1, 3)
    assert pixel_values.shape[0] == int(grid_thw.prod())
    assert len(processed["mm_placeholders"]["video"]) == 1

    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size
    expected_tokens = int(grid_thw.prod()) // merge_size**2
    video_token_id = processor.info.get_hf_config().video_token_id
    prompt_token_ids = processed["prompt_token_ids"]
    assert prompt_token_ids.count(video_token_id) == expected_tokens

    hf_processor = processor.info.get_hf_processor()
    expected_frame_wrappers = int(grid_thw[:, 0].sum())
    assert (
        prompt_token_ids.count(hf_processor.vision_start_token_id)
        == expected_frame_wrappers
    )
    assert (
        prompt_token_ids.count(hf_processor.vision_end_token_id)
        == expected_frame_wrappers
    )


def test_supports_multimodal_pruning() -> None:
    assert supports_multimodal_pruning(Cosmos3EdgeForConditionalGeneration)
    assert hasattr(Cosmos3EdgeForConditionalGeneration, "recompute_mrope_positions")
    assert hasattr(Cosmos3EdgeForConditionalGeneration, "_postprocess_video_embeds_evs")


@pytest.mark.parametrize("num_images", [1, 2])
def test_process_images(
    processor,
    image_assets: ImageTestAssets,
    num_images: int,
) -> None:
    images = [asset.pil_image for asset in image_assets[:num_images]]
    processed = processor(
        IMAGE_PLACEHOLDER * num_images,
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )

    _assert_image_outputs(processor, processed, num_images)


def test_process_video(processor) -> None:
    video_asset = VideoAsset(name="baby_reading", num_frames=8)
    video = (video_asset.np_ndarrays, video_asset.metadata)
    processed = processor(
        VIDEO_PLACEHOLDER,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={},
    )

    _assert_video_outputs(processor, processed)


@pytest.mark.parametrize("video_pruning_rate", [0.5, 0.75])
def test_process_video_with_evs(video_pruning_rate: float) -> None:
    """Processor-side EVS should shrink video token count and keep timestamps."""
    processor = _make_processor(video_pruning_rate=video_pruning_rate)
    video_asset = VideoAsset(name="baby_reading", num_frames=8)
    video = (video_asset.np_ndarrays, video_asset.metadata)
    processed = processor(
        VIDEO_PLACEHOLDER,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={},
    )

    mm_data = processed["mm_kwargs"].get_data()
    grid_thw = mm_data["video_grid_thw"]
    assert grid_thw.shape == (1, 3)

    timestamps = mm_data["timestamps"]
    assert len(timestamps) == 1
    assert len(timestamps[0]) == int(grid_thw[0, 0])

    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size
    tokens_per_frame = int(grid_thw[0, 1:].prod()) // merge_size**2
    num_frames = int(grid_thw[0, 0])
    expected_kept = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame,
        num_frames=num_frames,
        q=video_pruning_rate,
    )
    full_tokens = tokens_per_frame * num_frames
    video_token_id = processor.info.get_hf_config().video_token_id
    actual_video_tokens = processed["prompt_token_ids"].count(video_token_id)

    assert actual_video_tokens == expected_kept
    assert actual_video_tokens < full_tokens
