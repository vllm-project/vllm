# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.model_executor.models.cosmos3_edge import (
    blockmajor_to_raster,
    patch_merging_by_param,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MODEL_ID = "nvidia/Cosmos3-Edge"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
LOCAL_MODEL_PATH = os.getenv("COSMOS3_EDGE_MODEL_PATH")


@pytest.fixture(scope="module")
def processor():
    if LOCAL_MODEL_PATH:
        model_config = ModelConfig(
            LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2, "video": 1},
        )
    else:
        ctx = build_model_context(
            MODEL_ID,
            limit_mm_per_prompt={"image": 2, "video": 1},
        )
        model_config = ctx.model_config

    return MULTIMODAL_REGISTRY.create_processor(model_config)


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


def _pack_blockmajor(grid_thw: list[list[int]], merge_size: int) -> torch.Tensor:
    """Pack raster patch ids the way the processor's ``patchify()`` does.

    Mirrors ``image_processing_cosmos3_edge.patchify``: reshape into
    ``merge_size x merge_size`` blocks and emit block-major, per frame.
    """
    streams = []
    offset = 0
    for t, h, w in grid_thw:
        for _ in range(t):
            ids = torch.arange(offset, offset + h * w).reshape(h, w)
            streams.append(
                ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            offset += h * w
    return torch.cat(streams)


@pytest.mark.parametrize(
    "grid_thw,merge_size",
    [
        # w // merge_size not in {1, merge_size}: the permutation is not
        # self-inverse here, so a reversed direction cannot pass.
        ([[1, 4, 6]], 2),
        ([[1, 6, 10]], 2),
        ([[1, 6, 9]], 3),
        # multiple images and a multi-frame video in one packed batch
        ([[1, 4, 6], [1, 6, 10]], 2),
        ([[3, 4, 6]], 2),
        ([[1, 4, 6], [2, 6, 10], [1, 4, 4]], 2),
    ],
)
def test_blockmajor_to_raster_recovers_raster_order(
    grid_thw: list[list[int]],
    merge_size: int,
) -> None:
    """Reordering a block-major stream must yield raster order."""
    spatial_shapes = torch.tensor(
        [[h, w] for t, h, w in grid_thw for _ in range(t)],
        dtype=torch.int64,
    )
    packed = _pack_blockmajor(grid_thw, merge_size).unsqueeze(-1).float()

    reordered = blockmajor_to_raster(packed, spatial_shapes, merge_size)

    expected = torch.arange(packed.shape[0], dtype=torch.float32).unsqueeze(-1)
    assert torch.equal(reordered, expected)


def test_blockmajor_to_raster_is_noop_without_merging() -> None:
    packed = torch.randn(12, 3)
    spatial_shapes = torch.tensor([[3, 4]], dtype=torch.int64)

    assert blockmajor_to_raster(packed, spatial_shapes, 1) is packed


def test_blockmajor_to_raster_rejects_patch_count_mismatch() -> None:
    packed = torch.randn(20, 3)
    spatial_shapes = torch.tensor([[4, 6]], dtype=torch.int64)

    with pytest.raises(ValueError, match="do not match spatial_shapes"):
        blockmajor_to_raster(packed, spatial_shapes, 2)


@pytest.mark.parametrize("grid_thw", [[[1, 4, 6]], [[2, 6, 10], [1, 4, 4]]])
def test_reorder_then_merge_matches_processor_block_order(
    grid_thw: list[list[int]],
) -> None:
    """The projector must receive each 2x2 block as one contiguous group.

    ``patch_merging_by_param`` on the reordered stream has to reproduce what a
    plain reshape of the *un*-reordered block-major stream gives, which is what
    HF's ``Cosmos3EdgePatchMerger`` consumes.
    """
    merge_size = 2
    hidden_size = 8
    spatial_shapes = torch.tensor(
        [[h, w] for t, h, w in grid_thw for _ in range(t)],
        dtype=torch.int64,
    )
    order = _pack_blockmajor(grid_thw, merge_size)
    packed = torch.randn(order.shape[0], hidden_size)

    reordered = blockmajor_to_raster(packed, spatial_shapes, merge_size)
    merged = patch_merging_by_param(
        reordered, torch.tensor(grid_thw, dtype=torch.int64), merge_size
    )

    expected = packed.reshape(-1, merge_size * merge_size * hidden_size)
    assert torch.equal(merged, expected)
