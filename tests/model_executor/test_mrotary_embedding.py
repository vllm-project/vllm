from unittest.mock import Mock

import numpy as np
import torch
import pytest

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

IMAGE = 101
VIDEO = 102
VISION_START = 103
VISION_END = 104

SPATIAL_MERGE_SIZE_2 = 2

def create_image(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [IMAGE] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

def create_video(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [VIDEO] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

test_cases = [
    # text and image and video
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "image",
                "t": 1,
                "h": 4 * SPATIAL_MERGE_SIZE_2,
                "w": 6 * SPATIAL_MERGE_SIZE_2,
            },
            {
                "type": "video",
                "t": 4,
                "h": 8 * SPATIAL_MERGE_SIZE_2,
                "w": 12 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "tokens_per_second": 1.0,
        "groundtruth_input_positions": torch.tensor([
            [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 25, 26, 27, 28, 29, 30],
            [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 25, 26, 27, 28, 29, 30],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  5,  6,  7,  8,  9, 10,  5,  6,  7,  8,  9, 10,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ], dtype=torch.int64),
        "groundtruth_delta": -390,
    },
    # text and image
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "image",
                "t": 1,
                "h": 2 * SPATIAL_MERGE_SIZE_2,
                "w": 2 * SPATIAL_MERGE_SIZE_2,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "tokens_per_second": 1.0,
        "groundtruth_input_positions": torch.tensor([
            [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  5,  6,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  5,  6,  7,  8,  9, 10, 11, 12],
        ], dtype=torch.int64),
        "groundtruth_delta": -2,
    },
    # text and video
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "video",
                "t": 4,
                "h": 6 * SPATIAL_MERGE_SIZE_2,
                "w": 8 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "groundtruth_input_positions": torch.tensor([
            [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8, 13, 14, 15, 16, 17, 18],
            [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 15, 16, 17, 18],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ], dtype=torch.int64),
        "groundtruth_delta": -184,
    },
    # text only
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "groundtruth_input_positions": torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=torch.int64),
        "groundtruth_delta": 0,
    },
]

@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize("use_numba", [True, False])
def test_mrope_get_input_positions_and_delta_correctness(
    test_case,
    use_numba,
):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    tokens_per_second = test_case.get("tokens_per_second", 1.0)
    groundtruth_input_positions = test_case.get("groundtruth_input_positions")
    groundtruth_delta = test_case.get("groundtruth_delta")

    hf_config = Mock()
    hf_config.image_token_id = IMAGE
    hf_config.video_token_id = VIDEO
    hf_config.vision_start_token_id = VISION_START
    hf_config.vision_end_token_id = VISION_END

    hf_config.vision_config = Mock()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size
    hf_config.vision_config.tokens_per_second = tokens_per_second

    input_tokens = []
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []

    for item in input:
        if item["type"] == "tokens":
            input_tokens.extend(item["tokens"])
        elif item["type"] == "image":
            input_tokens.extend(create_image(item["t"], item["h"], item["w"], spatial_merge_size))
            image_grid_thw.append([item["t"], item["h"], item["w"]])
        elif item["type"] == "video":
            input_tokens.extend(create_video(item["t"], item["h"], item["w"], spatial_merge_size))
            video_grid_thw.append([item["t"], item["h"], item["w"]])
            second_per_grid_ts.append(item["second_per_grid_t"])
    
    if len(image_grid_thw) > 0:
        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.int64)
    else:
        image_grid_thw = torch.empty((0, 3), dtype=torch.int64)
    
    if len(video_grid_thw) > 0:
        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.int64)
    else:
        video_grid_thw = torch.empty((0, 3), dtype=torch.int64)

    input_positions, mrope_position_delta = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        use_numba=use_numba,
    )

    assert input_positions.dtype == groundtruth_input_positions.dtype
    assert input_positions.shape == groundtruth_input_positions.shape

    assert torch.equal(input_positions, groundtruth_input_positions)
    assert mrope_position_delta == groundtruth_delta
