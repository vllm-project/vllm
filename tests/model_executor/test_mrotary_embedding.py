from unittest.mock import Mock

import torch
import pytest

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

IMAGE = 101
VIDEO = 102
AUDIO = 103
VISION_START = 201
VISION_END = 202
AUDIO_START = 203
AUDIO_END = 204


SPATIAL_MERGE_SIZE_2 = 2

def create_image(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [IMAGE] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

def create_video(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [VIDEO] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

def create_audio(audio_feature_length: int):
    audio_token_num = (((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1)
    return [AUDIO_START] + [AUDIO] * audio_token_num + [AUDIO_END]

def create_video_with_audio(t: int, h: int, w: int, merge_size: int, audio_feature_length: int):
    audio_token_num = (((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1)
    return [VISION_START, AUDIO_START] + [VIDEO] * t * (h // merge_size) * (w // merge_size) + [VIDEO] * audio_token_num + [AUDIO_END, VISION_END]

vl_test_cases = [
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
    },
]

@pytest.mark.parametrize("test_case", vl_test_cases)
def test_vl_get_input_positions_and_delta_correctness(
    test_case,
):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    tokens_per_second = test_case.get("tokens_per_second", 1.0)

    hf_config = Mock()
    hf_config.image_token_id = IMAGE
    hf_config.video_token_id = VIDEO
    hf_config.vision_start_token_id = VISION_START
    hf_config.vision_end_token_id = VISION_END

    hf_config.vision_config = Mock()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size
    hf_config.vision_config.tokens_per_second = tokens_per_second

    hf_config.vision_config.rope_scaling = {
        "mrope_section" : [16, 24, 24],
    }
    hf_config.thinker_config = None

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
    
    input_positions_torch, mrope_position_delta_torch = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        use_numba=False,
    )

    input_positions_numba, mrope_position_delta_numba = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        use_numba=True,
    )
    assert input_positions_torch.dtype == input_positions_numba.dtype
    assert input_positions_torch.shape == input_positions_numba.shape

    assert torch.equal(input_positions_torch, input_positions_numba)
    assert mrope_position_delta_torch == mrope_position_delta_numba


omni_test_cases = [
    # text and image and video (with aduio)
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
                "t": 12,
                "h": 8 * SPATIAL_MERGE_SIZE_2,
                "w": 12 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
                "audio_feature_length": 4096,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "tokens_per_second": 1.0,
        "use_audio_with_video": True,
        "seconds_per_chunk": 4.0,
    },
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
    },
    # text and audio
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "audio",
                "audio_feature_length": 144,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
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
    },
]

@pytest.mark.parametrize("test_case", omni_test_cases)
def test_omni_get_input_positions_and_delta_correctness(
    test_case,
):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    use_audio_with_video = test_case.get("use_audio_with_video", False)
    tokens_per_second = test_case.get("tokens_per_second", 1.0)
    seconds_per_chunk = test_case.get("seconds_per_chunk", 4.0)

    hf_config = Mock()
   
    hf_config.thinker_config = Mock()
    hf_config.thinker_config.image_token_index = IMAGE
    hf_config.thinker_config.video_token_index = VIDEO
    hf_config.thinker_config.audio_token_index = AUDIO
    hf_config.thinker_config.vision_start_token_id = VISION_START
    hf_config.thinker_config.vision_end_token_id = VISION_END
    hf_config.thinker_config.audio_start_token_id = AUDIO_START
    hf_config.thinker_config.audio_end_token_id = AUDIO_END

    hf_config.thinker_config.seconds_per_chunk = seconds_per_chunk
    
    hf_config.thinker_config.vision_config = Mock()
    hf_config.thinker_config.vision_config.spatial_merge_size = spatial_merge_size
    hf_config.thinker_config.vision_config.tokens_per_second = tokens_per_second

    hf_config.thinker_config.rope_scaling = {
        "mrope_section" : [16, 24, 24],
    }

    input_tokens = []
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []
    audio_feature_lengths = []

    for item in input:
        if item["type"] == "tokens":
            input_tokens.extend(item["tokens"])
        elif item["type"] == "image":
            input_tokens.extend(create_image(item["t"], item["h"], item["w"], spatial_merge_size))
            image_grid_thw.append([item["t"], item["h"], item["w"]])
        elif item["type"] == "audio":
            input_tokens.extend(create_audio(item["audio_feature_length"]))
            audio_feature_lengths.append(item["audio_feature_length"])
        elif item["type"] == "video":
            if use_audio_with_video:
                input_tokens.extend(create_video_with_audio(item["t"], item["h"], item["w"], spatial_merge_size, item["audio_feature_length"]))
                audio_feature_lengths.append(item["audio_feature_length"])
            else:
                input_tokens.extend(create_video(item["t"], item["h"], item["w"], spatial_merge_size))
            video_grid_thw.append([item["t"], item["h"], item["w"]])
            second_per_grid_ts.append(item["second_per_grid_t"])

    audio_feature_lengths = torch.tensor(audio_feature_lengths, dtype=torch.int64)
    
    input_positions_torch, mrope_position_delta_torch = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        audio_feature_lengths=audio_feature_lengths,
        use_audio_in_video=use_audio_with_video,
        use_numba=False,
    )

    input_positions_numba, mrope_position_delta_numba = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        audio_feature_lengths=audio_feature_lengths,
        use_audio_in_video=use_audio_with_video,
        use_numba=True,
    )
    assert input_positions_torch.dtype == input_positions_numba.dtype
    assert input_positions_torch.shape == input_positions_numba.shape

    assert torch.equal(input_positions_torch, input_positions_numba)
    assert mrope_position_delta_torch == mrope_position_delta_numba
