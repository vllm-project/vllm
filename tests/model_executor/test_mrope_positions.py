# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.mrope_positions import (
    mrope_assign_next_input_positions, mrope_get_input_positions_and_delta,
    mrope_get_next_input_positions_tensor)

IMAGE = 101
VIDEO = 102
AUDIO = 103
VISION_START = 201
VISION_END = 202
AUDIO_START = 203
AUDIO_END = 204

SPATIAL_MERGE_SIZE_2 = 2


def create_image(t: int, h: int, w: int, merge_size: int):
    return [
        VISION_START
    ] + [IMAGE] * t * (h // merge_size) * (w // merge_size) + [VISION_END]


def create_video(t: int, h: int, w: int, merge_size: int):
    return [
        VISION_START
    ] + [VIDEO] * t * (h // merge_size) * (w // merge_size) + [VISION_END]


def create_audio(audio_feature_length: int):
    audio_token_num = (((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1)
    return [AUDIO_START] + [AUDIO] * audio_token_num + [AUDIO_END]


def create_video_with_audio(num_t: int, num_h: int, num_w: int,
                            merge_size: int, audio_feature_length: int,
                            t_ntoken_per_chunk: int, tokens_per_grid_t: float):
    audio_token_num = (((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1)
    added_audio_token_num = 0

    ret = [VISION_START, AUDIO_START]
    next_chunk_t = t_ntoken_per_chunk

    for t in range(num_t):
        video_t = int(t * tokens_per_grid_t)

        # audio tokens
        if video_t >= next_chunk_t:
            next_chunk_t += t_ntoken_per_chunk
            if added_audio_token_num < audio_token_num:
                chunked_audio_token_num = min(
                    t_ntoken_per_chunk,
                    audio_token_num - added_audio_token_num)
                ret.extend([AUDIO] * chunked_audio_token_num)
                added_audio_token_num += chunked_audio_token_num

        # video tokens
        ret.extend([VIDEO] * (num_h // merge_size * num_w // merge_size))

    # remaining audio tokens
    if added_audio_token_num < audio_token_num:
        ret.extend([AUDIO] * (audio_token_num - added_audio_token_num))

    return ret + [AUDIO_END, VISION_END]


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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
        "tokens_per_second":
        1.0,
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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
        "tokens_per_second":
        1.0,
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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
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


def make_vl_hf_config(spatial_merge_size=2, tokens_per_second=2):
    hf_config = Mock()
    hf_config.image_token_id = IMAGE
    hf_config.video_token_id = VIDEO
    hf_config.vision_start_token_id = VISION_START
    hf_config.vision_end_token_id = VISION_END

    hf_config.vision_config = Mock()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size
    hf_config.vision_config.tokens_per_second = tokens_per_second

    hf_config.vision_config.rope_scaling = {
        "mrope_section": [16, 24, 24],
    }
    hf_config.thinker_config = None

    return hf_config


@pytest.mark.parametrize("test_case", vl_test_cases)
def test_vl_get_input_positions_and_delta_correctness(test_case):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    tokens_per_second = test_case.get("tokens_per_second", 1.0)

    hf_config = make_vl_hf_config(spatial_merge_size, tokens_per_second)

    input_tokens = []
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []

    for item in input:
        if item["type"] == "tokens":
            input_tokens.extend(item["tokens"])
        elif item["type"] == "image":
            input_tokens.extend(
                create_image(item["t"], item["h"], item["w"],
                             spatial_merge_size))
            image_grid_thw.append([item["t"], item["h"], item["w"]])
        elif item["type"] == "video":
            input_tokens.extend(
                create_video(item["t"], item["h"], item["w"],
                             spatial_merge_size))
            video_grid_thw.append([item["t"], item["h"], item["w"]])
            second_per_grid_ts.append(item["second_per_grid_t"])

    input_positions_torch, mrope_position_delta_torch = \
        mrope_get_input_positions_and_delta(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_numba=False,
        )

    input_positions_numba, mrope_position_delta_numba = \
        mrope_get_input_positions_and_delta(
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
                "h": 2 * SPATIAL_MERGE_SIZE_2,
                "w": 4 * SPATIAL_MERGE_SIZE_2,
            },
            {
                "type": "video",
                "t": 4,
                "h": 2 * SPATIAL_MERGE_SIZE_2,
                "w": 4 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
                "audio_feature_length": 50,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
        "use_audio_with_video":
        True,
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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
        "tokens_per_second":
        1.0,
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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
        "tokens_per_second":
        1.0,
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
                "h": 2 * SPATIAL_MERGE_SIZE_2,
                "w": 3 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
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
        "spatial_merge_size":
        SPATIAL_MERGE_SIZE_2,
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


def make_omni_hf_config(
    spatial_merge_size=2,
    tokens_per_second=25,
    seconds_per_chunk=2.0,
):
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

    vision_config = Mock()
    hf_config.thinker_config.vision_config = vision_config
    vision_config.spatial_merge_size = spatial_merge_size
    vision_config.tokens_per_second = tokens_per_second

    hf_config.thinker_config.text_config = Mock()
    hf_config.thinker_config.text_config.rope_scaling = {
        "mrope_section": [16, 24, 24],
    }

    return hf_config


@pytest.mark.parametrize("test_case", omni_test_cases)
def test_omni_get_input_positions_and_delta_correctness(test_case):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    use_audio_with_video = test_case.get("use_audio_with_video", False)
    tokens_per_second = test_case.get("tokens_per_second", 25)
    seconds_per_chunk = test_case.get("seconds_per_chunk", 2.0)

    hf_config = make_omni_hf_config(
        spatial_merge_size,
        tokens_per_second,
        seconds_per_chunk,
    )

    t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)

    input_tokens = []
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []
    audio_feature_lengths = []

    for item in input:
        if item["type"] == "tokens":
            input_tokens.extend(item["tokens"])
        elif item["type"] == "image":
            input_tokens.extend(
                create_image(item["t"], item["h"], item["w"],
                             spatial_merge_size))
            image_grid_thw.append([item["t"], item["h"], item["w"]])
        elif item["type"] == "audio":
            input_tokens.extend(create_audio(item["audio_feature_length"]))
            audio_feature_lengths.append(item["audio_feature_length"])
        elif item["type"] == "video":
            if use_audio_with_video:
                tokens_per_grid_t = tokens_per_second * item[
                    "second_per_grid_t"]
                input_tokens.extend(
                    create_video_with_audio(item["t"], item["h"], item["w"],
                                            spatial_merge_size,
                                            item["audio_feature_length"],
                                            t_ntoken_per_chunk,
                                            tokens_per_grid_t))
                audio_feature_lengths.append(item["audio_feature_length"])
            else:
                input_tokens.extend(
                    create_video(item["t"], item["h"], item["w"],
                                 spatial_merge_size))
            video_grid_thw.append([item["t"], item["h"], item["w"]])
            second_per_grid_ts.append(item["second_per_grid_t"])

    audio_feature_lengths = torch.tensor(audio_feature_lengths,
                                         dtype=torch.int64)

    input_positions_torch, mrope_position_delta_torch = \
        mrope_get_input_positions_and_delta(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            audio_feature_lengths=audio_feature_lengths,
            use_audio_in_video=use_audio_with_video,
            use_numba=False,
        )

    input_positions_numba, mrope_position_delta_numba = \
        mrope_get_input_positions_and_delta(
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


@pytest.mark.parametrize("is_omni, modality", [
    (True, "image_grid_thw"),
    (True, "video_grid_thw"),
    (True, "audio_feature_lengths"),
    (False, "image_grid_thw"),
    (False, "video_grid_thw"),
])
def test_missing_mm_item_error(is_omni, modality):
    hf_config = make_omni_hf_config() if is_omni else make_vl_hf_config()
    input_tokens = [1, 2, 3, 4]
    image_grid_thw: list[list[int]] = []
    video_grid_thw: list[list[int]] = []
    second_per_grid_ts: list[float] = []
    audio_feature_lengths: list[int] = []
    if modality == "image_grid_thw":
        input_tokens.extend(
            [VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, VISION_END])
    elif modality == "video_grid_thw":
        if is_omni:
            input_tokens.extend(
                [VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, AUDIO, VISION_END])
        else:
            input_tokens.extend(
                [VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, VISION_END])
    elif modality == "audio_feature_lengths":
        input_tokens.extend(
            [AUDIO_START, AUDIO, AUDIO, AUDIO, AUDIO, AUDIO_END])

    with pytest.raises(ValueError) as exc_info:
        mrope_get_input_positions_and_delta(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            audio_feature_lengths=torch.tensor(audio_feature_lengths,
                                               dtype=torch.float64),
            use_audio_in_video=is_omni,
            use_numba=True,
        )

    assert f"{modality}[0] is missing" in str(exc_info.value)


@pytest.mark.parametrize("is_omni, modality", [
    (True, "image_grid_thw"),
    (True, "video_grid_thw"),
    (True, "audio_feature_lengths"),
    (False, "image_grid_thw"),
    (False, "video_grid_thw"),
])
def test_tokens_out_of_bound_error(is_omni, modality):
    hf_config = make_omni_hf_config() if is_omni else make_vl_hf_config()
    input_tokens = [1, 2, 3, 4]
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []
    audio_feature_lengths = []
    if modality == "image_grid_thw":
        input_tokens.extend(
            [VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, VISION_END])
        image_grid_thw.append([1, 8, 8])
    elif modality == "video_grid_thw":
        video_grid_thw.append([1, 8, 8])
        if is_omni:
            audio_feature_lengths.append(1000)
            input_tokens.extend(
                [VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, AUDIO, VISION_END])
        else:
            input_tokens.extend(
                [VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, VISION_END])
        second_per_grid_ts.append(1.0)
    elif modality == "audio_feature_lengths":
        audio_feature_lengths.append(1000)
        input_tokens.extend(
            [AUDIO_START, AUDIO, AUDIO, AUDIO, AUDIO, AUDIO_END])

    with pytest.raises(ValueError) as exc_info:
        mrope_get_input_positions_and_delta(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            audio_feature_lengths=torch.tensor(audio_feature_lengths,
                                               dtype=torch.float64),
            use_audio_in_video=is_omni,
            use_numba=True,
        )
    assert f"input_tokens out of bounds while processing {modality}[0]" in str(
        exc_info.value)


@pytest.mark.parametrize("is_omni, modality", [
    (True, "image_grid_thw"),
    (True, "video_grid_thw"),
    (True, "audio_feature_lengths"),
    (False, "image_grid_thw"),
    (False, "video_grid_thw"),
])
def test_unused_mm_items_error(is_omni, modality):
    hf_config = make_omni_hf_config() if is_omni else make_vl_hf_config()
    input_tokens = [1, 2, 3, 4]
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []
    audio_feature_lengths = []
    if modality == "image_grid_thw":
        image_grid_thw.append([1, 4, 4])
        image_grid_thw.append([1, 4, 4])
        input_tokens.extend(
            [VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, VISION_END])
    elif modality == "video_grid_thw":
        video_grid_thw.append([1, 4, 4])
        video_grid_thw.append([1, 4, 4])
        if is_omni:
            audio_feature_lengths.append(16)
            audio_feature_lengths.append(16)
            input_tokens.extend([
                VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, AUDIO, AUDIO, AUDIO,
                AUDIO, VISION_END
            ])
        else:
            input_tokens.extend(
                [VISION_START, VIDEO, VIDEO, VIDEO, VIDEO, VISION_END])
        second_per_grid_ts.append(1.0)
        second_per_grid_ts.append(1.0)
    elif modality == "audio_feature_lengths":
        audio_feature_lengths.append(16)
        audio_feature_lengths.append(16)
        input_tokens.extend(
            [AUDIO_START, AUDIO, AUDIO, AUDIO, AUDIO, AUDIO_END])

    with pytest.raises(ValueError) as exc_info:
        mrope_get_input_positions_and_delta(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            audio_feature_lengths=torch.tensor(audio_feature_lengths,
                                               dtype=torch.float64),
            use_audio_in_video=is_omni,
            use_numba=True,
        )

    assert f"{modality} has 1 unused item" in str(exc_info.value)


@pytest.mark.parametrize(
    "mrope_position_delta, context_len, seq_len, expected_output", [
        (0, 0, 1, [[0], [0], [0]]),
        (0, 5, 7, [[5, 6], [5, 6], [5, 6]]),
        (-10, 160, 163, [[150, 151, 152], [150, 151, 152], [150, 151, 152]]),
        (-10, 200, 201, [[190], [190], [190]]),
    ])
def test_mrope_get_next_input_positions_tensor(mrope_position_delta,
                                               context_len, seq_len,
                                               expected_output):
    input_positions = mrope_get_next_input_positions_tensor(
        mrope_position_delta=mrope_position_delta,
        context_len=context_len,
        seq_len=seq_len,
    )

    assert torch.equal(input_positions,
                       torch.tensor(expected_output, dtype=torch.int64))


@pytest.mark.parametrize(
    "mrope_position_delta, out_offset, context_len, seq_len, expected_output",
    [
        (0, 0, 0, 1, [[0], [0], [0]]),
        (0, 1, 5, 7, [[0, 5, 6], [0, 5, 6], [0, 5, 6]]),
        (-10, 2, 160, 163, [[0, 0, 150, 151, 152], [0, 0, 150, 151, 152],
                            [0, 0, 150, 151, 152]]),
        (-10, 4, 200, 201, [[0, 0, 0, 0, 190], [0, 0, 0, 0, 190],
                            [0, 0, 0, 0, 190]]),
    ])
def test_mrope_assign_next_input_positions(mrope_position_delta, out_offset,
                                           context_len, seq_len,
                                           expected_output):
    out = torch.zeros((3, out_offset + seq_len - context_len),
                      dtype=torch.int64)
    out_np = out.numpy()
    mrope_assign_next_input_positions(
        out=out_np,
        out_offset=out_offset,
        mrope_position_delta=mrope_position_delta,
        context_len=context_len,
        num_new_tokens=seq_len - context_len,
    )

    assert torch.equal(out, torch.tensor(expected_output, dtype=torch.int64))
