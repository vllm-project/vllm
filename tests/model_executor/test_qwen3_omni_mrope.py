# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    # get_mrope_input_positions returns CPU tensors (via torch.from_numpy).
    # Ensure the default device is CPU so the rest of the test tensors match.
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


TEXT_TOKEN_ID = 1
VISION_START_TOKEN_ID = 100
VISION_END_TOKEN_ID = 101
IMAGE_TOKEN_ID = 200
VIDEO_TOKEN_ID = 201
AUDIO_TOKEN_ID = 202
AUDIO_START_TOKEN_ID = 300
AUDIO_END_TOKEN_ID = 301


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass
class DummyConfig:
    position_id_per_seconds: int = 25
    vision_config: DummyVisionConfig = dataclasses.field(
        default_factory=DummyVisionConfig
    )


def make_model() -> Qwen3OmniMoeThinkerForConditionalGeneration:
    model = Qwen3OmniMoeThinkerForConditionalGeneration.__new__(
        Qwen3OmniMoeThinkerForConditionalGeneration  # type: ignore[type-abstract]  # protocol attrs unset
    )
    model.config = DummyConfig()
    return model


def _elem(data):
    return MultiModalFieldElem(
        data=torch.tensor(data),
        field=None,  # HACK.
    )


def make_image_feature(offset: int, grid_thw: tuple[int, int, int]):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem({"image_grid_thw": _elem(grid_thw)}),
        modality="image",
        identifier="DUMMY",
        mm_position=PlaceholderRange(
            offset=offset, length=int(torch.tensor(grid_thw).prod())
        ),
    )


def make_video_feature(
    offset: int, grid_thw: tuple[int, int, int], use_audio_in_video: bool = False
):
    data = {"video_grid_thw": _elem(grid_thw)}
    if use_audio_in_video:
        data["use_audio_in_video"] = _elem(True)
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(data),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(
            offset=offset, length=int(torch.tensor(grid_thw).prod())
        ),
    )


def make_audio_feature(offset: int, audio_feature_length: int):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {"audio_feature_lengths": _elem(audio_feature_length)}
        ),
        modality="audio",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=offset, length=0),
    )


def test_mrope_positions_image_at_prompt_end():
    """A prompt ending with <|vision_end|> must still produce seq_len positions."""
    model = make_model()
    # [<|vision_start|>, <|image_pad|> x 4, <|vision_end|>]
    input_tokens = (
        [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID]
    )
    mm_features = [make_image_feature(offset=1, grid_thw=(1, 2, 2))]

    positions, _ = model.get_mrope_input_positions(input_tokens, mm_features)

    assert positions.shape == (3, len(input_tokens))


def test_mrope_positions_audio_at_prompt_end():
    model = make_model()
    # audio_feature_length=16 -> 2 audio tokens.
    # [<|audio_start|>, <|audio_pad|> x 2, <|audio_end|>]
    input_tokens = [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * 2 + [AUDIO_END_TOKEN_ID]
    mm_features = [make_audio_feature(offset=1, audio_feature_length=16)]

    positions, _ = model.get_mrope_input_positions(input_tokens, mm_features)

    assert positions.shape == (3, len(input_tokens))


def test_mrope_positions_video_at_prompt_end():
    model = make_model()
    # [T, <|vision_start|>, <|video_pad|> x 8, <|vision_end|>]
    input_tokens = (
        [TEXT_TOKEN_ID]
        + [VISION_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * 8
        + [VISION_END_TOKEN_ID]
    )
    mm_features = [make_video_feature(offset=2, grid_thw=(2, 2, 2))]

    positions, _ = model.get_mrope_input_positions(input_tokens, mm_features)

    assert positions.shape == (3, len(input_tokens))


@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("trailing_tokens", [0, 3])
def test_mrope_positions_audio_in_video_boundaries(num_videos, trailing_tokens):
    """Interleaved BOS/EOS positions and decode delta match Qwen3-Omni HF."""
    model = make_model()
    video_tokens = (
        [VISION_START_TOKEN_ID, AUDIO_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * 4
        + [AUDIO_TOKEN_ID] * 2
        + [AUDIO_END_TOKEN_ID, VISION_END_TOKEN_ID]
    )
    input_tokens = [TEXT_TOKEN_ID] * 2
    mm_features = []
    expected = [torch.tensor([[0, 1]]).expand(3, -1)]
    video_positions = torch.tensor(
        [
            [2, 3, 4, 4, 4, 4, 4, 5, 6, 7],
            [2, 3, 4, 4, 5, 5, 4, 5, 6, 7],
            [2, 3, 4, 5, 4, 5, 4, 5, 6, 7],
        ]
    )
    for index in range(num_videos):
        offset = len(input_tokens) + 1
        mm_features.extend(
            [
                make_video_feature(offset, (1, 2, 2), use_audio_in_video=True),
                make_audio_feature(offset, audio_feature_length=16),
            ]
        )
        input_tokens.extend(video_tokens)
        expected.append(video_positions + index * 6)

    input_tokens.extend([TEXT_TOKEN_ID] * trailing_tokens)
    next_position = 2 + num_videos * 6
    expected.append((torch.arange(trailing_tokens) + next_position).expand(3, -1))
    positions, delta = model.get_mrope_input_positions(input_tokens, mm_features)

    torch.testing.assert_close(positions, torch.cat(expected, dim=1))
    assert delta == -4 * num_videos


def test_mrope_positions_image_then_text():
    """Media followed by text: positions must match the expected layout exactly.

    Before the fix this case passed the length guard but every position from
    the first <|image_pad|> onward was silently shifted by one slot.
    """
    model = make_model()
    # [T, T, <|vision_start|>, <|image_pad|> x 4, <|vision_end|>, T, T, T]
    input_tokens = (
        [TEXT_TOKEN_ID] * 2
        + [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * 4
        + [VISION_END_TOKEN_ID]
        + [TEXT_TOKEN_ID] * 3
    )
    mm_features = [make_image_feature(offset=3, grid_thw=(1, 2, 2))]

    positions, delta = model.get_mrope_input_positions(input_tokens, mm_features)

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 8],
            [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8],
        ]
    )
    torch.testing.assert_close(positions, expected)
    assert delta == 9 - len(input_tokens)
