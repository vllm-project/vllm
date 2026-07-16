# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.evs import recompute_mrope_positions

IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778

IMAGE_POSITIONS = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [2, 2, 2, 2],
    ]
)
EXPECTED_IMAGE_POSITIONS = torch.tensor(
    [
        [2, 2, 2, 2],
        [2, 2, 3, 3],
        [2, 3, 2, 3],
    ]
)


def _image_positions(metadata_channels: int) -> torch.Tensor:
    if metadata_channels == 5:
        return torch.cat([IMAGE_POSITIONS, torch.zeros(1, 4, dtype=torch.long)])
    return IMAGE_POSITIONS


def _recompute(
    input_ids: torch.LongTensor,
    multimodal_positions: list[torch.Tensor],
    num_computed_tokens: int,
) -> tuple[torch.LongTensor, int]:
    initial_positions = torch.arange(input_ids.numel()).expand(3, -1).clone()
    return recompute_mrope_positions(
        input_ids=input_ids,
        multimodal_positions=multimodal_positions,
        mrope_positions=initial_positions,
        num_computed_tokens=num_computed_tokens,
        vision_start_token_id=VISION_START_TOKEN_ID,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )


@pytest.mark.parametrize(
    ("has_future_media", "expected_delta"),
    [(False, -2), (True, -6)],
    ids=["last-media", "future-media"],
)
@pytest.mark.parametrize("metadata_channels", [4, 5])
def test_recompute_mrope_at_image_start_after_vision_start(
    metadata_channels: int,
    has_future_media: bool,
    expected_delta: int,
):
    input_tokens = [
        1,
        VISION_START_TOKEN_ID,
        *([IMAGE_TOKEN_ID] * 4),
        VISION_END_TOKEN_ID,
        2,
    ]
    if has_future_media:
        input_tokens.extend(
            [
                VISION_START_TOKEN_ID,
                *([IMAGE_TOKEN_ID] * 4),
                VISION_END_TOKEN_ID,
                3,
            ]
        )

    positions, delta = _recompute(
        torch.tensor(input_tokens),
        [_image_positions(metadata_channels)],
        num_computed_tokens=2,
    )

    torch.testing.assert_close(positions[:, 2:6], EXPECTED_IMAGE_POSITIONS)
    assert delta == expected_delta
