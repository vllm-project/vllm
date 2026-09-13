# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.video_prune.evs import recompute_mrope_positions

TEXT_TOKEN_ID = 1
TIMESTAMP_TOKEN_ID = 2
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778
VIDEO_TOKEN_ID = 888
IMAGE_TOKEN_ID = 999


def _image_positions(
    rows: int,
    cols: int,
    metadata_channels: int = 4,
) -> torch.Tensor:
    """Metadata for one image, as if it were first in the sequence.

    Channels are ``[t, h, w, max_width]``; Qwen3 VL appends an all-zero
    ``is_vision`` channel for images.
    """
    num_tokens = rows * cols
    positions = torch.stack(
        [
            torch.zeros(num_tokens, dtype=torch.long),
            torch.arange(rows).view(-1, 1).expand(rows, cols).flatten(),
            torch.arange(cols).view(1, -1).expand(rows, cols).flatten(),
            torch.full((num_tokens,), max(rows, cols), dtype=torch.long),
        ]
    )
    if metadata_channels == 5:
        return torch.cat([positions, torch.zeros(1, num_tokens, dtype=torch.long)])
    return positions


def _image_position_slice(
    length: int,
    offset: int,
    width: int = 4,
) -> torch.Tensor:
    """The ``[offset, offset + length)`` window of a ``width``-wide image."""
    indices = torch.arange(offset, offset + length)
    return torch.stack(
        [
            torch.zeros_like(indices),
            indices // width,
            indices % width,
            torch.full_like(indices, width),
        ]
    )


def _video_positions(num_frames: int, tokens_per_frame: int = 4) -> torch.Tensor:
    """Five-channel metadata spanning a whole video replacement block.

    A video block covers text entries as well as media entries: each frame is
    laid out as ``timestamp timestamp VISION_START video... VISION_END``.
    Channels are ``[t, h, w, is_vision_start, is_vision]``.
    """
    frame_stride = 4 + tokens_per_frame
    block = torch.zeros(5, num_frames * frame_stride, dtype=torch.long)
    for frame_index in range(num_frames):
        offset = frame_index * frame_stride
        block[3, offset + 2] = 1
        for token in range(tokens_per_frame):
            column = offset + 3 + token
            block[4, column] = 1
            block[0, column] = frame_index
            block[1, column] = token // 2
            block[2, column] = token % 2
    return block


def _video_input_ids(num_frames: int, tokens_per_frame: int = 4) -> list[int]:
    input_ids = [TEXT_TOKEN_ID]
    for _ in range(num_frames):
        input_ids += [TIMESTAMP_TOKEN_ID, TIMESTAMP_TOKEN_ID, VISION_START_TOKEN_ID]
        input_ids += [VIDEO_TOKEN_ID] * tokens_per_frame
        input_ids += [VISION_END_TOKEN_ID]
    return input_ids + [TEXT_TOKEN_ID]


def _initial_positions(length: int) -> torch.Tensor:
    return torch.arange(length).view(1, -1).expand(3, -1).clone()


def _recompute(
    input_ids: list[int],
    multimodal_positions: list[torch.Tensor],
    num_computed_tokens: int,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    if positions is None:
        positions = _initial_positions(len(input_ids))
    return recompute_mrope_positions(
        input_ids=torch.tensor(input_ids),
        multimodal_positions=multimodal_positions,
        mrope_positions=positions,
        num_computed_tokens=num_computed_tokens,
        vision_start_token_id=VISION_START_TOKEN_ID,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )


def _single_image_input_ids(has_future_media: bool) -> list[int]:
    input_ids = [
        TEXT_TOKEN_ID,
        VISION_START_TOKEN_ID,
        *([IMAGE_TOKEN_ID] * 4),
        VISION_END_TOKEN_ID,
        TEXT_TOKEN_ID,
    ]
    if has_future_media:
        input_ids += [
            VISION_START_TOKEN_ID,
            *([IMAGE_TOKEN_ID] * 4),
            VISION_END_TOKEN_ID,
            TEXT_TOKEN_ID,
        ]
    return input_ids


@pytest.mark.parametrize(
    ("has_future_media", "expected_delta"),
    [(False, -2), (True, -6)],
    ids=["last-media", "future-media"],
)
@pytest.mark.parametrize("metadata_channels", [4, 5])
def test_chunk_starting_on_first_media_token(
    metadata_channels: int,
    has_future_media: bool,
    expected_delta: int,
):
    """A cursor on the first media token belongs to the preceding VISION_START.

    Answering "are we inside a media segment?" from a media token count cannot
    see this: no media token has been counted yet at a ``VISION_START | media``
    boundary. Searching forward instead either raises (no later marker exists)
    or remaps the block onto a later media item.
    """
    input_ids = _single_image_input_ids(has_future_media)

    positions, delta = _recompute(
        input_ids,
        [_image_positions(2, 2, metadata_channels)],
        num_computed_tokens=2,
    )

    expected = torch.tensor([[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]])
    torch.testing.assert_close(positions[:, 2:6], expected)
    assert delta == expected_delta


@pytest.mark.parametrize("metadata_channels", [4, 5])
@pytest.mark.parametrize("num_computed_tokens", [0, 2])
def test_two_media_items_in_one_chunk(
    metadata_channels: int,
    num_computed_tokens: int,
):
    """Each media item in a chunk is placed on its own span.

    Media token counts accumulate across items, so a second item can read as a
    continuation of the first and be written over its span. Only the first
    block of a call can continue a media item left half-written by an earlier
    chunk.
    """
    first = _image_positions(2, 3, metadata_channels)
    second = _image_positions(2, 2, metadata_channels)
    input_ids = [
        TEXT_TOKEN_ID,
        VISION_START_TOKEN_ID,
        *([IMAGE_TOKEN_ID] * 6),
        VISION_END_TOKEN_ID,
        TEXT_TOKEN_ID,
        VISION_START_TOKEN_ID,
        *([IMAGE_TOKEN_ID] * 4),
        VISION_END_TOKEN_ID,
        TEXT_TOKEN_ID,
    ]

    positions, _ = _recompute(
        input_ids,
        [first, second],
        num_computed_tokens=num_computed_tokens,
    )

    # Both items keep their own relative layout, offset by their own anchor.
    torch.testing.assert_close(positions[:, 2:8], first[0:3] + 2)
    torch.testing.assert_close(positions[:, 11:15], second[0:3] + 8)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 8, 16, 64, 4096])
def test_chunked_prefill_converges_on_full_prefill(chunk_size: int):
    """Splitting a prompt into chunks must not change the final positions.

    The loop cursor has to advance to the end of what was written, not by the
    number of entries written: the latter ignores the text between media items
    and so lags inside the previous item from the second item onwards.
    """
    tokens_per_image = 8
    input_ids = [TEXT_TOKEN_ID] * 3
    media_ranges = []
    for _ in range(22):
        input_ids.append(VISION_START_TOKEN_ID)
        media_start = len(input_ids)
        input_ids += [IMAGE_TOKEN_ID] * tokens_per_image
        media_ranges.append((media_start, len(input_ids)))
        input_ids += [VISION_END_TOKEN_ID, TEXT_TOKEN_ID, TEXT_TOKEN_ID]
    input_ids += [TEXT_TOKEN_ID] * 5

    initial_positions = _initial_positions(len(input_ids))
    full_positions, _ = _recompute(
        input_ids,
        [_image_position_slice(tokens_per_image, 0) for _ in media_ranges],
        num_computed_tokens=0,
        positions=initial_positions,
    )

    chunked_positions = initial_positions
    for chunk_start in range(0, len(input_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(input_ids))
        chunk_positions = []
        for media_start, media_end in media_ranges:
            overlap_start = max(chunk_start, media_start)
            overlap_end = min(chunk_end, media_end)
            if overlap_start < overlap_end:
                chunk_positions.append(
                    _image_position_slice(
                        overlap_end - overlap_start,
                        overlap_start - media_start,
                    )
                )

        chunked_positions, _ = _recompute(
            input_ids,
            chunk_positions,
            num_computed_tokens=chunk_start,
            positions=chunked_positions,
        )

    assert torch.equal(chunked_positions, full_positions)


@pytest.mark.parametrize("metadata_channels", [4, 5])
def test_empty_position_tensors_are_skipped(metadata_channels: int):
    """Qwen3 VL emits empty position tensors for fully pruned frames."""
    image = _image_positions(2, 2, metadata_channels)
    input_ids = _single_image_input_ids(has_future_media=False)

    positions, delta = _recompute(
        input_ids,
        [
            torch.zeros(metadata_channels, 0, dtype=torch.long),
            image,
            torch.zeros(metadata_channels, 0, dtype=torch.long),
        ],
        num_computed_tokens=0,
    )

    torch.testing.assert_close(positions[:, 2:6], image[0:3] + 2)
    assert delta == -2


def test_video_block_is_anchored_at_its_first_timestamp_token():
    """A fresh video block starts ahead of its VISION_START marker.

    Its metadata also covers the timestamp and marker entries, so the anchor
    sits before them rather than one past the marker.
    """
    input_ids = _video_input_ids(2)
    block = _video_positions(2)

    positions, _ = _recompute(input_ids, [block], num_computed_tokens=0)

    torch.testing.assert_close(positions[:, 1 : 1 + block.shape[1]], block[0:3] + 1)


def test_each_video_item_is_anchored_separately():
    """Two video items in one chunk do not share an anchor."""
    input_ids = _video_input_ids(1) + _video_input_ids(1)[1:]
    first = _video_positions(1)
    second = _video_positions(1)

    positions, _ = _recompute(input_ids, [first, second], num_computed_tokens=0)

    torch.testing.assert_close(positions[:, 1 : 1 + first.shape[1]], first[0:3] + 1)
    second_start = 1 + first.shape[1] + 1
    torch.testing.assert_close(
        positions[:, second_start : second_start + second.shape[1]],
        second[0:3] + positions[-1, second_start - 1] + 1,
    )


@pytest.mark.parametrize("num_computed_tokens", [1, 2, 3])
def test_chunk_starting_inside_the_leading_timestamps(num_computed_tokens: int):
    """Timestamp tokens are text, so a video segment can already be active.

    This pins existing behaviour rather than asserting convergence: the anchor
    a partial video block's entries are offset from is not recoverable from the
    slice alone, so the base shifts with the chunk boundary. Chunked video
    therefore does not reproduce the full-prefill result, unchanged here.
    """
    input_ids = _video_input_ids(1)
    block = _video_positions(1)
    remaining = block[:, num_computed_tokens - 1 :]

    positions, _ = _recompute(input_ids, [remaining], num_computed_tokens)

    written = positions[
        :, num_computed_tokens : num_computed_tokens + remaining.shape[1]
    ]
    torch.testing.assert_close(written, remaining[0:3] + num_computed_tokens)


def test_returns_early_once_every_media_token_is_computed():
    """No media work is left, so positions pass through untouched."""
    input_ids = _single_image_input_ids(has_future_media=False)
    initial_positions = _initial_positions(len(input_ids))

    positions, delta = _recompute(
        input_ids,
        [_image_positions(2, 2)],
        num_computed_tokens=6,
        positions=initial_positions,
    )

    assert torch.equal(positions, initial_positions)
    assert delta == 0


def test_no_media_positions_passes_through():
    input_ids = _single_image_input_ids(has_future_media=False)
    initial_positions = _initial_positions(len(input_ids))

    positions, delta = _recompute(input_ids, [], 0, positions=initial_positions)

    assert torch.equal(positions, initial_positions)
    assert delta == 0
