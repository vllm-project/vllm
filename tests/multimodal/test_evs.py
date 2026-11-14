# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    recompute_mrope_positions,
)

TEXT, VISION_START, VISION_END, VIDEO, IMAGE = 1, 777, 778, 888, 999


def _media_only_positions(
    grid_thw: list[int], retained: list[int] | None = None
) -> torch.Tensor:
    """4-channel entry of one media item, as Qwen2.5-VL appends it."""
    positions = compute_mrope_for_media(grid_thw, spatial_merge_size=1)
    if retained is not None:
        positions = positions[retained]
    return positions.T


# Two images and a video pruned to its first frame plus two later tokens.
# The first image has fewer tokens than the text before it.
MEDIA = [
    (IMAGE, _media_only_positions([1, 2, 3])),
    (IMAGE, _media_only_positions([1, 2, 2])),
    (VIDEO, _media_only_positions([2, 2, 3], retained=[0, 1, 2, 3, 4, 5, 7, 10])),
]


def _build_prompt() -> tuple[list[int], list[tuple[int, int]], torch.Tensor]:
    """Token ids, media spans and M-RoPE positions of a prompt holding `MEDIA`."""
    input_ids: list[int] = []
    spans = []
    expected = []
    next_position = 0

    def add_text(token_ids: list[int]) -> None:
        nonlocal next_position
        input_ids.extend(token_ids)
        end = next_position + len(token_ids)
        expected.append(torch.arange(next_position, end).expand(3, -1))
        next_position = end

    add_text([TEXT] * 8)
    for token_id, mm_pos in MEDIA:
        add_text([VISION_START])
        spans.append((len(input_ids), len(input_ids) + mm_pos.shape[1]))
        input_ids.extend([token_id] * mm_pos.shape[1])
        expected.append(mm_pos[:3] + next_position)
        next_position += int(mm_pos[3, 0])
        add_text([VISION_END, TEXT, TEXT])
    return input_ids, spans, torch.cat(expected, dim=1)


INPUT_IDS, SPANS, EXPECTED = _build_prompt()


@pytest.mark.parametrize(
    "chunk_starts",
    [
        pytest.param([], id="one-chunk"),
        pytest.param([SPANS[1][0], SPANS[2][0]], id="on-first-token-of-each-media"),
        pytest.param([SPANS[0][0] + 2], id="inside-media-with-more-media-after"),
        pytest.param([SPANS[0][1] + 1], id="on-text-between-media"),
        pytest.param(list(range(1, len(INPUT_IDS))), id="one-token-chunks"),
    ],
)
def test_recompute_mrope_positions_media_only_chunked_prefill(chunk_starts: list[int]):
    """4-channel positions do not depend on where prefill chunks start."""
    positions = torch.arange(len(INPUT_IDS)).expand(3, -1)
    bounds = [0, *chunk_starts, len(INPUT_IDS)]
    for start, end in zip(bounds, bounds[1:]):
        # Mirrors the model runner: the part of each media item in the chunk.
        chunk = [
            mm_pos[:, max(start, lo) - lo : min(end, hi) - lo]
            for (lo, hi), (_, mm_pos) in zip(SPANS, MEDIA)
            if max(start, lo) < min(end, hi)
        ]
        if not chunk:
            continue
        positions, delta = recompute_mrope_positions(
            torch.tensor(INPUT_IDS),
            chunk,
            positions,
            num_computed_tokens=start,
            vision_start_token_id=VISION_START,
            image_token_id=IMAGE,
            video_token_id=VIDEO,
        )

    assert torch.equal(positions, EXPECTED)
    assert delta == EXPECTED.max() + 1 - len(INPUT_IDS)
