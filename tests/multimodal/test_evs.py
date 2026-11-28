# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.video_prune.evs import recompute_mrope_positions

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "prefix",
    [
        [11, 777],
        [11, 777, 888, 778, 22, 777],
    ],
)
def test_mrope_chunk_starts_after_vision_start_without_timestamps(prefix):
    """A chunk starting at the media placeholder needs no later VISION_START."""
    vision_start_token_id = 777
    video_token_id = 888
    image_token_id = 999
    num_computed_tokens = len(prefix)
    input_ids = torch.tensor(prefix + [888, 888, 778, 12], dtype=torch.long)
    mrope_positions = torch.arange(input_ids.numel()).repeat(3, 1)
    # Qwen2.5-VL uses four channels: t, h, w, max_width.
    mm_positions = [torch.tensor([[0, 0], [0, 0], [0, 1], [2, 2]], dtype=torch.long)]

    positions, delta = recompute_mrope_positions(
        input_ids=input_ids,
        multimodal_positions=mm_positions,
        mrope_positions=mrope_positions,
        num_computed_tokens=num_computed_tokens,
        vision_start_token_id=vision_start_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )

    start = num_computed_tokens
    assert torch.equal(
        positions[:, start : start + 2],
        mm_positions[0][:3] + start,
    )
    assert torch.equal(
        positions[:, start + 2 :],
        torch.tensor([start + 2, start + 3]).repeat(3, 1),
    )
    assert delta == 0
