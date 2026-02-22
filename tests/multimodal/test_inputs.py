# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import PlaceholderRange


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, 5),
        (torch.tensor([True, True, True, True, True]), 5),
        (torch.tensor([False, False, False, False, False]), 0),
        (torch.tensor([True, False, True, False, True]), 3),
        (torch.tensor([True]), 1),
    ],
)
def test_placeholder_range_get_num_embeds(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)
    assert pr.get_num_embeds() == expected


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, None),
        (
            torch.tensor([False, True, False, True, True]),
            torch.tensor([0, 1, 1, 2, 3]),
        ),
        (torch.tensor([True, True, True]), torch.tensor([1, 2, 3])),
    ],
)
def test_placeholder_range_embeds_cumsum(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)

    if expected is None:
        assert pr.embeds_cumsum is None
        return

    assert torch.equal(pr.embeds_cumsum, expected)
    # cached_property should return the same object on repeated access
    assert pr.embeds_cumsum is pr.embeds_cumsum
