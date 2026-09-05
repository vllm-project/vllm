# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grid-kwargs forwarding of `MultiModalMixin.get_mrope_input_positions`.

Grid kwargs for absent modalities must not be passed as explicit `None` to
HF `get_rope_index`: not every signature accepts them (HunYuanVL has no
`video_grid_thw` parameter).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.transformers.multimodal import MultiModalMixin
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

pytestmark = pytest.mark.cpu_test

_ABSENT = object()


def _image_feature(grid_thw, offset=2, length=4):
    # `gather_kwargs` only needs `key in data` and `data[key].data`.
    data = {"image_grid_thw": SimpleNamespace(data=grid_thw)}
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier="image-0",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


class _HunYuanStyleModel:
    """`get_rope_index` signature like HunYuanVL: no `video_grid_thw`."""

    def __init__(self):
        self.kwargs = None

    def get_rope_index(
        self, input_ids, mm_token_type_ids, image_grid_thw=None, attention_mask=None
    ):
        self.kwargs = {
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
        }
        return torch.zeros(4, 1, input_ids.shape[1], dtype=torch.int64), torch.zeros(
            1, 1, dtype=torch.int64
        )


class _QwenStyleModel:
    """`get_rope_index` signature like Qwen2-VL; sentinels detect omissions."""

    def __init__(self):
        self.kwargs = None

    def get_rope_index(self, input_ids, image_grid_thw=_ABSENT, video_grid_thw=_ABSENT):
        self.kwargs = {
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }
        return torch.zeros(3, 1, input_ids.shape[1], dtype=torch.int64), torch.zeros(
            1, 1, dtype=torch.int64
        )


def test_omits_absent_grid_kwargs():
    # Pristine main fails here: HunYuanVL's signature has no `video_grid_thw`
    # parameter, but it is passed unconditionally.
    model = _HunYuanStyleModel()

    grid = torch.tensor([1, 4, 4])
    positions, delta = MultiModalMixin.get_mrope_input_positions(
        SimpleNamespace(model=model), [1] * 10, [_image_feature(grid)]
    )

    assert torch.equal(model.kwargs["image_grid_thw"], grid.unsqueeze(0))
    assert torch.equal(
        model.kwargs["mm_token_type_ids"],
        torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]),
    )
    assert positions.shape == (4, 10)
    assert delta == 0


def test_forwards_present_image_grid():
    model = _QwenStyleModel()

    MultiModalMixin.get_mrope_input_positions(
        SimpleNamespace(model=model), [1] * 8, [_image_feature(torch.tensor([1, 2, 2]))]
    )

    assert torch.equal(model.kwargs["image_grid_thw"], torch.tensor([[1, 2, 2]]))
    assert model.kwargs["video_grid_thw"] is _ABSENT
