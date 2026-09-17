# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import pytest
import torch

from vllm.model_executor.models.keye_vl1_5 import KeyeVL1_5ForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 2


@dataclass
class DummyConfig:
    vision_config: DummyVisionConfig = field(default_factory=DummyVisionConfig)


def make_model(config: DummyConfig) -> KeyeVL1_5ForConditionalGeneration:
    model = object.__new__(KeyeVL1_5ForConditionalGeneration)
    model.config = config
    return model


def make_mm_feature(
    *,
    modality: str,
    offset: int,
    length: int,
    grid_thw: tuple[int, int, int] | list[tuple[int, int, int]],
    is_embed: list[bool] | None = None,
) -> MultiModalFeatureSpec:
    field_name = "image_grid_thw" if modality == "image" else "video_grid_thw"
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                field_name: MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality=modality,
        identifier="DUMMY",
        mm_position=PlaceholderRange(
            offset=offset,
            length=length,
            is_embed=None if is_embed is None else torch.tensor(is_embed),
        ),
    )


def test_get_mrope_input_positions_text_only():
    model = make_model(DummyConfig())

    positions, delta = model.get_mrope_input_positions(
        input_tokens=[11, 12, 13, 14, 15],
        mm_features=[],
    )

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ]
    )

    assert torch.equal(positions, expected)
    assert delta == 0


def test_get_mrope_input_positions_single_image():
    model = make_model(DummyConfig())
    mm_features = [
        make_mm_feature(
            modality="image",
            offset=1,
            length=4,
            grid_thw=(1, 4, 4),
        )
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=[10, 20, 21, 22, 23, 30, 31],
        mm_features=mm_features,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4],
            [0, 1, 1, 2, 2, 3, 4],
            [0, 1, 2, 1, 2, 3, 4],
        ]
    )

    assert torch.equal(positions, expected)
    assert delta == -2


def test_get_mrope_input_positions_video_uses_embed_ranges():
    model = make_model(DummyConfig())
    mm_features = [
        make_mm_feature(
            modality="video",
            offset=1,
            length=8,
            grid_thw=[(2, 4, 2)],
            is_embed=[False, False, True, True, False, False, True, True],
        )
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=[10, 101, 102, 20, 21, 103, 104, 30, 31, 40, 41],
        mm_features=mm_features,
    )

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 3, 5, 6, 7, 7, 9, 10],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3, 3, 5, 6, 7, 7, 9, 10],
        ]
    )

    assert torch.equal(positions, expected)
    assert delta == 0
