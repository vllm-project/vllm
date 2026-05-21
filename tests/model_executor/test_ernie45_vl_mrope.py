# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.ernie45_vl import (
    Ernie4_5_VLMoeForConditionalGeneration,
)
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
class DummyConfig:
    spatial_conv_size: int = 2
    temporal_conv_size: int = 2


def make_model(config: DummyConfig) -> Ernie4_5_VLMoeForConditionalGeneration:
    model = object.__new__(Ernie4_5_VLMoeForConditionalGeneration)
    model.config = config
    return model


def make_mm_feature(
    *,
    modality: str,
    offset: int,
    length: int,
    grid_thw: tuple[int, int, int],
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
        mm_position=PlaceholderRange(offset=offset, length=length),
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


def test_get_mrope_input_positions_interleaved_image_and_video():
    model = make_model(DummyConfig())
    mm_features = [
        make_mm_feature(
            modality="image",
            offset=1,
            length=4,
            grid_thw=(1, 4, 4),
        ),
        make_mm_feature(
            modality="video",
            offset=7,
            length=2,
            grid_thw=(2, 4, 2),
        ),
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=[10, 20, 21, 22, 23, 30, 31, 40, 41, 50, 51],
        mm_features=mm_features,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4, 5, 5, 7, 8],
            [0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 1, 2, 3, 4, 5, 5, 7, 8],
        ]
    )

    assert torch.equal(positions, expected)
    assert delta == -2
