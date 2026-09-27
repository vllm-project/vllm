# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import pytest
import torch

from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLForConditionalGeneration,
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
class DummyVisionConfig:
    spatial_merge_size: int = 2
    patch_size: int = 14


@dataclass
class DummyConfig:
    image_token_id: int = 151655
    video_token_id: int = 151654
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_config: DummyVisionConfig = field(default_factory=DummyVisionConfig)


def make_model(config: DummyConfig) -> PaddleOCRVLForConditionalGeneration:
    model = object.__new__(PaddleOCRVLForConditionalGeneration)
    model.config = config
    return model


def make_mm_feature(
    *,
    offset: int,
    length: int,
    image_grid_thw: tuple[int, int, int],
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "image_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(image_grid_thw),
                    field=None,
                ),
            }
        ),
        modality="image",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_get_mrope_input_positions_text_only():
    model = make_model(DummyConfig())
    input_tokens = [11, 12, 13, 14, 15]
    positions, delta = model.get_mrope_input_positions(
        input_tokens=input_tokens,
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
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    t, h, w = 1, 2, 2
    num_image_tokens = t * h * w

    input_tokens = (
        [10]
        + [model.config.vision_start_token_id]
        + [model.config.image_token_id] * num_image_tokens
        + [model.config.vision_end_token_id]
        + [30, 31]
    )

    mm_features = [
        make_mm_feature(
            offset=2,  # 1 (text) + 1 (vision_start)
            length=num_image_tokens,
            image_grid_thw=(t, h * spatial_merge_size, w * spatial_merge_size),
        )
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=mm_features,
    )

    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5, 6],
            [0, 1, 2, 2, 3, 3, 4, 5, 6],
            [0, 1, 2, 3, 2, 3, 4, 5, 6],
        ]
    )

    assert torch.equal(positions, expected)
    expected_delta = (positions.max().item() + 1) - len(input_tokens)
    assert delta == expected_delta


def test_get_mrope_input_positions_multiple_images():
    model = make_model(DummyConfig())
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    t1, h1, w1 = 1, 2, 2
    num1 = t1 * h1 * w1

    t2, h2, w2 = 1, 1, 3
    num2 = t2 * h2 * w2

    input_tokens = (
        [10]
        + [model.config.vision_start_token_id]
        + [model.config.image_token_id] * num1
        + [model.config.vision_end_token_id]
        + [20, 21]
        + [model.config.vision_start_token_id]
        + [model.config.image_token_id] * num2
        + [model.config.vision_end_token_id]
        + [30]
    )

    mm_features = [
        make_mm_feature(
            offset=2,
            length=num1,
            image_grid_thw=(t1, h1 * spatial_merge_size, w1 * spatial_merge_size),
        ),
        make_mm_feature(
            offset=2 + num1 + 1 + 2 + 1,
            length=num2,
            image_grid_thw=(t2, h2 * spatial_merge_size, w2 * spatial_merge_size),
        ),
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=mm_features,
    )

    assert positions.shape == (3, 15)
    assert not torch.equal(positions[:, 2:6], torch.arange(4).expand(3, 4) + 2)
    assert not torch.equal(positions[:, 10:13], torch.arange(3).expand(3, 3) + 10)


def test_get_mrope_input_positions_image_at_start():
    model = make_model(DummyConfig())
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    t, h, w = 1, 2, 2
    num_tokens = t * h * w

    input_tokens = (
        [model.config.vision_start_token_id]
        + [model.config.image_token_id] * num_tokens
        + [model.config.vision_end_token_id]
        + [10, 11]
    )

    mm_features = [
        make_mm_feature(
            offset=1,  # start token at index 0
            length=num_tokens,
            image_grid_thw=(t, h * spatial_merge_size, w * spatial_merge_size),
        )
    ]

    positions, delta = model.get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=mm_features,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4, 5],
            [0, 1, 1, 2, 2, 3, 4, 5],
            [0, 1, 2, 1, 2, 3, 4, 5],
        ]
    )

    assert torch.equal(positions, expected)
