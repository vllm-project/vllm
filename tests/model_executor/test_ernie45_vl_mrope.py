# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
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


@pytest.mark.parametrize("frame_count, expected_tokens", [(1, 4), (3, 8), (4, 8)])
def test_video_token_estimate_covers_padded_frame_pairs(frame_count, expected_tokens):
    from vllm.model_executor.models.ernie45_vl import Ernie4_5_VLProcessingInfo

    info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=2),
            spatial_conv_size=2,
            temporal_conv_size=2,
        ),
        ctx=SimpleNamespace(
            model_config=SimpleNamespace(trust_remote_code=False),
            get_merged_mm_kwargs=lambda kwargs: kwargs,
        ),
    )
    _, tokens = Ernie4_5_VLProcessingInfo._get_vision_info(
        info,
        image_width=8,
        image_height=8,
        num_frames=frame_count,
        do_resize=False,
        image_processor=SimpleNamespace(size={}),
        mm_kwargs={},
    )
    assert tokens == expected_tokens


@pytest.mark.parametrize("frame_count", [1, 3, 4])
@pytest.mark.parametrize("kind", ["numpy", "torch", "list"])
@pytest.mark.parametrize("supports_metadata", [False, True])
def test_video_frame_pairs_preserve_metadata(
    monkeypatch, frame_count, kind, supports_metadata
):
    from vllm.model_executor.models.ernie45_vl import Ernie4_5VLMultiModalProcessor
    from vllm.multimodal.processing.processor import (
        BaseMultiModalProcessor,
        HFMultiModalInputs,
    )

    frames = np.arange(frame_count * 12).reshape(frame_count, 2, 2, 3)
    if kind == "torch":
        frames = torch.from_numpy(frames)
    elif kind == "list":
        frames = list(frames)
    metadata = {
        "frames_indices": list(range(frame_count)),
        "fps": 2.0,
        "duration": 10.0,
        "total_num_frames": 20,
    }
    inputs = HFMultiModalInputs(
        {"text": "video", "videos": [(frames, metadata)]}, {}, {}
    )
    monkeypatch.setattr(
        BaseMultiModalProcessor, "_get_hf_mm_inputs", lambda *args: inputs
    )
    processor = object.__new__(Ernie4_5VLMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace(
            supports_video_metadata=supports_metadata
        )
    )
    result = processor._get_hf_mm_inputs(None, {})
    video = result.hf_data["videos"][0]
    padded, padded_metadata = video if supports_metadata else (video, None)
    expected_count = frame_count + frame_count % 2
    assert len(padded) == expected_count
    np.testing.assert_array_equal(np.asarray(padded)[:frame_count], np.asarray(frames))
    np.testing.assert_array_equal(np.asarray(padded)[-1], np.asarray(frames)[-1])
    assert len(frames) == frame_count
    assert metadata["frames_indices"] == list(range(frame_count))
    if padded_metadata is not None:
        assert padded_metadata["frames_indices"] == list(range(frame_count)) + (
            [frame_count - 1] if frame_count % 2 else []
        )
        assert padded_metadata["fps"] == metadata["fps"]
        assert padded_metadata["total_num_frames"] == 20


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
    model = object.__new__(
        Ernie4_5_VLMoeForConditionalGeneration  # type: ignore[type-abstract]  # protocol attrs unset
    )
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
