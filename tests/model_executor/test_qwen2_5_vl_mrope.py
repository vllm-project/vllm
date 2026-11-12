# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import pytest
import torch

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
    recompute_mrope_positions,
)

pytestmark = pytest.mark.skip_global_cleanup

TEXT, VISION_START, VISION_END, VIDEO, IMAGE = 1, 777, 778, 888, 999

# Both grids are taller than wide, so the width is not their largest position.
VIDEO_GRID_THW = (3, 3, 2)
IMAGE_GRID_THW = (1, 3, 2)
VIDEO_PRUNING_RATE = 0.5
# The first frame is always kept. The other kept cells depend on the content.
VIDEO_RETAINED = [0, 1, 2, 3, 4, 5, 8, 13, 17]


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass
class DummyConfig:
    vision_config: DummyVisionConfig = field(default_factory=DummyVisionConfig)


def _build_prompt(layout: list[str]):
    """Build `TEXT TEXT <media> TEXT ... TEXT TEXT` with a pruned video.

    Returns the prompt, its features, and the positions that the encoder
    returns for the kept tokens of each media item.
    """
    input_tokens = [TEXT, TEXT]
    mm_features = []
    mm_positions = []
    for modality in layout:
        if modality == "video":
            grid_thw, token_id, retained = VIDEO_GRID_THW, VIDEO, VIDEO_RETAINED
            t, h, w = grid_thw
            assert len(retained) == compute_retained_tokens_count(
                h * w, t, VIDEO_PRUNING_RATE
            )
        else:
            grid_thw, token_id = IMAGE_GRID_THW, IMAGE
            retained = list(range(grid_thw[1] * grid_thw[2]))

        input_tokens.append(VISION_START)
        mm_features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem(
                    {
                        f"{modality}_grid_thw": MultiModalFieldElem(
                            data=torch.tensor(grid_thw), field=None
                        )
                    }
                ),
                modality=modality,
                identifier="DUMMY",
                mm_position=PlaceholderRange(
                    offset=len(input_tokens), length=len(retained)
                ),
            )
        )
        mm_positions.append(
            compute_mrope_for_media(torch.tensor(grid_thw), spatial_merge_size=1)[
                retained
            ].T
        )
        input_tokens += [token_id] * len(retained) + [VISION_END, TEXT]
    input_tokens += [TEXT, TEXT]
    return input_tokens, mm_features, mm_positions


@pytest.mark.parametrize(
    "layout",
    [["video"], ["video", "image"], ["image", "video"]],
    ids=lambda layout: ",".join(layout),
)
def test_get_mrope_input_positions_pruned_video(layout: list[str]):
    model = object.__new__(
        Qwen2_5_VLForConditionalGeneration  # type: ignore[type-abstract]  # protocol attrs unset
    )
    model.config = DummyConfig()
    model.is_multimodal_pruning_enabled = True

    input_tokens, mm_features, mm_positions = _build_prompt(layout)
    positions, delta = model.get_mrope_input_positions(input_tokens, mm_features)

    # One position per prompt token, although the video grid has more cells.
    assert positions.shape == (3, len(input_tokens))

    # Text and generated tokens are not recomputed when the media items are a
    # prefix cache hit, so they must already have the positions that the
    # recomputation gives them.
    input_ids = torch.tensor(input_tokens)
    recomputed, recomputed_delta = recompute_mrope_positions(
        input_ids, mm_positions, positions, 0, VISION_START, IMAGE, VIDEO
    )
    is_text = (input_ids != IMAGE) & (input_ids != VIDEO)
    assert torch.equal(positions[:, is_text], recomputed[:, is_text])
    assert delta == recomputed_delta
