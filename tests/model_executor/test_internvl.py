# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.internvl import InternVLChatModel

pytestmark = pytest.mark.skip_global_cleanup


def make_model() -> InternVLChatModel:
    model = object.__new__(InternVLChatModel)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(image_size=448))
    return model


def test_image_embeds_are_not_consumed_as_video():
    image_embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(
        image_embeds=image_embeds,
        pixel_values_flat_video=torch.zeros(2, 3, 448, 448),
        video_num_patches=torch.tensor([1, 1]),
        video_token_id=torch.tensor([151667]),
    )

    assert torch.equal(modalities["images"]["data"], image_embeds)
    assert modalities["videos"]["type"] == "pixel_values_videos"
