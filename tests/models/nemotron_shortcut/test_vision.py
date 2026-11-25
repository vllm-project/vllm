# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vision tower's video tubelets: `video_patch_conv` embeds a tubelet of T
frames, and the rest of the encoder runs as for one frame of the same size."""

import pytest
import torch
from transformers.models.pixtral import PixtralVisionConfig

from vllm.models.nemotron_shortcut.nvidia.vision import NemotronShortcutVisionModel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from .utils import build

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _inference(dist_init):
    with torch.inference_mode():
        yield


def test_tubelet_of_a_repeated_frame_encodes_as_the_frame() -> None:
    """With `video_patch_conv` set to half the image patch conv for each of
    T = 2 frames, a tubelet of one frame twice embeds as that frame does, so
    the encoder gives the frame's features, at the frame's RoPE positions."""
    config = PixtralVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=256,
        patch_size=16,
    )
    with set_default_torch_dtype(torch.bfloat16):
        tower = build(lambda: NemotronShortcutVisionModel(config, 2))
    assert tower.video_patch_conv is not None
    half = tower.patch_conv.weight / 2
    tower.video_patch_conv.weight.copy_(torch.cat([half, half], dim=1))

    frames = torch.randn(2, 3, 64, 96, device="cuda", dtype=torch.bfloat16)
    images = tower(list(frames))
    tubelets = tower.encode_tubelets([torch.cat([frame, frame]) for frame in frames])
    assert [t.shape for t in tubelets] == [i.shape for i in images] == [(24, 64)] * 2
    for tubelet, image in zip(tubelets, images):
        torch.testing.assert_close(tubelet, image, atol=2e-2, rtol=2e-2)
