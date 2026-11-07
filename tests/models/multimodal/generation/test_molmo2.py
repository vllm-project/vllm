# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end regression test for Molmo2 fps-mode video sampling."""

import pytest

from vllm import SamplingParams
from vllm.assets.base import get_vllm_public_assets

from ....conftest import VllmRunner
from ....multimodal.utils import create_video_from_image

MODEL = "allenai/Molmo2-4B"


@pytest.mark.core_model
def test_fps_mode_serves_ntsc_video_like_integer_fps(
    vllm_runner: type[VllmRunner], tmp_path
):
    """A 29.97 fps clip must reach the model like the same clip at 30 fps.

    With ``frame_sample_mode="fps"`` the molmo2 video loader truncated 29.97 to
    29 and failed the request with ``sampling_fps=2 must divide video_fps=29``.
    """
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )
    videos = {}
    for name, fps in (("integer", 30.0), ("ntsc", 30000 / 1001)):
        videos[name] = tmp_path / f"{name}.mp4"
        create_video_from_image(
            str(image_path), str(videos[name]), num_frames=120, fps=fps
        )

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path=str(tmp_path),
        media_io_kwargs={
            "video": {"num_frames": 32, "frame_sample_mode": "fps", "sampling_fps": 2}
        },
    ) as vllm_model:
        patch_id = vllm_model.llm.model_config.hf_config.image_patch_id
        num_video_patches = {}
        for name, path in videos.items():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": path.as_uri()}},
                        {"type": "text", "text": "What is in this video?"},
                    ],
                }
            ]
            (output,) = vllm_model.llm.chat(
                messages, sampling_params=SamplingParams(temperature=0, max_tokens=8)
            )
            num_video_patches[name] = output.prompt_token_ids.count(patch_id)

    assert num_video_patches["integer"] > 0
    assert num_video_patches["ntsc"] == num_video_patches["integer"]
