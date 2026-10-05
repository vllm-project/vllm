# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import AutoModelForImageTextToText

from vllm.assets.video import VideoAsset
from vllm.envs import disable_envs_cache

from ....conftest import HfRunner, VllmRunner
from ...utils import check_logprobs_close
from .vlm_utils import model_utils

VIDEO_ASSET = VideoAsset("baby_reading", num_frames=8)

VIDEO_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "prompt": (
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": {
        "prompt": (
            "<|im_start|>user <video>\nDescribe this video.<|im_end|>"
            "<|im_start|>assistant\n"
        ),
    },
    "CohereLabs/North-Micro-Vision-Instruct": {
        "prompt": (
            "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            "<|VISION_START|><|VIDEO_PAD|><|VISION_END|>Describe this video."
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
}


@pytest.fixture(autouse=True)
def use_spawn_for_video_models(monkeypatch):
    """Single-process workaround for V1 fork safety deadlock issue
    (vllm-project/vllm/issues/17676). Running multiple video models together
    under pytest can cause (possibly flaky) hangs, so they are grouped under
    the same config. Using VLLM_WORKER_MULTIPROC_METHOD=spawn avoids the
    deadlock and allows worker processes to terminate cleanly, and release
    GPU memory between test runs until the issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("model_id", list(VIDEO_MODEL_SETTINGS))
def test_transformers_video_generation(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model_id: str,
):
    prompt = VIDEO_MODEL_SETTINGS[model_id]["prompt"]
    video = (VIDEO_ASSET.np_ndarrays, VIDEO_ASSET.metadata)

    with vllm_runner(
        model_id,
        model_impl="transformers",
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt], 128, num_logprobs=10, videos=[video]
        )

    with hf_runner(
        model_id, dtype="bfloat16", auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_model = model_utils.qwen3_vl_patch_hf_runner(hf_model)
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [prompt], 128, num_logprobs=10, videos=[video]
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
