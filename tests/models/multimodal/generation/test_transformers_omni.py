# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import (
    Gemma3nForConditionalGeneration,
    Gemma4ForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.envs import disable_envs_cache

from ....conftest import HfRunner, VllmRunner
from ....utils import large_gpu_mark
from ...utils import check_logprobs_close
from .vlm_utils import model_utils

AUDIO_ASSET = AudioAsset("mary_had_lamb")
IMAGE_ASSET = ImageAsset("stop_sign")
VIDEO_ASSET = VideoAsset("baby_reading", num_frames=4)

OMNI_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-Omni-3B": {
        "prompt": (
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            "<|vision_bos|><|IMAGE|><|vision_eos|>"
            "<|vision_bos|><|VIDEO|><|vision_eos|>"
            "Describe what you see and hear.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "auto_cls": Qwen2_5OmniThinkerForConditionalGeneration,
        "vllm_runner_kwargs": {
            "gpu_memory_utilization": 0.85,
        },
    },
    "google/gemma-3n-E2B-it": {
        "prompt": (
            "<bos><start_of_turn>user\n"
            "<audio_soft_token><image_soft_token>"
            "Describe what you see and hear.<end_of_turn>\n<start_of_turn>model\n"
        ),
        "auto_cls": Gemma3nForConditionalGeneration,
        "modalities": ("audio", "image"),
        "patch_hf_runner": model_utils.gemma3n_patch_hf_runner,
        "vllm_runner_kwargs": {
            "gpu_memory_utilization": 0.8,
        },
    },
    "google/gemma-4-E2B-it": {
        "prompt": (
            "<bos><|turn>user\n"
            "<|audio|><|image|><|video|>Describe what you see and hear.<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        ),
        "auto_cls": Gemma4ForConditionalGeneration,
        "vllm_runner_kwargs": {
            "gpu_memory_utilization": 0.85,
            # FlashInfer cannot do gemma4's 512 head size, see
            # https://github.com/vllm-project/vllm/issues/40677
            "attention_backend": "TRITON_ATTN",
        },
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "prompt": (
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe what you see and hear.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "auto_cls": Qwen3OmniMoeThinkerForConditionalGeneration,
        "min_gb": 80,
    },
}


@pytest.fixture(autouse=True)
def use_spawn_for_omni_models(monkeypatch):
    """Single-process workaround for V1 fork safety deadlock issue
    (vllm-project/vllm/issues/17676). Running multiple omni models together
    under pytest can cause (possibly flaky) hangs, so they are grouped under
    the same config. Using VLLM_WORKER_MULTIPROC_METHOD=spawn avoids the
    deadlock and allows worker processes to terminate cleanly, and release
    GPU memory between test runs until the issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(model_id, marks=large_gpu_mark(min_gb=settings["min_gb"]))
        if "min_gb" in settings
        else model_id
        for model_id, settings in OMNI_MODEL_SETTINGS.items()
    ],
)
def test_transformers_omni_generation(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model_id: str,
):
    """Every modality the model takes in one request, against the same model on HF."""
    settings = OMNI_MODEL_SETTINGS[model_id]
    modalities = settings.get("modalities", ("audio", "image", "video"))
    assets = {
        "audios": [AUDIO_ASSET.audio_and_sample_rate],
        "images": [IMAGE_ASSET.pil_image],
        "videos": [(VIDEO_ASSET.np_ndarrays, VIDEO_ASSET.metadata)],
    }
    mm_data = {f"{m}s": assets[f"{m}s"] for m in modalities}

    with vllm_runner(
        model_id,
        model_impl="transformers",
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={m: 1 for m in modalities},
        **{"max_model_len": 4096, **settings.get("vllm_runner_kwargs", {})},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [settings["prompt"]],
            128,
            num_logprobs=10,
            **mm_data,
        )
        # M-RoPE runs for every request, so a prompt with no items must work too
        vllm_model.generate_greedy(["What is 2+2?"], 8)

    with hf_runner(
        model_id, dtype="bfloat16", auto_cls=settings["auto_cls"]
    ) as hf_model:
        if "video" in modalities:
            hf_model = model_utils.qwen3_vl_patch_hf_runner(hf_model)
        if (patch := settings.get("patch_hf_runner")) is not None:
            hf_model = patch(hf_model)
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [settings["prompt"]],
            128,
            num_logprobs=10,
            **mm_data,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
