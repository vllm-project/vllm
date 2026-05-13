# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import pytest

from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from ....conftest import IMAGE_ASSETS, VIDEO_ASSETS
from ....utils import create_new_process_for_each_test
from .vlm_utils.builders import sample_frames_with_video_metadata


@dataclass
class VitCudagraphTestConfig:
    model: str
    modalities: list[str] = field(default_factory=lambda: ["image", "video"])
    image_prompt: str | None = None
    video_prompt: str | None = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_tokens: int = 64
    max_num_seqs: int = 2
    num_video_frames: int = 16
    needs_video_metadata: bool = False
    vllm_runner_kwargs: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)


def params_with_marks(
    configs: dict[str, VitCudagraphTestConfig],
) -> list[pytest.param]:
    return [
        pytest.param(model_id, marks=cfg.marks) for model_id, cfg in configs.items()
    ]


def qwen_vl_chat_template(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
    "qwen2_5_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=False,
        marks=[pytest.mark.core_model],
    ),
    "qwen3_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "qwen3_5": VitCudagraphTestConfig(
        model="Qwen/Qwen3.5-0.8B",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "qwen2_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=False,
        marks=[pytest.mark.core_model],
    ),
}


def get_compilation_config():
    return {
        "cudagraph_mm_encoder": True,
        "encoder_cudagraph_max_vision_items_per_batch": 1,
        "encoder_cudagraph_max_frames_per_batch": 16,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@create_new_process_for_each_test()
def test_vit_cudagraph_image(model_id, vllm_runner, image_assets):
    config = MODEL_CONFIGS[model_id]

    if "image" not in config.modalities:
        pytest.skip(f"{model_id} does not support the image modality.")

    image_prompts = IMAGE_ASSETS.prompts(
        {
            "stop_sign": config.image_prompt,  # type: ignore[typeddict-item]
            "cherry_blossom": config.image_prompt,  # type: ignore[typeddict-item]
        }
    )
    images = [[asset.pil_image] for asset in image_assets]

    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        compilation_config=get_compilation_config(),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            image_prompts, config.max_tokens, images=images
        )

        # Basic validation that we got a response
        assert len(outputs) == 2
        output_ids, output_text = outputs[0]

        # Ensure we got some output
        assert len(output_ids) > 0
        assert len(output_text) > 0

        # Ensure the output is a string
        assert isinstance(output_text, str)


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@create_new_process_for_each_test()
def test_vit_cudagraph_video(model_id, vllm_runner, video_assets):
    config = MODEL_CONFIGS[model_id]

    if "video" not in config.modalities:
        pytest.skip(f"{model_id} does not support the video modality")

    video_prompts = VIDEO_ASSETS.prompts(
        {
            "baby_reading": config.video_prompt,  # type: ignore[typeddict-item]
        }
    )
    if config.needs_video_metadata:
        sampled_vids = [
            sample_frames_with_video_metadata(
                (asset.np_ndarrays, asset.metadata), config.num_video_frames
            )
            for asset in video_assets
        ]
    else:
        sampled_vids = [
            sample_frames_from_video(asset.np_ndarrays, config.num_video_frames)
            for asset in video_assets
        ]
    videos = [sampled_vids[0]]

    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"video": 1},
        compilation_config=get_compilation_config(),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            video_prompts, config.max_tokens, videos=videos
        )

        # Basic validation that we got a response
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]

        # Ensure we got some output
        assert len(output_ids) > 0
        assert len(output_text) > 0

        # Ensure the output is a string
        assert isinstance(output_text, str)
