# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.assets.image import ImageAsset
from vllm.platforms import current_platform

from ....conftest import VIDEO_ASSETS
from ....utils import multi_gpu_test
from ...utils import check_logprobs_close
from .vlm_utils.builders import sample_frames_with_video_metadata

models = ["Qwen/Qwen2.5-VL-3B-Instruct"]
target_dtype = "bfloat16"

VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def qwen2_5_vl_chat_template(*query):
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{''.join(query)}<|im_end|><|im_start|>assistant\n"  # noqa: E501


VIDEO_PROMPTS = VIDEO_ASSETS.prompts(
    {
        "baby_reading": qwen2_5_vl_chat_template(
            VIDEO_PLACEHOLDER,
            "Describe this video with a short sentence ",
            "(no more than 20 words)",
        ),
    }
)


WINDOW_ATTN_IMAGE_PROMPT = qwen2_5_vl_chat_template(
    IMAGE_PLACEHOLDER,
    "Describe the image.",
)
IMAGE_ONLY_LIMIT_MM_PER_PROMPT = {"image": 1, "video": 0}


def _window_attention_regression_image():
    # image from regression issue: https://github.com/vllm-project/vllm/issues/15122
    image = ImageAsset("hato").pil_image
    return image.resize((image.width // 2, image.height // 2))


def _encoder_cudagraph_config(*, max_vision_items: int) -> dict:
    return {
        "cudagraph_mm_encoder": True,
        "encoder_cudagraph_max_vision_items_per_batch": max_vision_items,
    }


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "video_pruning_rate", [0.0] if current_platform.is_cpu() else [0.0, 0.75]
)
@pytest.mark.parametrize("num_frames", [16])
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize(
    "use_bytecode_hook", [True] if current_platform.is_cpu() else [True, False]
)
def test_qwen2_5_vl_evs_functionality(
    vllm_runner,
    video_assets,
    model,
    video_pruning_rate: float,
    num_frames: int,
    dtype: str,
    max_tokens: int,
    use_bytecode_hook: bool,
    monkeypatch,
) -> None:
    """Test EVS (Efficient Video Sampling) functionality with different
    pruning rates.
    """
    # Set the environment variable for this test
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    # Sample frames from video assets
    sampled_vids = [
        sample_frames_with_video_metadata(
            (asset.np_ndarrays, asset.metadata), num_frames
        )
        for asset in video_assets
    ]

    prompts = [VIDEO_PROMPTS[0]]
    videos = [sampled_vids[0]]

    # Initialize model with EVS configuration
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4000,
        dtype=dtype,
        limit_mm_per_prompt={"video": 1},
        video_pruning_rate=video_pruning_rate,
    ) as vllm_model:
        # Generate output - this should not crash
        outputs = vllm_model.generate_greedy(prompts, max_tokens, videos=videos)

        # Basic validation that we got a response
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]

        # Ensure we got some output
        assert len(output_ids) > 0
        assert len(output_text) > 0

        # Ensure the output is a string
        assert isinstance(output_text, str)


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "video_pruning_rate", [0.0] if current_platform.is_cpu() else [0.0, 0.75]
)
@pytest.mark.parametrize("num_frames", [16])
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize(
    "use_bytecode_hook", [True] if current_platform.is_cpu() else [True, False]
)
def test_qwen2_5_vl_evs_batched_videos(
    vllm_runner,
    video_assets,
    model,
    video_pruning_rate: float,
    num_frames: int,
    dtype: str,
    max_tokens: int,
    use_bytecode_hook: bool,
    monkeypatch,
) -> None:
    """Test EVS functionality with batched videos.

    This test validates that:
    1. The model handles batched video inputs correctly with EVS
    2. Both pruning configurations work with multiple videos
    3. The model doesn't crash when processing multiple videos simultaneously
    """
    # Set the environment variable for this test
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")
    # Sample frames from video assets
    sampled_vids = [
        sample_frames_with_video_metadata(
            (asset.np_ndarrays, asset.metadata), num_frames
        )
        for asset in video_assets
    ]

    # Test batched videos
    prompts = [VIDEO_PROMPTS[0], VIDEO_PROMPTS[0]]
    videos = [sampled_vids[0], sampled_vids[0]]  # Use same video twice for testing

    # Initialize model with EVS configuration
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4000,
        max_num_seqs=2,
        dtype=dtype,
        limit_mm_per_prompt={"video": 2},
        tensor_parallel_size=1,
        video_pruning_rate=video_pruning_rate,
    ) as vllm_model:
        # Generate output - this should not crash
        outputs = vllm_model.generate_greedy(prompts, max_tokens, videos=videos)

        # Basic validation that we got responses for both videos
        assert len(outputs) == 2

        for output_ids, output_text in outputs:
            # Ensure we got some output for each video
            assert len(output_ids) > 0
            assert len(output_text) > 0

            # Ensure the output is a string
            assert isinstance(output_text, str)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", models)
def test_qwen2_5_vl_evs_pipeline_parallel(vllm_runner, video_assets, model) -> None:
    """Regression: later PP stages must use the M-RoPE positions that EVS
    recomputes on the first stage, the only one that runs the vision encoder."""
    asset = video_assets[0]
    video = sample_frames_with_video_metadata((asset.np_ndarrays, asset.metadata), 16)
    outputs = {}
    for pp_size in (1, 2):
        with vllm_runner(
            model,
            max_model_len=4000,
            dtype=target_dtype,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 1},
            video_pruning_rate=0.75,
            pipeline_parallel_size=pp_size,
            distributed_executor_backend="mp",
        ) as vllm_model:
            (outputs[pp_size],) = vllm_model.generate_greedy_logprobs(
                [VIDEO_PROMPTS[0]],
                max_tokens=16,
                num_logprobs=5,
                num_prompt_logprobs=0,
                videos=[video],
                include_prompt_token_ids=True,
            )

    def prompt_logprobs(output) -> torch.Tensor:
        *_, token_ids, positions = output
        return torch.tensor(
            [
                position[token_id].logprob
                for token_id, position in zip(token_ids[1:], positions[1:])
            ]
        )

    diff = prompt_logprobs(outputs[1]) - prompt_logprobs(outputs[2])
    mean_diff = diff.abs().mean().item()
    # Without the relayed positions the mean difference is about 0.2.
    assert mean_diff < 0.05, f"PP=2 prompt logprobs differ from PP=1: {mean_diff=}"
    # Decode positions come from the per-request M-RoPE delta, which EVS also
    # changes.
    check_logprobs_close(
        outputs_0_lst=[outputs[1][:3]],
        outputs_1_lst=[outputs[2][:3]],
        name_0="pp1",
        name_1="pp2",
    )


@pytest.mark.core_model
@pytest.mark.skipif(
    current_platform.is_cpu(), reason="17k-token video prefill is too slow on CPU"
)
@pytest.mark.parametrize("model", models)
def test_qwen2_5_vl_video_fits_max_model_len(vllm_runner, model) -> None:
    num_frames = 60
    video = np.zeros((num_frames, 504, 896, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    prompt = qwen2_5_vl_chat_template(VIDEO_PLACEHOLDER, "Describe the video.")

    with vllm_runner(
        model,
        runner="generate",
        max_model_len=20000,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        dtype=target_dtype,
        limit_mm_per_prompt={"image": 0, "video": 1},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy([prompt], 8, videos=[(video, metadata)])

    output_ids, output_text = outputs[0]
    assert len(output_ids) > 17280
    assert len(output_text) > 0


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize(
    "use_bytecode_hook", [True] if current_platform.is_cpu() else [True, False]
)
def test_qwen2_5_vl_window_attention_image(
    vllm_runner,
    model,
    dtype: str,
    max_tokens: int,
    use_bytecode_hook: bool,
    monkeypatch,
) -> None:
    """Regression test for Qwen2.5 window-attention image path."""
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    prompt = [WINDOW_ATTN_IMAGE_PROMPT]
    images = [[_window_attention_regression_image()]]

    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        dtype=dtype,
        limit_mm_per_prompt=IMAGE_ONLY_LIMIT_MM_PER_PROMPT,
        compilation_config=_encoder_cudagraph_config(max_vision_items=1),
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompt, max_tokens, images=images)

        assert len(outputs) == 1
        output_ids, output_text = outputs[0]
        assert len(output_ids) > 0
        assert len(output_text) > 0
        assert isinstance(output_text, str)


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize(
    "use_bytecode_hook", [True] if current_platform.is_cpu() else [True, False]
)
def test_qwen2_5_vl_window_attention_image_batch(
    vllm_runner,
    model,
    dtype: str,
    max_tokens: int,
    use_bytecode_hook: bool,
    monkeypatch,
) -> None:
    """Regression test window-attention with a small image batch."""
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1" if use_bytecode_hook else "0")

    image = _window_attention_regression_image()
    prompts = [WINDOW_ATTN_IMAGE_PROMPT, WINDOW_ATTN_IMAGE_PROMPT]
    images = [[image], [image]]

    with vllm_runner(
        model,
        runner="generate",
        max_model_len=4096,
        max_num_seqs=2,
        dtype=dtype,
        limit_mm_per_prompt=IMAGE_ONLY_LIMIT_MM_PER_PROMPT,
        compilation_config=_encoder_cudagraph_config(max_vision_items=2),
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens, images=images)

        assert len(outputs) == 2
        for output_ids, output_text in outputs:
            assert len(output_ids) > 0
            assert len(output_text) > 0
            assert isinstance(output_text, str)
