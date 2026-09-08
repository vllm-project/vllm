# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.assets.image import ImageAsset
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from ....conftest import VIDEO_ASSETS

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


def test_qwen2_5_vl_encoder_cudagraph_normalizes_pixels() -> None:
    """Graph replay and eager fallback match the normal image input path."""

    class InputNorm:
        def __init__(self):
            self.dtypes: list[torch.dtype] = []

        def __call__(
            self, pixel_values: torch.Tensor, dtype: torch.dtype
        ) -> torch.Tensor:
            self.dtypes.append(dtype)
            return pixel_values.to(dtype) + 1

    class Visual:
        dtype = torch.float16

        def prepare_encoder_metadata(
            self, *_args, **_kwargs
        ) -> dict[str, torch.Tensor]:
            return {}

        def __call__(
            self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
        ) -> torch.Tensor:
            self.pixel_values = pixel_values
            self.grid_thw = grid_thw
            return pixel_values

    visual = Visual()
    input_norm = InputNorm()
    adapter = SimpleNamespace(
        visual=visual,
        input_norm=input_norm,
        get_input_modality=lambda _: "image",
        _get_grid_thw_by_modality=lambda kwargs: kwargs["image_grid_thw"],
        _get_pixel_values_by_modality=lambda kwargs: kwargs["pixel_values"],
    )
    pixel_values = torch.zeros(4, 12)
    mm_kwargs = {
        "pixel_values": pixel_values,
        "image_grid_thw": [[1, 2, 2]],
    }

    replay = (
        Qwen2_5_VLForConditionalGeneration.prepare_encoder_cudagraph_replay_buffers(
            adapter, mm_kwargs, max_batch_size=1, max_frames_per_batch=1
        )
    )
    expected = torch.ones_like(pixel_values, dtype=visual.dtype)
    torch.testing.assert_close(replay.values["pixel_values"], expected)

    output = Qwen2_5_VLForConditionalGeneration.encoder_eager_forward(
        adapter, mm_kwargs
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(visual.pixel_values, expected)
    assert visual.grid_thw == [[1, 2, 2]]
    assert input_norm.dtypes == [visual.dtype, visual.dtype]


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
        sample_frames_from_video(asset.np_ndarrays, num_frames)
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
        sample_frames_from_video(asset.np_ndarrays, num_frames)
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
