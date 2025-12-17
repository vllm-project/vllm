# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal.video import sample_frames_from_video

from ....conftest import VIDEO_ASSETS

models = ["Qwen/Qwen2.5-VL-3B-Instruct"]
target_dtype = "bfloat16"

VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


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


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("video_pruning_rate", [0.0, 0.75])
@pytest.mark.parametrize("num_frames", [16])
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("use_bytecode_hook", [True, False])
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
@pytest.mark.parametrize("video_pruning_rate", [0.0, 0.75])
@pytest.mark.parametrize("num_frames", [16])
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("use_bytecode_hook", [True, False])
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
