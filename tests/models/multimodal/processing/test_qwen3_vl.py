# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3VL video processing, specifically temporal_patch_size validation."""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

# Use a smaller Qwen2VL model for testing since Qwen3VL may not be publicly available
# The temporal_patch_size validation logic is shared between Qwen2VL/Qwen2.5VL/Qwen3VL
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [
        2,  # Valid: multiple of temporal_patch_size=2
        4,  # Valid: multiple of temporal_patch_size=2
        8,  # Valid: multiple of temporal_patch_size=2
    ],
)
def test_video_valid_frame_count_passes(
    model_id: str,
    num_frames: int,
) -> None:
    """
    Test that videos with valid frame counts (multiples of temporal_patch_size)
    are processed successfully.

    Qwen2VL/Qwen3VL use temporal_patch_size=2, so frame counts must be even.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Create synthetic video with valid frame count
    height, width = 224, 224
    video_array = np.random.randint(
        0, 255, (num_frames, height, width, 3), dtype=np.uint8
    )

    metadata = {
        "do_sample_frames": False,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
    }

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = {"video": [(video_array, metadata)]}

    hf_processor_mm_kwargs = {"do_sample_frames": False}

    # Should process successfully
    processed_inputs = processor.apply(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    assert "prompt_token_ids" in processed_inputs
    assert len(processed_inputs["prompt_token_ids"]) > 0


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [
        1,  # Invalid: not a multiple of temporal_patch_size=2
        3,  # Invalid: not a multiple of temporal_patch_size=2
        5,  # Invalid: not a multiple of temporal_patch_size=2
        7,  # Invalid: not a multiple of temporal_patch_size=2
    ],
)
def test_video_invalid_frame_count_raises_error(
    model_id: str,
    num_frames: int,
) -> None:
    """
    Test that videos with invalid frame counts (not multiples of temporal_patch_size)
    raise an informative error.

    When do_sample_frames=False (pre-processed video), the user is responsible
    for ensuring the frame count is properly aligned, similar to how qwen-vl-utils
    handles it with round_by_factor/FRAME_FACTOR.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Create synthetic video with invalid frame count
    height, width = 224, 224
    video_array = np.random.randint(
        0, 255, (num_frames, height, width, 3), dtype=np.uint8
    )

    metadata = {
        "do_sample_frames": False,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
    }

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = {"video": [(video_array, metadata)]}

    hf_processor_mm_kwargs = {"do_sample_frames": False}

    # Should raise ValueError with informative message
    with pytest.raises(ValueError) as exc_info:
        processor.apply(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

    error_msg = str(exc_info.value)
    assert str(num_frames) in error_msg
    assert "temporal_patch_size" in error_msg


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_error_message_suggests_qwen_vl_utils(model_id: str) -> None:
    """
    Test that the error message provides actionable guidance,
    specifically mentioning qwen-vl-utils functions.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Create video with odd frame count
    num_frames = 5
    height, width = 224, 224
    video_array = np.random.randint(
        0, 255, (num_frames, height, width, 3), dtype=np.uint8
    )

    metadata = {
        "do_sample_frames": False,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
    }

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = {"video": [(video_array, metadata)]}

    hf_processor_mm_kwargs = {"do_sample_frames": False}

    with pytest.raises(ValueError) as exc_info:
        processor.apply(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

    error_msg = str(exc_info.value)

    # Check that actionable guidance is provided
    assert "smart_nframes" in error_msg or "round_by_factor" in error_msg
    assert "qwen-vl-utils" in error_msg
