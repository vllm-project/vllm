# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.assets.video import VideoAsset
from vllm.model_executor.models.glm4_1v import (
    Glm4vForConditionalGeneration,
    Glm4vProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.multimodal.video import DynamicVideoBackend, VideoBackend

from ...utils import build_model_context


@pytest.mark.parametrize(
    (
        "max_video_pixels",
        "max_tokens",
        "expected_num_frames",
    ),
    [
        (47_040_000, 124_988, 11),
        (47_040_000, 30_000, 24),
        (100_352_000, 124_988, 21),
        (100_352_000, 30_000, 7),
        (100_352_000, 0, 1),
    ],
)
def test_get_max_video_frames_matches_glm_resize(
    max_video_pixels: int,
    max_tokens: int,
    expected_num_frames: int,
):
    info = Mock(spec=Glm4vProcessingInfo)
    info.get_image_size_with_most_features.return_value = (2184, 2184)
    info._get_video_max_pixels.return_value = max_video_pixels
    vision_config = info.get_hf_config.return_value.vision_config
    vision_config.patch_size = 14
    vision_config.spatial_merge_size = 2
    vision_config.temporal_patch_size = 2
    info._get_vision_info.side_effect = lambda **kwargs: (
        Glm4vProcessingInfo._get_vision_info(info, **kwargs)
    )

    num_frames = Glm4vProcessingInfo._get_max_video_frames(
        info,
        max_tokens=max_tokens,
    )

    assert num_frames == expected_num_frames
    assert info._get_video_max_pixels.call_count == 1
    assert info._get_vision_info.call_count == 600


def test_encoder_cudagraph_uses_model_video_frame_limit():
    model = Mock()

    assert Glm4vForConditionalGeneration.get_max_frames_per_video(model) == 600


@pytest.mark.parametrize("model_id", ["zai-org/GLM-4.1V-9B-Thinking"])
@pytest.mark.parametrize("expected_toks_per_frame", [299])
@pytest.mark.parametrize(
    "num_frames, fps, expected_grid_t",
    [
        # pre-sampled fixed frames (unexpected behavior,
        # but we still expect it to work without errors)
        (32, 1, 16),
        (32, 2, 16),
        (128, 1, 64),
        (128, 2, 64),
        # post-sampled frames (expected behavior)
        (-1, 1, 5),
        (-1, 2, 10),
    ],
)
def test_processor_override(
    model_id: str,
    expected_toks_per_frame: int,
    expected_grid_t: int,
    fps: int,
    num_frames: int,
):
    """Ensure GLM4vMultiModalProcessor can handle video frames properly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {"fps": fps}

    # Build the image str / prompt based on the number of images we pass
    video_assets = VideoAsset(name="baby_reading", num_frames=num_frames)
    prompt = "<|begin_of_video|><|video|><|end_of_video|>"

    video, metadata = video_assets.np_ndarrays, video_assets.metadata
    metadata["fps"] = fps
    mm_data = {"video": [(video, metadata)]}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    video_token_id = tokenizer.convert_tokens_to_ids(hf_processor.video_token)
    video_tok_count = processed_inputs["prompt_token_ids"].count(video_token_id)
    grid_t, _, _ = processed_inputs["mm_kwargs"].get_data()["video_grid_thw"][0]

    assert grid_t == expected_grid_t
    assert video_tok_count == expected_toks_per_frame * grid_t


@pytest.mark.parametrize("model_id", ["zai-org/GLM-4.1V-9B-Thinking"])
@pytest.mark.parametrize("fps", [2])
@pytest.mark.parametrize("backend", ["opencv", "pyav"])
def test_video_loader_consistency(
    model_id: str,
    fps: int,
    backend: str,
):
    """
    Ensure dynamic video loader (pre-sampled by loader) and normal video
    loader (post-sampled by processor) produce same video processing outputs.
    """
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {"fps": fps}

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|begin_of_video|><|video|><|end_of_video|>"

    video_path = VideoAsset(name="baby_reading", num_frames=-1).video_path
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    static_video, static_metadata = VideoBackend.load_bytes(
        video_bytes, backend=backend
    )
    dynamic_video, dynamic_metadata = DynamicVideoBackend.load_bytes(
        video_bytes, fps=fps, backend=backend
    )

    # pre-sampled loader shouldn't read all frames
    assert len(dynamic_video) < len(static_video)

    static_mm_data = {"video": [(static_video, static_metadata)]}
    dynamic_mm_data = {"video": [(dynamic_video, dynamic_metadata)]}

    static_outputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(static_mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )
    dynamic_outputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(dynamic_mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    assert static_outputs["prompt_token_ids"] == dynamic_outputs["prompt_token_ids"]
    assert batched_tensors_equal(
        static_outputs["mm_kwargs"].get_data(),
        dynamic_outputs["mm_kwargs"].get_data(),
    )
