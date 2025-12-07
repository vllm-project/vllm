# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.video import VideoAsset
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.multimodal.video import OpenCVDynamicVideoBackend, OpenCVVideoBackend

from ...utils import build_model_context


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
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.renderer_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {"fps": fps}

    # Build the image str / prompt based on the number of images we pass
    video_assets = VideoAsset(name="baby_reading", num_frames=num_frames)
    prompt = "<|begin_of_video|><|video|><|end_of_video|>"

    video, metadata = video_assets.np_ndarrays, video_assets.metadata
    metadata["fps"] = fps
    mm_data = {"video": [(video, metadata)]}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    video_token_id = tokenizer.convert_tokens_to_ids(hf_processor.video_token)
    video_tok_count = processed_inputs["prompt_token_ids"].count(video_token_id)
    grid_t, _, _ = processed_inputs["mm_kwargs"].get_data()["video_grid_thw"][0]

    assert grid_t == expected_grid_t
    assert video_tok_count == expected_toks_per_frame * grid_t


@pytest.mark.parametrize("model_id", ["zai-org/GLM-4.1V-9B-Thinking"])
@pytest.mark.parametrize("fps", [2])
def test_video_loader_consistency(
    model_id: str,
    fps: int,
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
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.renderer_config)
    hf_processor_mm_kwargs = {"fps": fps}

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|begin_of_video|><|video|><|end_of_video|>"

    video_path = VideoAsset(name="baby_reading", num_frames=-1).video_path
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    static_video, static_metadata = OpenCVVideoBackend.load_bytes(video_bytes)
    dynamic_video, dynamic_metadata = OpenCVDynamicVideoBackend.load_bytes(
        video_bytes, fps=fps
    )

    # pre-sampled loader shouldn't read all frames
    assert len(dynamic_video) < len(static_video)

    static_mm_data = {"video": [(static_video, static_metadata)]}
    dynamic_mm_data = {"video": [(dynamic_video, dynamic_metadata)]}

    static_outputs = processor.apply(prompt, static_mm_data, hf_processor_mm_kwargs)
    dynamic_outputs = processor.apply(prompt, dynamic_mm_data, hf_processor_mm_kwargs)

    assert static_outputs["prompt_token_ids"] == dynamic_outputs["prompt_token_ids"]
    assert batched_tensors_equal(
        static_outputs["mm_kwargs"].get_data(),
        dynamic_outputs["mm_kwargs"].get_data(),
    )
