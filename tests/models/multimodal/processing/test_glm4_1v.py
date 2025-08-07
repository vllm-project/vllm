# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.video import VideoAsset
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["zai-org/GLM-4.1V-9B-Thinking"])
@pytest.mark.parametrize("expected_toks_per_frame", [299])
@pytest.mark.parametrize("num_frames", [32, 128])
@pytest.mark.parametrize("fps, expected_grid_t", [(1, 5), (2, 10)])
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

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    video_token_id = tokenizer.convert_tokens_to_ids(hf_processor.video_token)
    video_tok_count = processed_inputs["prompt_token_ids"].count(
        video_token_id)
    grid_t, _, _ = processed_inputs["mm_kwargs"]["video_grid_thw"][0]

    assert grid_t == expected_grid_t
    assert video_tok_count == expected_toks_per_frame * grid_t
