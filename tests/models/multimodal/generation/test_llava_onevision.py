# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np

from vllm import SamplingParams

from ....conftest import VllmRunner


def test_llava_onevision_mixed_frame_count_videos(vllm_runner: type[VllmRunner]):
    """Preserve all video tokens when one request has unequal frame counts."""
    frame_counts = (2, 3)
    videos = [
        np.full((frames, 384, 384, 3), 128, dtype=np.uint8) for frames in frame_counts
    ]
    prompt = (
        "<|im_start|>user\n<video>\n<video>\nDescribe the videos."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    with vllm_runner(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 0, "video": 2},
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
    ) as model:
        llm = model.get_llm()
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"video": videos}},
            SamplingParams(temperature=0, max_tokens=2, logprobs=1),
        )
        assert len(outputs) == 1
        output = outputs[0]
        config = llm.model_config.hf_config
        # This checkpoint pools each frame to 14 x 14 tokens, plus one newline
        # token per video.
        expected_video_tokens = sum(frames * 196 + 1 for frames in frame_counts)
        assert output.prompt_token_ids.count(config.video_token_index) == (
            expected_video_tokens
        )
        completion = output.outputs[0]
        assert completion.token_ids
        assert completion.logprobs is not None
        for token_id, logprobs in zip(completion.token_ids, completion.logprobs):
            assert math.isfinite(logprobs[token_id].logprob)
