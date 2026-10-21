# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.video import VideoAsset

from ...utils import check_logprobs_close

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
VIDEO_PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.core_model
def test_qwen3_vl_evs_chunked_prefill(vllm_runner):
    """Splitting a pruned video across prefill chunks must not change its
    M-RoPE positions, so the output matches prefilling it in one chunk."""
    video = VideoAsset("baby_reading", num_frames=16)
    videos = [(video.np_ndarrays, video.metadata)]

    outputs = []
    for max_num_batched_tokens in (8192, 128):
        with vllm_runner(
            MODEL,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=1,
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"video": 1},
            video_pruning_rate=0.75,
        ) as vllm_model:
            outputs.append(
                vllm_model.generate_greedy_logprobs(
                    [VIDEO_PROMPT], 32, num_logprobs=5, videos=videos
                )
            )

    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="one_chunk",
        name_1="chunked",
    )
