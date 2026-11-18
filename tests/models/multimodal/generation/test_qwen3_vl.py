# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import check_outputs_equal
from .vlm_utils.builders import sample_frames_with_video_metadata

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# The processor wraps every frame in <|vision_start|>...<|vision_end|> itself. The
# chat template's outer pair is left out: with it, the cache-miss path is itself
# wrong under pruning (#62937, fixed by #62959), and this test would compare against it.
VIDEO_PLACEHOLDER = "<|video_pad|>"

# Shared text after the video, several KV blocks long, so that a repeated request
# reuses the cache past the end of the video and the video is not encoded again.
TEXT_AFTER_VIDEO = (
    "You are reviewing a short home video. Look at the people, the objects they "
    "hold, where they are, what they do from the first frame to the last, and "
    "how the scene changes over time. Use plain words, name only what is "
    "visible, do not guess at things outside the frame, and keep the answer to "
    "a single short sentence that a child could understand. "
) * 3


def qwen3_vl_chat_template(*query: str) -> str:
    return f"<|im_start|>user\n{''.join(query)}<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.core_model
@pytest.mark.skipif(
    current_platform.is_cpu(), reason="Video pruning is not tested on CPU"
)
@pytest.mark.parametrize(
    "video_pruning_method,video_pruning_rate",
    [("evs", 0.0), ("evs", 0.5), ("vidcom2", 0.5)],
    ids=["unpruned", "evs-0.5", "vidcom2-0.5"],
)
def test_qwen3_vl_pruned_video_prefix_cache_hit(
    vllm_runner, video_assets, video_pruning_method: str, video_pruning_rate: float
) -> None:
    """A prompt whose whole video is a prefix cache hit must give the same greedy
    output as the cache miss: the positions of the text after a pruned video do
    not depend on whether the video was encoded in this request.
    """
    video = sample_frames_with_video_metadata(
        (video_assets[0].np_ndarrays, video_assets[0].metadata), 16
    )
    prompt = qwen3_vl_chat_template(
        VIDEO_PLACEHOLDER,
        TEXT_AFTER_VIDEO,
        "Describe the video in one sentence.",
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)

    with vllm_runner(
        MODEL,
        runner="generate",
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"video": 1},
        video_pruning_rate=video_pruning_rate,
        video_pruning_method=video_pruning_method,
        enable_prefix_caching=True,
    ) as vllm_model:
        inputs = vllm_model.get_inputs([prompt], videos=[video])
        outputs = [vllm_model.llm.generate(inputs, params)[0] for _ in range(2)]
        vision_end = vllm_model.llm.get_tokenizer().convert_tokens_to_ids(
            "<|vision_end|>"
        )

    # The second request must reuse the cache past the end of the video, which
    # ends at the last <|vision_end|> (one per frame).
    ids = outputs[1].prompt_token_ids
    video_end = len(ids) - 1 - ids[::-1].index(vision_end)
    assert outputs[0].num_cached_tokens == 0
    assert outputs[1].num_cached_tokens > video_end

    miss, hit = ([(list(o.outputs[0].token_ids), o.outputs[0].text)] for o in outputs)
    check_outputs_equal(
        outputs_0_lst=miss, outputs_1_lst=hit, name_0="cache_miss", name_1="cache_hit"
    )
