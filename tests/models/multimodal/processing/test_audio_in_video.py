# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for Qwen2.5-Omni and Qwen3-Omni audio-in-video processor
caching.

Tests the use_audio_in_video feature where audio is extracted from video and
processed together with video frames in an interleaved manner.

Regression test: when use_audio_in_video=True and the multimodal processor
cache is warm, the second request goes through MultiModalProcessorSenderCache
which sets mm_kwargs["video"] items to None on a cache hit.  The processor
must still detect use_audio_in_video=True (via token-count heuristic) and
produce the same prompt_token_ids as the first (cache-miss) request.

Without the fix the cache-hit path left use_audio_in_video=False, causing
audio placeholder tokens to be inserted separately instead of being derived
from the interleaved video placeholders – yielding a different (wrong) token
sequence on every subsequent request for the same video.
"""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorSenderCache

from ....multimodal.utils import random_audio, random_video
from ...utils import build_model_context

MODELS = [
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
]


def create_mm_data(num_videos: int) -> dict[str, list]:
    # Small video (8 frames, 64×64) and ~0.5 s of audio at 16 kHz so the test
    # stays fast even without a GPU.
    mm_data = dict[str, list](video=[], audio=[])
    for i in range(num_videos):
        rng = np.random.RandomState(i)
        video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)
        audio, sr = random_audio(rng, min_len=8000, max_len=8001, sr=16000)
        mm_data["video"].append(video)
        mm_data["audio"].append((audio, sr))
    return mm_data


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("num_videos", [1, 2])
def test_audio_in_video_cache_correctness(model_id: str, num_videos: int) -> None:
    """
    Regression test for https://github.com/vllm-project/vllm/pull/36800

    MultiModalProcessorSenderCache.get_and_update_item returns (None, updates)
    on a cache hit, so mm_kwargs["video"] items become None on the second call.
    The Qwen processor override of _maybe_apply_prompt_updates must detect
    use_audio_in_video=True via token-count heuristics and re-derive the audio
    placeholders correctly.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": num_videos, "image": 0, "video": num_videos},
        mm_processor_cache_gb=1,
    )

    # Baseline: no cache, always processes from scratch.
    baseline_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=None
    )
    # Sender cache: on a cache hit returns (None, prompt_updates) for each
    # item, setting mm_kwargs["video"] = [None] – the exact condition that
    # triggered the original bug.
    sender_cache = MultiModalProcessorSenderCache(ctx.model_config)
    cached_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=sender_cache
    )

    video_token_id = baseline_processor.info.get_hf_config().video_token_id

    mm_data = create_mm_data(num_videos)
    hf_processor_mm_kwargs = {"use_audio_in_video": True}

    def run(processor):
        return processor(
            [video_token_id],
            mm_items=baseline_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )["prompt_token_ids"]

    baseline_ids = run(baseline_processor)

    # First call on the sender-cache processor: cache miss.
    # mm_kwargs["video"] items are real tensors; use_audio_in_video is
    # detected normally from the item data.
    first_ids = run(cached_processor)
    assert first_ids == baseline_ids, (
        "Cache-miss call produced different prompt_token_ids than baseline.\n"
        f"  baseline  : {baseline_ids}\n"
        f"  cache-miss: {first_ids}"
    )

    # Second call on the sender-cache processor: cache hit.
    # MultiModalProcessorSenderCache.get_and_update_item returns (None, …),
    # so mm_kwargs["video"] = [None].  Before the fix, use_audio_in_video was
    # not detected, yielding wrong token ids.
    second_ids = run(cached_processor)
    assert second_ids == baseline_ids, (
        "Cache-hit call produced different prompt_token_ids than baseline.\n"
        "This is the regression introduced when use_audio_in_video detection\n"
        "fails for None mm_kwargs items on a cache hit.\n"
        f"  baseline : {baseline_ids}\n"
        f"  cache-hit: {second_ids}"
    )
