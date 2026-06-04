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


def create_paired_mm_data(video_seed: int, audio_seed: int) -> dict[str, list]:
    """One video + one audio with independently controllable content. Use the
    same ``audio_seed`` across requests to get a byte-identical audio track."""
    video = random_video(
        np.random.RandomState(video_seed),
        min_frames=8,
        max_frames=9,
        min_wh=64,
        max_wh=65,
    )
    audio, sr = random_audio(
        np.random.RandomState(audio_seed),
        min_len=8000,
        max_len=8001,
        sr=16000,
    )
    return dict[str, list](video=[video], audio=[(audio, sr)])


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
            [video_token_id] * num_videos,
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


@pytest.mark.parametrize("model_id", MODELS)
def test_audio_in_video_same_audio_different_video(model_id: str) -> None:
    """Regression test for https://github.com/vllm-project/vllm/issues/44538

    With ``use_audio_in_video=True`` the audio track is extracted from the
    video into a separate multimodal item that the processor cache hashes by
    content. Two requests with *different* videos but a *byte-identical* audio
    track collide on the audio cache entry: the second request's video misses
    the cache while its audio hits, so the HF processor is run on the video
    alone — but with ``use_audio_in_video=True`` it expects the audio
    interleaved and raises ``StopIteration`` (HTTP 400 in serving).

    The existing ``test_audio_in_video_cache_correctness`` reuses a single
    video across turns, so this cross-request collision is not covered there.

    The fix couples ``video[i]`` with ``audio[i]`` (via
    ``_get_mm_cache_coupled_groups``) so a cache-missed video drags its paired
    audio back through the HF processor even when the audio is individually
    cached. This test asserts the second request neither raises nor produces
    wrong tokens.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )

    baseline_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=None
    )
    sender_cache = MultiModalProcessorSenderCache(ctx.model_config)
    cached_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=sender_cache
    )

    video_token_id = baseline_processor.info.get_hf_config().video_token_id
    hf_processor_mm_kwargs = {"use_audio_in_video": True}

    # Different videos, byte-identical audio track (same audio_seed).
    mm_data_a = create_paired_mm_data(video_seed=1, audio_seed=99)
    mm_data_b = create_paired_mm_data(video_seed=2, audio_seed=99)

    def run(processor, mm_data):
        return processor(
            [video_token_id],
            mm_items=baseline_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )["prompt_token_ids"]

    baseline_b_ids = run(baseline_processor, mm_data_b)

    # Warm the cache with request A (video A + audio X).
    run(cached_processor, mm_data_a)

    # Request B: different video, SAME audio. Audio is a cache hit; video is a
    # miss. Without the coupling fix the HF processor receives the video with
    # no paired audio and raises StopIteration; with it, the audio is
    # reprocessed alongside the video and the tokens match the baseline.
    second_ids = run(cached_processor, mm_data_b)
    assert second_ids == baseline_b_ids, (
        "Same-audio/different-video cache-hit produced different "
        "prompt_token_ids than baseline (issue #44538).\n"
        f"  baseline : {baseline_b_ids}\n"
        f"  cache-hit: {second_ids}"
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_audio_in_video_same_video_different_audio(model_id: str) -> None:
    """Reverse-polarity companion to the #44538 regression: *same* video,
    *different* audio.

    Here the video is a cache hit while the audio misses. The video's
    interleaved placeholder layout depends on the paired audio's length, so
    serving the cached (stale) video layout for the new audio would yield wrong
    prompt_token_ids. The coupling fix reprocesses the whole group, so the
    video layout is rebuilt for the new audio.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )

    baseline_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=None
    )
    sender_cache = MultiModalProcessorSenderCache(ctx.model_config)
    cached_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=sender_cache
    )

    video_token_id = baseline_processor.info.get_hf_config().video_token_id
    hf_processor_mm_kwargs = {"use_audio_in_video": True}

    # Same video, different audio (different length → different layout).
    mm_data_a = create_paired_mm_data(video_seed=7, audio_seed=1)
    mm_data_b = create_paired_mm_data(video_seed=7, audio_seed=2)

    def run(processor, mm_data):
        return processor(
            [video_token_id],
            mm_items=baseline_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )["prompt_token_ids"]

    baseline_b_ids = run(baseline_processor, mm_data_b)

    run(cached_processor, mm_data_a)  # warm cache with (video, audio A)
    second_ids = run(cached_processor, mm_data_b)  # video hit, audio miss
    assert second_ids == baseline_b_ids, (
        "Same-video/different-audio cache-hit reused a stale interleaved video "
        "layout (issue #44538, reverse polarity).\n"
        f"  baseline : {baseline_b_ids}\n"
        f"  cache-hit: {second_ids}"
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_audio_in_video_cache_keys_do_not_change_semantic_hashes(
    model_id: str,
) -> None:
    """The coupling fix rewrites *processor-cache keys* only; the semantic
    ``mm_info.hashes`` (which feed the encoder-output and prefix caches) must
    stay identical between the cached and uncached paths, or cached vs uncached
    runs would diverge in downstream cache identity."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )

    baseline_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=None
    )
    sender_cache = MultiModalProcessorSenderCache(ctx.model_config)
    cached_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, cache=sender_cache
    )

    video_token_id = baseline_processor.info.get_hf_config().video_token_id
    hf_processor_mm_kwargs = {"use_audio_in_video": True}
    mm_data = create_paired_mm_data(video_seed=3, audio_seed=4)

    def hashes(processor):
        return processor(
            [video_token_id],
            mm_items=baseline_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )["mm_hashes"]

    assert hashes(baseline_processor) == hashes(cached_processor), (
        "Coupling fix must not change the semantic mm_hashes; only the internal "
        "processor-cache keys are group-aware."
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_audio_in_video_partial_residency_reprocesses_whole_group(
    model_id: str,
) -> None:
    """Group-aware keys alone are not enough: a coupled group's members are
    still stored as independent cache entries, so LRU eviction can drop one
    member while the other survives. If only the video is evicted, a later
    request would otherwise get video-miss / audio-hit and run the HF processor
    on the video without its paired audio (the #44538 StopIteration).

    Group-miss expansion in ``_get_cache_missing_items`` must therefore drag the
    still-resident audio back into the reprocessed set. This drives that method
    directly with a fake cache simulating audio-resident / video-evicted, which
    is deterministic (no reliance on LRU eviction order).
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config, cache=None)

    mm_data = create_paired_mm_data(video_seed=5, audio_seed=6)
    mm_items = processor.info.parse_mm_data(mm_data)

    coupled_groups = processor._get_mm_cache_coupled_groups(
        mm_items, {"use_audio_in_video": True}
    )
    assert coupled_groups, "coupling must be active for use_audio_in_video"

    class _AudioResidentVideoEvictedCache:
        """Simulates the audio member surviving in cache while the video member
        was evicted (partial residency)."""

        def is_cached(self, keys: list[str]) -> list[bool]:
            return [key.startswith("audio:") for key in keys]

    mm_cache_keys = {"video": ["video:0"], "audio": ["audio:0"]}

    mm_needs_processing, mm_missing_items = processor._get_cache_missing_items(
        cache=_AudioResidentVideoEvictedCache(),
        mm_data_items=mm_items,
        mm_cache_keys=mm_cache_keys,
        coupled_groups=coupled_groups,
    )

    # Video missed; the resident audio is dragged back in by group expansion.
    assert mm_needs_processing["video"] == [True]
    assert mm_needs_processing["audio"] == [True]
    # Both must be present in the data fed to the HF processor.
    assert mm_missing_items.get_count("video", strict=False) == 1
    assert mm_missing_items.get_count("audio", strict=False) == 1
