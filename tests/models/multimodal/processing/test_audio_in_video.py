# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen2.5-Omni and Qwen3-Omni audio-in-video processor
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

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import LruKeyReplicatedSenderCache
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async

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
    """Regression test for https://github.com/vllm-project/vllm/pull/36800.

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

    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Sender cache: on a cache hit returns (None, prompt_updates) for each
    # item, setting mm_kwargs["video"] = [None] – the exact condition that
    # triggered the original bug.
    sender_cache = LruKeyReplicatedSenderCache(ctx.model_config)

    video_token_id = processor.info.get_hf_config().video_token_id

    mm_data = create_mm_data(num_videos)
    hf_processor_mm_kwargs = {"use_audio_in_video": True}

    def run(processor, cache):
        return processor(
            [video_token_id] * num_videos,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            cache=cache,
        )["prompt_token_ids"]

    baseline_ids = run(processor, cache=None)

    # First call on the sender-cache processor: cache miss.
    # mm_kwargs["video"] items are real tensors; use_audio_in_video is
    # detected normally from the item data.
    first_ids = run(processor, cache=sender_cache)
    assert first_ids == baseline_ids, (
        "Cache-miss call produced different prompt_token_ids than baseline.\n"
        f"  baseline  : {baseline_ids}\n"
        f"  cache-miss: {first_ids}"
    )

    # Second call on the sender-cache processor: cache hit.
    # MultiModalProcessorSenderCache.get_and_update_item returns (None, …),
    # so mm_kwargs["video"] = [None].  Before the fix, use_audio_in_video was
    # not detected, yielding wrong token ids.
    second_ids = run(processor, cache=sender_cache)
    assert second_ids == baseline_ids, (
        "Cache-hit call produced different prompt_token_ids than baseline.\n"
        "This is the regression introduced when use_audio_in_video detection\n"
        "fails for None mm_kwargs items on a cache hit.\n"
        f"  baseline : {baseline_ids}\n"
        f"  cache-hit: {second_ids}"
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_use_audio_in_video_without_audio_track(model_id: str) -> None:
    """A video with no audio track, combined with `use_audio_in_video=True`,
    must raise a clear `ValueError` naming the mismatch, rather than an
    opaque `StopIteration`-derived error from the underlying HF processor.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    rng = np.random.RandomState(0)
    video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)

    # No "audio" key at all in this request's data: this is the "video has
    # no audio track" case. (The audio limit above is 1, not 0, so that
    # `mm_counts["audio"]` is still declared for this request; a 0 limit
    # would make vLLM omit the "audio" key from `mm_counts` entirely, which
    # trips an unrelated assertion in `_apply_hf_processor_main`.)
    mm_data = {"video": [video]}

    with pytest.raises(ValueError, match="doesn't have audio track"):
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={"use_audio_in_video": True},
        )


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("has_image", [False, True])
def test_audio_in_video_metadata_only(
    model_id, num_videos, is_async, has_image, monkeypatch
):
    """Published metadata preserves interleaved placeholders without preprocessing."""
    from vllm.model_executor.models.qwen2_5_omni_thinker import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    model_cls = (
        Qwen3OmniMoeThinkerForConditionalGeneration
        if "Qwen3-Omni" in model_id
        else Qwen2_5OmniThinkerForConditionalGeneration
    )
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": num_videos, "video": num_videos, "image": 1},
    )
    mm_config = ctx.model_config.get_multimodal_config()
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_placeholders = [
        model_cls.get_placeholder_str("video", i) for i in range(num_videos)
    ]
    assert all(isinstance(token, str) for token in video_placeholders)
    prompt = "".join(token for token in video_placeholders if token is not None)
    kwargs = {"use_audio_in_video": True}
    uuids = {
        modality: [f"source-{i}" for i in range(num_videos)]
        for modality in ("video", "audio")
    }
    mm_data = create_mm_data(num_videos)
    if has_image:
        from PIL import Image

        mm_data["image"] = [Image.new("RGB", (64, 64), "green")]
        uuids["image"] = ["image-source"]
        image_placeholder = model_cls.get_placeholder_str("image", 0)
        assert image_placeholder is not None
        prompt = image_placeholder + prompt
    raw = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        mm_uuid_items=uuids,
        hf_processor_mm_kwargs=kwargs,
    )
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality=modality,
            identifier=f"{modality}-{i}",
            mm_position=position,
        )
        for modality in mm_data
        for i, (item, position) in enumerate(
            zip(raw["mm_kwargs"][modality], raw["mm_placeholders"][modality])
        )
    ]
    published = collect_ec_item_metadata(
        features, PlaceholderMetadataResolver(ctx.model_config)
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_embeds",
                    "video_embeds": published[f"video-{i}"]["metadata"],
                    "audio_embeds": published[f"audio-{i}"]["metadata"],
                    "uuid": f"source-{i}",
                }
                for i in range(num_videos)
            ],
        }
    ]
    if has_image:
        messages[0]["content"].insert(
            0,
            {
                "type": "image_embeds",
                "image_embeds": published["image-0"]["metadata"],
                "uuid": "image-source",
            },
        )
    if is_async:
        conversation, metadata, remote_uuids = asyncio.run(
            parse_chat_messages_async(
                messages, ctx.model_config, "openai", mm_processor_kwargs=kwargs
            )
        )
    else:
        conversation, metadata, remote_uuids = parse_chat_messages(
            messages, ctx.model_config, "openai", mm_processor_kwargs=kwargs
        )
    assert conversation[0]["content"] == (
        ([{"type": "image"}] if has_image else []) + [{"type": "video"}] * num_videos
    )
    assert remote_uuids == uuids

    def unexpected_preprocessing(*args, **kwargs):
        pytest.fail("Metadata-only audio/video must not call the HF processor")

    monkeypatch.setattr(processor, "_call_hf_processor", unexpected_preprocessing)
    remote = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(metadata),
        mm_uuid_items=remote_uuids,
        hf_processor_mm_kwargs=kwargs,
    )
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_hashes"] == raw["mm_hashes"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]
    for key in (
        "video_grid_thw",
        "second_per_grid_ts",
        "use_audio_in_video",
        "audio_feature_lengths",
    ):
        torch.testing.assert_close(
            remote["mm_kwargs"].get_data()[key],
            raw["mm_kwargs"].get_data()[key],
            check_dtype=False,
        )

    model = model_cls.__new__(model_cls)
    setattr(model, "config", processor.info.get_hf_config())  # noqa: B010
    remote_features = [
        MultiModalFeatureSpec(
            data=remote["mm_kwargs"][f.modality][int(f.identifier.split("-")[-1])],
            modality=f.modality,
            identifier=f.identifier,
            mm_position=f.mm_position,
        )
        for f in features
    ]
    get_positions = getattr(model, "get_mrope_input_positions")  # noqa: B009
    raw_positions, raw_delta = get_positions(raw["prompt_token_ids"], features)
    remote_positions, remote_delta = get_positions(
        remote["prompt_token_ids"], remote_features
    )
    assert raw_delta == remote_delta
    torch.testing.assert_close(raw_positions, remote_positions)
