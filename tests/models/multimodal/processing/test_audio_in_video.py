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
from vllm.multimodal.utils import get_use_audio_in_video
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async

from ....multimodal.utils import random_audio, random_video
from ...utils import build_model_context

MODELS = [
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"use_audio_in_video": True}, True),
        ({"use_audio_in_video": False}, False),
        ({"videos_kwargs": {"use_audio_in_video": True}}, True),
        ({"videos_kwargs": {"use_audio_in_video": False}}, False),
        # The HF video scope wins over the shared flat namespace.
        (
            {
                "use_audio_in_video": False,
                "videos_kwargs": {"use_audio_in_video": True},
            },
            True,
        ),
    ],
)
def test_get_use_audio_in_video_prefers_video_scope(
    kwargs: dict, expected: bool
) -> None:
    """`use_audio_in_video` must resolve from the flat and scoped forms alike."""
    assert get_use_audio_in_video(kwargs) is expected


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
    prompt = "".join(
        model_cls.get_placeholder_str("video", i) for i in range(num_videos)
    )
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
        prompt = model_cls.get_placeholder_str("image", 0) + prompt
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
    parse = parse_chat_messages_async if is_async else parse_chat_messages
    parsed = parse(messages, ctx.model_config, "openai", mm_processor_kwargs=kwargs)
    conversation, metadata, remote_uuids = asyncio.run(parsed) if is_async else parsed
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
    model.config = processor.info.get_hf_config()
    remote_features = [
        MultiModalFeatureSpec(
            data=remote["mm_kwargs"][f.modality][int(f.identifier.split("-")[-1])],
            modality=f.modality,
            identifier=f.identifier,
            mm_position=f.mm_position,
        )
        for f in features
    ]
    raw_positions, raw_delta = model.get_mrope_input_positions(
        raw["prompt_token_ids"], features
    )
    remote_positions, remote_delta = model.get_mrope_input_positions(
        remote["prompt_token_ids"], remote_features
    )
    assert raw_delta == remote_delta
    torch.testing.assert_close(raw_positions, remote_positions)


@pytest.mark.parametrize("model_id", MODELS)
def test_engine_level_use_audio_in_video(model_id: str) -> None:
    """`--mm-processor-kwargs` must reach the prompt updates, not just HF.

    The flag is configured on the engine, so the request passes no
    `hf_processor_mm_kwargs`. The HF processor interleaves audio into the video
    placeholder, and the processor must derive the audio placeholders from it
    instead of also emitting the standalone audio update.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_kwargs={"use_audio_in_video": True},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    mm_data = create_mm_data(1)
    engine_ids = processor(
        [video_token_id],
        mm_items=processor.info.parse_mm_data(mm_data),
    )["prompt_token_ids"]

    request_ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
    )
    request_ids = MULTIMODAL_REGISTRY.create_processor(request_ctx.model_config)(
        [video_token_id],
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"use_audio_in_video": True},
    )["prompt_token_ids"]

    assert engine_ids == request_ids


@pytest.mark.parametrize("model_id", MODELS)
def test_videos_kwargs_scoped_use_audio_in_video(model_id: str) -> None:
    """The scoped `videos_kwargs` form must behave like the flat one.

    `use_audio_in_video` is an HF `videos_kwargs` key, so configuring it inside
    that scope must not be silently ignored by the placeholder logic.
    """
    mm_data = create_mm_data(1)
    limit = {"audio": 1, "image": 0, "video": 1}

    flat_ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit,
        mm_processor_kwargs={"use_audio_in_video": True},
    )
    scoped_ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit,
        mm_processor_kwargs={"videos_kwargs": {"use_audio_in_video": True}},
    )

    def run(ctx):
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        video_token_id = processor.info.get_hf_config().video_token_id
        return processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(mm_data),
        )["prompt_token_ids"]

    assert run(scoped_ctx) == run(flat_ctx)


@pytest.mark.parametrize("model_id", MODELS)
def test_encoder_profiler_pairs_dummy_audio_with_video(model_id: str) -> None:
    """The encoder profiler requests dummy input for one modality at a time.

    With `use_audio_in_video` in effect, the HF processor rejects a video
    without its paired audio track, so the dummy counts must be completed with
    the companion audio item.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_kwargs={"use_audio_in_video": True},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    assert processor._pair_dummy_mm_counts({"video": 1}) == {
        "audio": 1,
        "video": 1,
    }

    dummy_inputs = processor.get_dummy_mm_inputs({"video": 1})
    assert len(dummy_inputs["mm_kwargs"]["video"]) == 1
    assert len(dummy_inputs["mm_kwargs"]["audio"]) == 1
