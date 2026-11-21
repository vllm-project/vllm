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
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import LruKeyReplicatedSenderCache
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import VIDEO_AUDIO_UUID_PREFIX, get_use_audio_in_video
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
        ({"load_audio_from_video": True}, True),
        ({"videos_kwargs": {"load_audio_from_video": True}}, True),
        ({"load_audio_from_video": False, "use_audio_in_video": True}, False),
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
@pytest.mark.parametrize("audio_kwarg", ["load_audio_from_video", "use_audio_in_video"])
def test_audio_kwarg_alias_matches_legacy_behavior(
    model_id: str, audio_kwarg: str
) -> None:
    """The Transformers flag must drive the same processor guards as the alias."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id
    mm_items = processor.info.parse_mm_data(create_mm_data(1))

    with pytest.raises(VLLMValidationError, match="has no audio track"):
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(
                {"video": [mm_items["video"][0]], "audio": [None]}
            ),
            mm_uuid_items={"audio": [VIDEO_AUDIO_UUID_PREFIX + "silent"]},
            hf_processor_mm_kwargs={audio_kwarg: True},
        )


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
def test_audio_in_video_partial_cache_hit(model_id: str) -> None:
    """A cached audio track must not be mistaken for a missing one.

    The second request has different video content (video cache miss) but the
    same audio content (audio cache hit). The HF processor then only sees the
    missing video, so the "video has no audio track" check must use the
    pre-cache request counts, not the post-cache item counts.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = LruKeyReplicatedSenderCache(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    first = create_mm_data(1)
    second = create_mm_data(1)
    # Different video content, identical audio content: only audio is cached.
    second["video"] = [
        random_video(
            np.random.RandomState(123), min_frames=8, max_frames=9, min_wh=64, max_wh=65
        )
    ]
    second["audio"] = first["audio"]

    def run(mm_data: dict[str, list]):
        return processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={"use_audio_in_video": True},
            cache=cache,
        )

    first_out = run(first)
    second_out = run(second)

    assert len(second_out["prompt_token_ids"]) == len(first_out["prompt_token_ids"])


@pytest.mark.parametrize("model_id", MODELS)
def test_partial_cache_hit_without_audio_data_reports_clear_error(
    model_id: str,
) -> None:
    """A new video paired with a UUID-only audio must be rejected clearly.

    The processor cache stores processed kwargs rather than raw media, so the
    audio data needed to process the new video together with its audio is not
    available. The request must fail with an actionable message instead of an
    assertion deep in the processor.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = LruKeyReplicatedSenderCache(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    processor(
        [video_token_id],
        mm_items=processor.info.parse_mm_data(create_mm_data(1)),
        mm_uuid_items={"video": ["video-0"], "audio": ["audio-0"]},
        hf_processor_mm_kwargs={"use_audio_in_video": True},
        cache=cache,
    )

    new_video = random_video(
        np.random.RandomState(123), min_frames=8, max_frames=9, min_wh=64, max_wh=65
    )
    with pytest.raises(VLLMValidationError, match="only referenced by UUID"):
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(
                {"video": [new_video], "audio": [None]}
            ),
            mm_uuid_items={"video": ["video-1"], "audio": ["audio-0"]},
            hf_processor_mm_kwargs={"use_audio_in_video": True},
            cache=cache,
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
    # The encoder parses the real video, which keys the extracted audio track
    # under a UUID derived from the video's; the decoder must resolve the same.
    uuids = {
        "video": [f"source-{i}" for i in range(num_videos)],
        "audio": [f"vllm-audio-in-video:source-{i}" for i in range(num_videos)],
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


@pytest.mark.parametrize("audio_kwarg", ["load_audio_from_video", "use_audio_in_video"])
def test_nemotron_silent_video_degrades_per_video(audio_kwarg: str):
    """A silent video must not inject an audio context in Nemotron-Omni.

    The renderer degrades a video without an audio track to a None audio
    item (see `parse_chat_messages_silent_video_degrades_to_video_only`).
    NanoNemotronVLMultiModalProcessor.apply must not treat that None as a
    real audio clip: it drops the None entries and their UUIDs so only videos
    that actually carry audio get an AUDIO_CONTEXT injected.

    Runs the real registry processor on CPU with ffmpeg-generated videos.
    """
    import av

    from vllm.transformers_utils.processors.nano_nemotron_vl import AUDIO_CONTEXT

    model_id = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 2, "video": 2},
        mm_processor_kwargs={audio_kwarg: True},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        videos_bytes: list[tuple[str, bytes]] = []
        for name, with_audio in (
            ("silent.mp4", False),
            ("with_audio.mp4", True),
        ):
            path = tmp / name
            src = ["-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=4"]
            if with_audio:
                cmd = [
                    *src,
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(path),
                ]
            else:
                cmd = [*src, "-c:v", "mpeg4", "-an", str(path)]
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *cmd],
                check=True,
            )
            videos_bytes.append((name, path.read_bytes()))

    def decode_frames(video_bytes: bytes) -> np.ndarray:
        container = av.open(BytesIO(video_bytes))
        frames = [
            np.asarray(f.to_image().resize((64, 64))) for f in container.decode(video=0)
        ]
        container.close()
        return np.stack(frames)

    def video_item(video_bytes: bytes) -> tuple:
        frames = decode_frames(video_bytes)
        # No `original_video_bytes`: the renderer-provided audio must be used
        # as is, without decoding the video a second time.
        metadata = {
            "fps": 2,
            "duration": 2.0,
            "total_num_frames": len(frames),
            "frames_indices": list(range(len(frames))),
            "video_backend": "nemotron_vl",
            "do_sample_frames": False,
        }
        return frames, metadata

    silent_item = video_item(videos_bytes[0][1])
    audible_item = video_item(videos_bytes[1][1])

    # Renderer parity: a silent video's audio is None, an audible one's is a
    # real (waveform, sample_rate) pair.
    from vllm.multimodal.media.audio import load_audio_pyav

    audible_audio = load_audio_pyav(BytesIO(videos_bytes[1][1]), max_duration_s=None)

    def run(mm_data, names, audio_uuids=None):
        mm_items = processor.info.parse_mm_data(mm_data)
        out = processor(
            tokenizer.encode("<video>" * len(names), add_special_tokens=False),
            mm_items=mm_items,
            mm_uuid_items={
                "video": [f"video-{name}" for name in names],
                "audio": audio_uuids or [f"audio-{name}" for name in names],
            },
        )
        text = tokenizer.decode(out["prompt_token_ids"], skip_special_tokens=False)
        return text, out["mm_hashes"]

    # A silent video degrades to video-only: no audio context is injected.
    # The renderer marks that sentinel with a prefixed UUID.
    silent_text, _ = run(
        {"video": [silent_item], "audio": [None]},
        ["silent"],
        [VIDEO_AUDIO_UUID_PREFIX + "silent"],
    )
    assert AUDIO_CONTEXT not in silent_text

    # A `None` audio under a plain UUID references audio sent in an earlier
    # request; it is not a silent video, and this path cannot read the data
    # back from the cache, so it must be rejected rather than dropped.
    with pytest.raises(VLLMValidationError, match="references an audio track by UUID"):
        run({"video": [silent_item], "audio": [None]}, ["silent"], ["audio-silent"])

    # Mixed request: only the audible video contributes its audio context,
    # the silent video must not add a second one.
    audible_text, audible_hashes = run(
        {"video": [audible_item], "audio": [audible_audio]}, ["audible"]
    )
    mixed_text, mixed_hashes = run(
        {"video": [silent_item, audible_item], "audio": [None, audible_audio]},
        ["silent", "audible"],
        [VIDEO_AUDIO_UUID_PREFIX + "silent", "audio-audible"],
    )
    assert audible_text.count(AUDIO_CONTEXT) > 0
    assert mixed_text.count(AUDIO_CONTEXT) == audible_text.count(AUDIO_CONTEXT)

    # The audible video's audio keeps its own UUID. Hashing it under the
    # silent video's UUID could hit another request's cache entry.
    assert mixed_hashes["audio"] == audible_hashes["audio"]


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("num_extra_audios", [0, 1])
def test_silent_video_reports_clear_error(model_id: str, num_extra_audios: int) -> None:
    """A video without an audio track must fail with an actionable message.

    The renderer keeps `mm_data`/`mm_uuids` aligned by inserting a `None`
    audio item under a `vllm-audio-in-video:` UUID. Neither the processor
    cache (which reports a misleading "Cache miss") nor the prompt updates
    (`assert audio is not None`) can consume that item, so the processor
    rejects the request before either path runs, whether or not the request
    also carries separate audio parts.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1 + num_extra_audios, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    rng = np.random.RandomState(0)
    video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)
    extra_audios = [
        random_audio(rng, min_len=8000, max_len=8001, sr=16000)
        for _ in range(num_extra_audios)
    ]

    with pytest.raises(VLLMValidationError, match="has no audio track"):
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(
                {"video": [video], "audio": [None, *extra_audios]}
            ),
            mm_uuid_items={
                "audio": ["vllm-audio-in-video:silent", *[None] * num_extra_audios]
            },
            hf_processor_mm_kwargs={"use_audio_in_video": True},
        )


@pytest.mark.parametrize("model_id", MODELS)
def test_cached_video_audio_reuse_is_not_a_silent_video(model_id: str) -> None:
    """A data-less reference to a cached video and its audio must be reused.

    `None` items with UUIDs refer to media already in the processor cache, so
    a 1:1 pairing of `None` video and audio items is not a silent video: only
    the renderer's `vllm-audio-in-video:` UUID marks one.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = LruKeyReplicatedSenderCache(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    def run(mm_data: dict[str, list]):
        return processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(mm_data),
            mm_uuid_items={"video": ["video-0"], "audio": ["audio-0"]},
            hf_processor_mm_kwargs={"use_audio_in_video": True},
            cache=cache,
        )

    cached = run(create_mm_data(1))
    reused = run({"video": [None], "audio": [None]})

    assert reused["prompt_token_ids"] == cached["prompt_token_ids"]
    assert reused["mm_hashes"] == cached["mm_hashes"]
    assert reused["mm_placeholders"] == cached["mm_placeholders"]


@pytest.mark.parametrize("model_id", MODELS)
def test_audio_only_none_audio_is_not_a_silent_video(model_id: str) -> None:
    """Audio-only uuid reuse also carries `None` items and must not be rejected.

    `use_audio_in_video` can be set engine-wide, so an audio-only request with
    an empty `input_audio.data` + uuid inherits it. Only a video-derived `None`
    audio item is a silent video; the guard must not fire without a video.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    prompt = [getattr(processor.info.get_hf_config(), "audio_token_id", 1)]

    try:
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data({"audio": [None]}),
            hf_processor_mm_kwargs={"use_audio_in_video": True},
        )
    except VLLMValidationError as exc:
        assert "has no audio track" not in str(exc)
    except Exception:  # noqa: BLE001 - downstream may reject the bare `None`
        pass


@pytest.mark.parametrize("model_id", MODELS)
def test_video_with_extra_audio_is_not_a_silent_video(model_id: str) -> None:
    """A video plus a separate audio part must not hit the silent-video guard.

    The extra audio part can itself be a uuid reuse (`None`). Without the
    renderer's `vllm-audio-in-video:` UUID it is not a silent-video sentinel.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 2, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    video_token_id = processor.info.get_hf_config().video_token_id

    rng = np.random.RandomState(0)
    video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)

    try:
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(
                {"video": [video], "audio": [None, None]}
            ),
            hf_processor_mm_kwargs={"use_audio_in_video": True},
        )
    except VLLMValidationError as exc:
        assert "has no audio track" not in str(exc)
    except Exception:  # noqa: BLE001 - downstream may reject the bare `None`
        pass
