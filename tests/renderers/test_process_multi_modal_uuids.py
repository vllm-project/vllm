# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from copy import deepcopy
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.entrypoints.serve import create_error_response
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.parse import parse_mm_uuids
from vllm.multimodal.processing.processor import MultiModalProcessorCacheMissError
from vllm.renderers.chat_utils import can_conversation_defer_media
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers.registry import cached_tokenizer_from_config

cherry_pil_image = ImageAsset("cherry_blossom").pil_image
stop_pil_image = ImageAsset("stop_sign").pil_image
baby_reading_video_asset = VideoAsset("baby_reading")
baby_reading_video = (
    baby_reading_video_asset.np_ndarrays,
    baby_reading_video_asset.metadata,
)


def _build_renderer(
    *, mm_cache_gb: float = 4.0, enable_prefix_caching: bool = True
) -> HfRenderer:
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=128,
        mm_processor_cache_gb=mm_cache_gb,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=enable_prefix_caching),
    )

    return HfRenderer(
        vllm_config,
        cached_tokenizer_from_config(model_config),
    )


def _build_text_only_renderer() -> HfRenderer:
    model_config = ModelConfig(model="openai-community/gpt2", max_model_len=128)

    return HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )


def test_text_only_model_mm_data_maps_to_bad_request():
    """Sending multimodal data to a text-only model is a client mistake, so it
    must surface as a ValueError and reach the client as HTTP 400, not 500."""
    renderer = _build_text_only_renderer()

    with pytest.raises(ValueError, match="text-only") as exc_info:
        renderer._process_multimodal(
            prompt=[1],
            mm_data={"image": [cherry_pil_image]},
            mm_uuids=None,
            mm_processor_kwargs=None,
        )

    error_response = create_error_response(exc_info.value)
    assert error_response.error.code == HTTPStatus.BAD_REQUEST


def _render_chat(renderer, conversations, params, use_async, **kwargs):
    if use_async:
        return asyncio.run(renderer.render_chat_async(conversations, params, **kwargs))
    return renderer.render_chat(conversations, params, **kwargs)


def _mock_media(monkeypatch, modality, media, use_async):
    fetch = AsyncMock(return_value=media) if use_async else Mock(return_value=media)
    method = f"fetch_{modality}" + ("_async" if use_async else "")
    monkeypatch.setattr(MediaConnector, method, fetch)
    return fetch


def _media_messages(name, *items):
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": name}]
            + [
                {
                    "type": f"{modality}_url",
                    f"{modality}_url": {
                        "url": f"https://example.com/{uuid}.{modality}"
                    },
                    "uuid": uuid,
                }
                for modality, uuid in items
            ],
        }
    ]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("modality", "media", "skip_early_mm_lookup"),
    [
        pytest.param("image", cherry_pil_image, False, id="image"),
        pytest.param("video", baby_reading_video, False, id="video"),
        pytest.param(
            "video",
            baby_reading_video,
            True,
            id="video-skip-early-mm-lookup",
        ),
    ],
)
def test_cached_uuid_skips_url_loading(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    media: object,
    skip_early_mm_lookup: bool,
    use_async: bool,
):
    renderer = _build_renderer()
    fetch_media = _mock_media(monkeypatch, modality, media, use_async)
    messages = _media_messages("Describe", (modality, f"test-{modality}-uuid"))

    params = ChatParams(skip_early_mm_lookup=skip_early_mm_lookup)
    _, first_prompts = _render_chat(renderer, [messages], params, use_async)
    _, second_prompts = _render_chat(renderer, [messages], params, use_async)

    first_input = first_prompts[0]
    second_input = second_prompts[0]
    assert first_input["mm_hashes"] == second_input["mm_hashes"]
    assert first_input["prompt_token_ids"] == second_input["prompt_token_ids"]
    assert first_input["mm_placeholders"] == second_input["mm_placeholders"]
    assert fetch_media.call_count == (2 if skip_early_mm_lookup else 1)


@pytest.mark.parametrize(
    ("modality", "media"),
    [
        pytest.param("image", cherry_pil_image, id="image"),
        pytest.param("video", baby_reading_video, id="video"),
    ],
)
def test_uuid_cache_eviction_falls_back_to_url(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    media: object,
):
    renderer = _build_renderer()
    fetch_media = _mock_media(monkeypatch, modality, media, use_async=False)
    messages = _media_messages("Describe", (modality, f"test-{modality}-uuid"))

    renderer.render_chat([messages], ChatParams())
    renderer.clear_mm_cache()
    fetch_media.reset_mock()

    _, prompts = renderer.render_chat([messages], ChatParams())

    assert len(prompts[0]["mm_hashes"][modality]) == 1
    assert fetch_media.call_count == 1


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_partial_uuid_hit_retries_only_affected_conversation(monkeypatch, use_async):
    """A miss reloads all URLs in its conversation without rerendering a batch peer."""
    renderer = _build_renderer()
    fetch_image = _mock_media(monkeypatch, "image", cherry_pil_image, use_async)
    fetch_video = _mock_media(monkeypatch, "video", baby_reading_video, use_async)
    cached = _media_messages("Cached conversation", ("image", "cached-image"))
    partial = _media_messages("Partial conversation", ("image", "partial-image"))
    params = ChatParams()
    _, warm_prompts = _render_chat(renderer, [cached, partial], params, use_async)
    partial[0]["content"].append(
        _media_messages("", ("video", "missing-video"))[0]["content"][1]
    )
    conversations = [cached, partial]
    original = deepcopy(conversations)
    fetch_image.reset_mock()
    fetch_video.reset_mock()
    method = "render_messages_async" if use_async else "render_messages"
    mock_type = AsyncMock if use_async else Mock
    render_messages = mock_type(wraps=getattr(renderer, method))
    monkeypatch.setattr(renderer, method, render_messages)

    _, prompts = _render_chat(renderer, conversations, params, use_async)

    assert [
        call.args[0][0]["content"][0]["text"] for call in render_messages.call_args_list
    ] == ["Partial conversation"]
    fetch_image.assert_called_once_with("https://example.com/partial-image.image")
    assert fetch_video.call_count == 1
    assert prompts[0]["prompt_token_ids"] == warm_prompts[0]["prompt_token_ids"]
    assert prompts[0]["mm_hashes"] == warm_prompts[0]["mm_hashes"]
    assert set(prompts[1]["mm_hashes"]) == {"image", "video"}
    assert conversations == original


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"}],
            True,
            id="image-url",
        ),
        pytest.param(
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AAAA", "format": "wav"},
                    "uuid": "audio",
                }
            ],
            True,
            id="input-audio",
        ),
        pytest.param(
            [{"type": "image_url", "image_url": None, "uuid": "image"}],
            False,
            id="uuid-without-data",
        ),
        pytest.param(
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "", "format": "wav"},
                    "uuid": "audio",
                }
            ],
            False,
            id="input-audio-without-data",
        ),
        pytest.param(
            [
                {"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"},
                {"type": "video_url", "video_url": {"url": "url"}},
            ],
            False,
            id="missing-uuid",
        ),
        pytest.param("Describe", False, id="text-only"),
    ],
)
def test_can_conversation_defer_media(content, expected):
    """Media loading is deferred only if all media has UUIDs and some has data."""
    messages = [{"role": "user", "content": content}]

    assert can_conversation_defer_media(messages) is expected


def test_can_conversation_defer_media_does_not_consume_iterators():
    part = {"type": "image_url", "image_url": {"url": "url"}, "uuid": "image"}
    content = iter([part])

    assert not can_conversation_defer_media([{"role": "user", "content": content}])
    assert next(content) == part


def _wrap_process_for_engine(monkeypatch, renderer, use_async):
    method = "process_for_engine_async" if use_async else "process_for_engine"
    mock_type = AsyncMock if use_async else Mock
    process = mock_type(wraps=getattr(renderer, method))
    monkeypatch.setattr(renderer, method, process)
    return process


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "case", ["skip-cache", "no-cache", "audio-in-video", "missing-uuid"]
)
def test_ineligible_conversation_loads_media_upfront(monkeypatch, use_async, case):
    """Conversations that cannot defer media are processed once, not retried."""
    renderer = _build_renderer(mm_cache_gb=0 if case == "no-cache" else 4)
    fetch_image = _mock_media(monkeypatch, "image", cherry_pil_image, use_async)
    messages = _media_messages("Describe", ("image", "image"))
    if case == "missing-uuid":
        del messages[0]["content"][1]["uuid"]
    params = ChatParams()
    if case == "audio-in-video":
        params = ChatParams(mm_processor_kwargs={"use_audio_in_video": True})
    process = _wrap_process_for_engine(monkeypatch, renderer, use_async)

    _render_chat(
        renderer,
        [messages],
        params,
        use_async,
        skip_mm_cache=case == "skip-cache",
    )

    assert process.call_count == 1
    assert fetch_image.call_count == 1


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_uuid_only_cache_miss_is_not_retried(monkeypatch, use_async):
    """Without a URL to fall back to, a cache miss fails without a retry."""
    renderer = _build_renderer()
    messages = _media_messages("Describe", ("image", "image"))
    messages[0]["content"][1]["image_url"] = None
    process = _wrap_process_for_engine(monkeypatch, renderer, use_async)

    with pytest.raises(MultiModalProcessorCacheMissError):
        _render_chat(renderer, [messages], ChatParams(), use_async)

    assert process.call_count == 1


def test_batch_cache_misses_reload_media_concurrently(monkeypatch):
    """Retried conversations load media concurrently and keep the batch order."""
    renderer = _build_renderer()
    active_fetches = max_active_fetches = 0

    async def fetch_image(*args, **kwargs):
        nonlocal active_fetches, max_active_fetches
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        try:
            await asyncio.sleep(0.01)
            return cherry_pil_image
        finally:
            active_fetches -= 1

    monkeypatch.setattr(MediaConnector, "fetch_image_async", fetch_image)
    conversations = [
        _media_messages(f"Conversation {i}", ("image", f"image-{i}")) for i in range(2)
    ]

    out_conversations, prompts = _render_chat(
        renderer, conversations, ChatParams(), use_async=True
    )
    _, cached_prompts = _render_chat(
        renderer, conversations, ChatParams(), use_async=True
    )

    assert max_active_fetches == 2
    assert all(
        f"Conversation {i}" in str(conversation)
        for i, conversation in enumerate(out_conversations)
    )
    assert [p["mm_hashes"] for p in prompts] == [p["mm_hashes"] for p in cached_prompts]


def test_multi_modal_uuids_length_mismatch_raises():
    renderer = _build_renderer()

    mm_data = {"image": [cherry_pil_image, stop_pil_image]}

    # Mismatch: 2 items but only 0 uuids provided
    mm_uuids = {"image": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1a")

    # Mismatch: 2 items but only 1 uuid provided
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1b")


def test_multi_modal_uuids_missing_modality_raises():
    renderer = _build_renderer()

    mm_data = {
        "image": [cherry_pil_image],
        "video": None,
    }

    # Only image uuids provided; video missing should raise
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="is empty but .* is missing"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-2")


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_none_and_passes_through(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_video,
    }

    # Use a consistent two-image scenario across all configurations
    mm_uuids = {"image": [None, "hash_stop"], "video": None}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-3"
    )

    assert processed_mm_uuids == mm_uuids


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_empty(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    # While None means cached multi-modal input requiring UUIDs
    # an empty list means no multi-modal input
    mm_data = {"image": [], "video": [], "audio": None}  # type: ignore[var-annotated]
    mm_uuids = {"image": [], "video": None, "audio": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-4"
    )

    assert processed_mm_uuids == mm_uuids


@pytest.mark.parametrize(
    "mm_uuids, expected",
    [
        (
            {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]},
            {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]},
        ),
        (
            {"image": [None, "hash_stop"], "video": None},
            {"image": ["req-42-image-0", "hash_stop"], "video": ["req-42-video-0"]},
        ),
        (
            {"image": ["", None]},
            {"image": ["", "req-42-image-1"], "video": ["req-42-video-0"]},
        ),
        (
            {},
            {
                "image": ["req-42-image-0", "req-42-image-1"],
                "video": ["req-42-video-0"],
            },
        ),
    ],
)
def test_multi_modal_uuids_preserved_when_caching_disabled(mm_uuids, expected):
    """Only missing UUIDs get request-local IDs when both caches are disabled."""
    renderer = _build_renderer(mm_cache_gb=0.0, enable_prefix_caching=False)

    request_id = "req-42"
    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_video,
    }

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, request_id
    )

    assert processed_mm_uuids == expected
