# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from copy import deepcopy
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.entrypoints.serve import create_error_response
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.parse import parse_mm_uuids
from vllm.multimodal.processing.processor import MultiModalProcessorCacheMissError
from vllm.renderers.base import BaseRenderer
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers.registry import cached_tokenizer_from_config

cherry_pil_image = ImageAsset("cherry_blossom").pil_image
stop_pil_image = ImageAsset("stop_sign").pil_image
baby_reading_np_ndarrays = VideoAsset("baby_reading").np_ndarrays
kimi_vision_chunk_image = {
    "type": "image",
    "image": cherry_pil_image,
    "uuid": "test-image-uuid",
}
kimi_vision_chunk_video = {
    "type": "video_chunk",
    "video_chunk": [Image.fromarray(frame) for frame in baby_reading_np_ndarrays[:4]],
    "uuid": "test-video-uuid",
    "video_idx": 0,
    "prompt": ("<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"),
}


def _build_renderer(
    *,
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    mm_cache_gb: float = 4.0,
    enable_prefix_caching: bool = True,
    trust_remote_code: bool = False,
) -> HfRenderer:
    model_config = ModelConfig(
        model=model,
        max_model_len=128,
        mm_processor_cache_gb=mm_cache_gb,
        trust_remote_code=trust_remote_code,
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
        pytest.param("video", baby_reading_np_ndarrays, False, id="video"),
        pytest.param(
            "video",
            baby_reading_np_ndarrays,
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
    media_url = f"https://example.com/test.{modality}"
    media_uuid = f"test-{modality}-uuid"
    fetch_media = _mock_media(monkeypatch, modality, media, use_async)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this {modality}."},
                {
                    "type": f"{modality}_url",
                    f"{modality}_url": {"url": media_url},
                    "uuid": media_uuid,
                },
            ],
        }
    ]

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
        pytest.param("video", baby_reading_np_ndarrays, id="video"),
    ],
)
def test_uuid_cache_eviction_falls_back_to_url(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    media: object,
):
    renderer = _build_renderer()
    mm_processor_cache = renderer.mm_processor_cache
    assert mm_processor_cache is not None

    fetch_media = Mock(return_value=media)
    monkeypatch.setattr(MediaConnector, f"fetch_{modality}", fetch_media)

    media_url = f"https://example.com/test.{modality}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this {modality}."},
                {
                    "type": f"{modality}_url",
                    f"{modality}_url": {"url": media_url},
                    "uuid": f"test-{modality}-uuid",
                },
            ],
        }
    ]

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
    fetch_video = _mock_media(monkeypatch, "video", baby_reading_np_ndarrays, use_async)
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
    ].count("Cached conversation") == 1
    assert render_messages.call_count == 3
    fetch_image.assert_called_once_with("https://example.com/partial-image.image")
    assert fetch_video.call_count == 1
    assert prompts[0]["prompt_token_ids"] == warm_prompts[0]["prompt_token_ids"]
    assert prompts[0]["mm_hashes"] == warm_prompts[0]["mm_hashes"]
    assert set(prompts[1]["mm_hashes"]) == {"image", "video"}
    assert conversations == original

    fetch_image.reset_mock()
    fetch_video.reset_mock()
    _, cached_prompts = _render_chat(renderer, conversations, params, use_async)
    fetch_image.assert_not_called()
    fetch_video.assert_not_called()
    assert [p["prompt_token_ids"] for p in cached_prompts] == [
        p["prompt_token_ids"] for p in prompts
    ]


def test_async_uuid_render_preserves_cache_update_order(monkeypatch):
    """Sender cache updates must follow the order of returned conversations."""
    renderer = _build_renderer()
    conversations = [
        _media_messages("first", ("image", "shared")),
        _media_messages("second", ("image", "shared")),
    ]
    completed = []

    async def render(_self, batch, **kwargs):
        name = batch[0][0]["content"][0]["text"]
        if name == "first":
            await asyncio.sleep(0)
        completed.append(name)
        return [[]], [{"prompt": name}]

    monkeypatch.setattr(BaseRenderer, "render_chat_async", render)
    _, prompts = _render_chat(renderer, conversations, ChatParams(), True)

    assert completed == [prompt["prompt"] for prompt in prompts] == ["first", "second"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_missing_uuid_loads_all_media(monkeypatch, use_async):
    renderer = _build_renderer()
    fetch_image = _mock_media(monkeypatch, "image", cherry_pil_image, use_async)
    messages = _media_messages("Describe", ("image", "cached-image"))
    _render_chat(renderer, [messages], ChatParams(), use_async)
    messages[0]["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/no-uuid.image"},
        }
    )
    fetch_image.reset_mock()

    _render_chat(renderer, [messages], ChatParams(), use_async)

    assert fetch_image.call_count == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_uuid_without_url_still_requires_cache_hit(monkeypatch, use_async):
    renderer = _build_renderer()
    fetch_image = _mock_media(monkeypatch, "image", cherry_pil_image, use_async)
    messages = _media_messages("Describe", ("image", "missing-image"))
    messages[0]["content"][1]["image_url"]["url"] = None

    with pytest.raises(MultiModalProcessorCacheMissError, match="data is not provided"):
        _render_chat(renderer, [messages], ChatParams(), use_async)

    fetch_image.assert_not_called()


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "case", ["skip-cache", "no-cache", "audio-in-video", "other-media"]
)
def test_ineligible_conversation_renders_original_messages(
    monkeypatch, use_async, case
):
    renderer = _build_renderer(mm_cache_gb=0 if case == "no-cache" else 4)
    messages = _media_messages("Describe", ("image", "image"))
    params = ChatParams()
    if case == "audio-in-video":
        params = ChatParams(mm_processor_kwargs={"use_audio_in_video": True})
    elif case == "other-media":
        messages[0]["content"].append(
            {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/audio.wav"},
                "uuid": "audio",
            }
        )
    mock_type = AsyncMock if use_async else Mock
    render = mock_type(return_value=([[]], [{}]))
    method = "render_chat_async" if use_async else "render_chat"
    monkeypatch.setattr(BaseRenderer, method, render)

    _render_chat(
        renderer, [messages], params, use_async, skip_mm_cache=case == "skip-cache"
    )

    assert render.call_count == 1
    assert render.call_args.args[0][0] is messages


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "error", [ValueError("invalid media"), MultiModalProcessorCacheMissError("missing")]
)
def test_uuid_render_retry_is_bounded_and_only_handles_cache_misses(
    monkeypatch, use_async, error
):
    renderer = _build_renderer()
    messages = _media_messages("Describe", ("image", "image"))
    mock_type = AsyncMock if use_async else Mock
    render = mock_type(side_effect=error)
    method = "render_chat_async" if use_async else "render_chat"
    monkeypatch.setattr(BaseRenderer, method, render)

    with pytest.raises(type(error), match=str(error)):
        _render_chat(renderer, [messages], ChatParams(), use_async)

    assert render.call_count == (
        2 if isinstance(error, MultiModalProcessorCacheMissError) else 1
    )


@pytest.mark.parametrize(
    ("modality", "media"),
    [
        pytest.param("image", kimi_vision_chunk_image, id="image"),
        pytest.param("video", kimi_vision_chunk_video, id="video"),
    ],
)
def test_early_uuid_lookup_is_disabled_for_unified_vision_chunks(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    media: object,
):
    renderer = _build_renderer(
        model="moonshotai/Kimi-K2.5",
        trust_remote_code=True,
    )
    mm_processor_cache = renderer.mm_processor_cache
    assert mm_processor_cache is not None

    is_cached_item = Mock(return_value=True)
    monkeypatch.setattr(mm_processor_cache, "is_cached_item", is_cached_item)
    fetch_media = Mock(return_value=media)
    monkeypatch.setattr(MediaConnector, f"fetch_{modality}", fetch_media)

    media_url = f"https://example.com/test.{modality}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this {modality}."},
                {
                    "type": f"{modality}_url",
                    f"{modality}_url": {"url": media_url},
                    "uuid": f"test-{modality}-uuid",
                },
            ],
        }
    ]

    monkeypatch.setattr(
        renderer, "process_for_engine", lambda prompt, *args, **kwargs: prompt
    )
    conversations, prompts = renderer.render_chat([messages], ChatParams())

    assert len(conversations[0]) == 1
    assert prompts[0]["multi_modal_data"] == {"vision_chunk": [media]}
    assert fetch_media.call_count == 1
    is_cached_item.assert_not_called()


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
        "video": baby_reading_np_ndarrays,
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


def test_multi_modal_uuids_ignored_when_caching_disabled():
    # When both processor cache is 0 and prefix caching disabled, the
    # processor builds overrides from request id instead of using user UUIDs.
    renderer = _build_renderer(mm_cache_gb=0.0, enable_prefix_caching=False)

    request_id = "req-42"
    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }
    mm_uuids = {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, request_id
    )

    # Expect request-id-based overrides are passed through
    assert set(mm_uuids.keys()) == {"image", "video"}
    assert len(mm_uuids["image"]) == 2
    assert len(mm_uuids["video"]) == 1
    assert processed_mm_uuids["image"][0].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][0].endswith("-0")
    assert processed_mm_uuids["image"][1].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][1].endswith("-1")
    assert processed_mm_uuids["video"][0].startswith(
        f"{request_id}-video-"
    ) and processed_mm_uuids["video"][0].endswith("-0")
