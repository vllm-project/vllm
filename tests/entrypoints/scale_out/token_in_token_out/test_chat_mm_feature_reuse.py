# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat-path tests for forwarded multimodal feature reuse.

The generate endpoint's validations (SHM cache-handle authentication,
metadata-only/EC consistency, placeholder bounds) are shared with the
chat reuse path via ``engine_input_from_features``; these tests pin the
chat side so it cannot drift. Importing this module is itself the smoke
test that ``vllm.renderers.online_renderer`` imports cleanly (the
``MultiModalFeatures`` return annotation is TYPE_CHECKING-only).
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    engine_input_from_features,
)
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.renderers.online_renderer import (
    OnlineRenderer,
    _reused_prompt_token_ids,
)

MODEL_NAME = "openai-community/gpt2"


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _valid_features(
    token_len: int = 3,
) -> MultiModalFeatures:
    return MultiModalFeatures(
        mm_hashes={"image": ["image_A"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=1)]},
        kwargs_data={
            "image": [
                encode_mm_kwargs_item(
                    MultiModalKwargsItem(
                        {
                            "pixel_values": MultiModalFieldElem(
                                data=[1.0], field=MultiModalBatchedField()
                            ),
                        }
                    )
                )
            ]
        },
    )


def _fake_renderer() -> MagicMock:
    renderer = MagicMock()
    renderer.validate_token_ids = MagicMock()
    renderer.mm_processor_cache = None
    return renderer


class _FakeRequest:
    """Minimal request stand-in for ``_reused_prompt_token_ids``."""

    def __init__(
        self,
        kv: dict[str, Any] | None = None,
        echo: bool = False,
        messages: list[Any] | None = None,
    ):
        self.kv_transfer_params = kv
        self.echo = echo
        self.messages = messages


# ---------------------------------------------------------------------------
# _reused_prompt_token_ids
# ---------------------------------------------------------------------------


def test_features_without_ids_warns_and_is_ignored(caplog):
    request = _FakeRequest(kv={"prompt_features": {"mm_hashes": {}}})
    with caplog.at_level("WARNING"):
        ids, features = _reused_prompt_token_ids(request, _fake_renderer())
    assert ids is None and features is None
    assert "prompt_token_ids is absent" in caplog.text


def test_echo_skips_feature_validation():
    # Malformed features must not 400 a request that ignores them anyway.
    request = _FakeRequest(
        kv={"prompt_token_ids": [1, 2, 3], "prompt_features": "not-a-dict"},
        echo=True,
    )
    ids, features = _reused_prompt_token_ids(request, _fake_renderer())
    assert ids is None and features is None


def test_malformed_features_raises_validation_error():
    request = _FakeRequest(
        kv={"prompt_token_ids": [1, 2, 3], "prompt_features": "not-a-dict"}
    )
    with pytest.raises(VLLMValidationError, match="prompt_features"):
        _reused_prompt_token_ids(request, _fake_renderer())


def test_non_text_with_features_keeps_reuse_path():
    request = _FakeRequest(
        kv={
            "prompt_token_ids": [1, 2, 3],
            "prompt_features": _valid_features().model_dump(),
        },
        messages=[{"role": "user", "content": [{"type": "image_url"}]}],
    )
    ids, features = _reused_prompt_token_ids(
        request, _fake_renderer(), request.messages
    )
    assert ids == [1, 2, 3]
    assert features is not None
    assert features.mm_hashes == {"image": ["image_A"]}


def test_non_text_without_features_drops_ids():
    request = _FakeRequest(
        kv={"prompt_token_ids": [1, 2, 3]},
        messages=[{"role": "user", "content": [{"type": "image_url"}]}],
    )
    ids, features = _reused_prompt_token_ids(
        request, _fake_renderer(), request.messages
    )
    assert ids is None and features is None


# ---------------------------------------------------------------------------
# engine_input_from_features (shared by generate and chat)
# ---------------------------------------------------------------------------


def test_engine_input_from_features_success_shape():
    features = _valid_features()
    engine_input = engine_input_from_features(
        [5, 6, 7], features, cache_salt="salt", mm_cache=None
    )
    assert engine_input["type"] == "multimodal"
    assert engine_input["prompt_token_ids"] == [5, 6, 7]
    assert engine_input["mm_hashes"] == {"image": ["image_A"]}
    assert engine_input["mm_placeholders"]["image"][0].offset == 0
    assert engine_input["mm_placeholders"]["image"][0].length == 1
    assert engine_input["cache_salt"] == "salt"


def test_engine_input_from_features_rejects_out_of_bounds():
    features = MultiModalFeatures(
        mm_hashes={"image": ["image_A"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=200, length=64)]},
    )
    with pytest.raises(ValueError, match="within the token_ids sequence"):
        engine_input_from_features([1] * 100, features, mm_cache=None)


def test_engine_input_from_features_rejects_metadata_only_without_ec():
    features = MultiModalFeatures(
        mm_hashes={"image": ["image_A"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=1)]},
        kwargs_data={"image": [None]},
        mm_metadata={
            "image": [
                encode_mm_kwargs_item(
                    MultiModalKwargsItem(
                        {
                            "image_grid_thw": MultiModalFieldElem(
                                data=[1], field=MultiModalBatchedField()
                            ),
                        }
                    )
                )
            ]
        },
    )
    with pytest.raises(ValueError, match="metadata-only"):
        engine_input_from_features([1, 2, 3], features, mm_cache=None)
    # With EC transfer configured the same payload is accepted.
    engine_input = engine_input_from_features(
        [1, 2, 3], features, mm_cache=None, ec_transfer_params={"mode": "x"}
    )
    assert engine_input["type"] == "multimodal"


def test_engine_input_from_features_authenticates_shm_handles():
    features = _valid_features()
    cache = MagicMock()
    cache.validate_input_item.side_effect = ValueError(
        "Invalid SHM handle signature for cache key."
    )
    with pytest.raises(ValueError, match="SHM handle"):
        engine_input_from_features([1, 2, 3], features, mm_cache=cache)
    (validated_item, validated_hash), _ = cache.validate_input_item.call_args
    assert validated_hash == "image_A"
    assert "pixel_values" in validated_item


# ---------------------------------------------------------------------------
# Chat-path integration: preprocess_chat with forwarded features
# ---------------------------------------------------------------------------


def _chat_request(kv: dict[str, Any], with_image: bool = True) -> ChatCompletionRequest:
    if with_image:
        content: Any = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
    else:
        content = "hello"
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=1,
        kv_transfer_params=dict(kv),
    )


def _online_renderer(cache: MagicMock | None = None) -> OnlineRenderer:
    renderer = _fake_renderer()
    renderer.tokenizer = MagicMock()
    renderer.render_chat_async = AsyncMock(return_value=([], []))
    if cache is not None:
        renderer.mm_processor_cache = cache
    return OnlineRenderer(
        model_config=MockModelConfig(),
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


@pytest.mark.asyncio
async def test_preprocess_chat_uses_shared_builder_with_features():
    """Forwarded ids + features skip rendering and build a multimodal input."""
    online_renderer = _online_renderer()
    request = _chat_request(
        {
            "prompt_token_ids": [5, 6, 7],
            "prompt_features": _valid_features().model_dump(),
        }
    )
    conversation, engine_inputs = await online_renderer.preprocess_chat(
        request, request.messages, None, "auto", {}
    )
    assert conversation == []
    assert len(engine_inputs) == 1
    assert engine_inputs[0]["type"] == "multimodal"
    assert engine_inputs[0]["prompt_token_ids"] == [5, 6, 7]
    online_renderer.renderer.render_chat_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_preprocess_chat_rejects_forged_shm_handle():
    cache = MagicMock()
    cache.validate_input_item.side_effect = ValueError(
        "Invalid SHM handle signature for cache key."
    )
    online_renderer = _online_renderer(cache=cache)
    request = _chat_request(
        {
            "prompt_token_ids": [5, 6, 7],
            "prompt_features": _valid_features().model_dump(),
        }
    )
    with pytest.raises(VLLMValidationError, match="prompt_features"):
        await online_renderer.preprocess_chat(
            request, request.messages, None, "auto", {}
        )
    cache.validate_input_item.assert_called_once()


@pytest.mark.asyncio
async def test_preprocess_chat_rejects_out_of_bounds_placeholders():
    features = MultiModalFeatures(
        mm_hashes={"image": ["image_A"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=200, length=64)]},
    )
    online_renderer = _online_renderer()
    request = _chat_request(
        {
            "prompt_token_ids": [5, 6, 7],
            "prompt_features": features.model_dump(),
        }
    )
    with pytest.raises(VLLMValidationError, match="within the token_ids"):
        await online_renderer.preprocess_chat(
            request, request.messages, None, "auto", {}
        )


def test_preprocess_chat_harmony_drops_ids_with_features(monkeypatch):
    """Harmony is not a vision path: ids forwarded with features are dropped
    and messages are re-rendered, matching pre-feature behavior."""
    online_renderer = _online_renderer()

    def _tokens_input(prompt_token_ids, cache_salt=None):
        return {"type": "token", "prompt_token_ids": prompt_token_ids}

    tokens_input = MagicMock(side_effect=_tokens_input)
    render_for_completion = MagicMock(return_value=[8, 9])
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.tokens_input", tokens_input
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion",
        render_for_completion,
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.build_harmony_preamble",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.parse_chat_inputs_to_harmony_messages",
        MagicMock(return_value=[]),
    )

    # Text-only ids: fast path, forwarded ids go straight to the engine.
    request = _chat_request({"prompt_token_ids": [1, 2, 3]}, with_image=False)
    messages, engine_inputs = online_renderer._make_request_with_harmony(request)
    assert tokens_input.call_count == 1
    assert engine_inputs[0]["prompt_token_ids"] == [1, 2, 3]
    render_for_completion.assert_not_called()

    # Ids + features: ids dropped, messages re-rendered instead. The engine
    # input now comes from render_for_completion, not the forwarded ids.
    request = _chat_request(
        {
            "prompt_token_ids": [1, 2, 3],
            "prompt_features": _valid_features().model_dump(),
        },
        with_image=False,
    )
    messages, engine_inputs = online_renderer._make_request_with_harmony(request)
    assert render_for_completion.call_count == 1
    assert tokens_input.call_count == 2  # slow path re-builds from rendered ids
    assert tokens_input.call_args.args[0] == [8, 9]
    assert engine_inputs[0]["prompt_token_ids"] == [8, 9]
