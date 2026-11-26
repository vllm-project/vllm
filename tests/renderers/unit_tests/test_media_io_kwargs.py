# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import dataclasses
import sys

import pytest

from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.multimodal.utils import encode_video_url
from vllm.renderers.base import BaseRenderer
from vllm.renderers.chat_utils import ChatCompletionMessageParam
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers.registry import cached_tokenizer_from_config

# Sentinel distinguishing "key absent" from an explicit `None`/`{}`.
_UNSET = object()

DEFAULT_KW = {"image": {"num_frames": 32, "fps": 2}}
REQ_KW = {"image": {"num_frames": 16}}
MERGED_KW = {"image": {"num_frames": 16, "fps": 2}}


# =========================================================================
# Stub renderer -- skips the heavyweight BaseRenderer.__init__
# =========================================================================
class _StubRenderer(BaseRenderer):
    """Captures what the downstream stages of each render path receive."""

    def __init__(self, rendered_prompt=None):
        # Deliberately no super().__init__: none of the tested paths touch
        # the resources, tokenizer, or cache created there.
        self._applied_extras: list[dict] = []
        self._chat_params_seen: list[ChatParams] = []
        self._eng_prompts: list = []
        # The dict `render_messages` hands back. The chat-path merge helper
        # only touches prompts that carry multimodal payload, so tests for it
        # need a prompt that has `multi_modal_data`/`multi_modal_uuids`.
        self._rendered_prompt = (
            {"prompt_token_ids": [1]} if rendered_prompt is None else rendered_prompt
        )

    # -- chat downstream stages --------------------------------------
    def render_messages(self, messages, params):
        self._chat_params_seen.append(params)
        # Shallow copy so the caller's fixture dict is never mutated by the
        # chat merge helper.
        return [], dict(self._rendered_prompt)

    # -- shared downstream stages ------------------------------------
    def tokenize_prompts(self, prompts, params):
        return list(prompts)

    async def tokenize_prompts_async(self, prompts, params):
        return list(prompts)

    def process_for_engine(
        self, prompt, arrival_time, *, skip_mm_cache=False, tok_params=None
    ):
        self._eng_prompts.append(prompt)
        return prompt

    async def process_for_engine_async(
        self, prompt, arrival_time, *, skip_mm_cache=False, tok_params=None
    ):
        self._eng_prompts.append(prompt)
        return prompt

    def _apply_prompt_extras(self, prompts, prompt_extras):
        self._applied_extras.append(dict(prompt_extras or {}))


# =========================================================================
# Builders / spies
# =========================================================================
def _make_chat_params(media_io_kwargs=None) -> ChatParams:
    kwargs = {}
    for f in dataclasses.fields(ChatParams):
        if f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            kwargs[f.name] = f.default_factory()
    kwargs["media_io_kwargs"] = media_io_kwargs
    return ChatParams(**kwargs)


def _make_extras(media_io_kwargs):
    if media_io_kwargs is _UNSET:
        return {}
    return {"media_io_kwargs": media_io_kwargs}


def _install_merge_spy(monkeypatch):
    """Record `(defaults, overrides)` passed to `merge_media_io_kwargs`."""
    calls = []

    def spy(defaults, overrides):
        calls.append((defaults, overrides))
        return merge_media_io_kwargs(defaults, overrides)

    base_mod = sys.modules[BaseRenderer.__module__]
    monkeypatch.setattr(base_mod, "merge_media_io_kwargs", spy)
    return calls


# =========================================================================
# Completion path
# =========================================================================
class TestCmplPath:
    """`render_cmpl` / `render_cmpl_async` contract.

    Request-level `media_io_kwargs` are merged into every multimodal
    completion prompt; `prompt_embeds` and text-only prompts are left
    untouched.  Priority is request > prompt default.
    """

    @staticmethod
    def _run(prompts, media_io_kwargs=_UNSET, is_async=False):
        renderer = _StubRenderer()
        extras = _make_extras(media_io_kwargs)
        tok_params = object()  # never inspected: all stages are stubbed
        if is_async:
            asyncio.run(
                renderer.render_cmpl_async(prompts, tok_params, prompt_extras=extras)
            )
        else:
            renderer.render_cmpl(prompts, tok_params, prompt_extras=extras)
        return renderer

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_multi_modal_data_merged(self, is_async):
        r = self._run([{"multi_modal_data": {"image": "<img>"}}], REQ_KW, is_async)
        assert r._eng_prompts[0]["media_io_kwargs"] == REQ_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_multi_modal_uuids_merged(self, is_async):
        r = self._run([{"multi_modal_uuids": {"image": ["uuid-0"]}}], REQ_KW, is_async)
        assert r._eng_prompts[0]["media_io_kwargs"] == REQ_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_text_only_untouched(self, is_async):
        r = self._run([{"prompt_token_ids": [1, 2]}], REQ_KW, is_async)
        assert "media_io_kwargs" not in r._eng_prompts[0]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_prompt_embeds_skipped(self, is_async):
        r = self._run([{"prompt_embeds": [[0.1, 0.2]]}], REQ_KW, is_async)
        assert "media_io_kwargs" not in r._eng_prompts[0]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_request_overrides_prompt_defaults(self, monkeypatch, is_async):
        calls = _install_merge_spy(monkeypatch)
        prompt = {
            "multi_modal_data": {"image": "<img>"},
            "media_io_kwargs": DEFAULT_KW,
        }
        r = self._run([prompt], REQ_KW, is_async)

        assert r._eng_prompts[0]["media_io_kwargs"] == MERGED_KW
        # (defaults=prompt-level, overrides=request-level)
        assert calls[-1] == (DEFAULT_KW, REQ_KW)

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_unset_request_keeps_prompt_defaults(self, is_async):
        prompt = {
            "multi_modal_data": {"image": "<img>"},
            "media_io_kwargs": DEFAULT_KW,
        }
        r = self._run([prompt], _UNSET, is_async)
        assert r._eng_prompts[0]["media_io_kwargs"] == DEFAULT_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_empty_dict_is_noop(self, is_async):
        prompt = {
            "multi_modal_data": {"image": "<img>"},
            "media_io_kwargs": DEFAULT_KW,
        }
        r = self._run([prompt], {}, is_async)
        assert r._eng_prompts[0]["media_io_kwargs"] == DEFAULT_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_enc_dec_targets_merged_independently(self, is_async):
        enc = {"multi_modal_data": {"image": "<img>"}}
        dec = {"multi_modal_uuids": {"image": ["uuid-0"]}}
        prompt = {"encoder_prompt": enc, "decoder_prompt": dec}
        r = self._run([prompt], REQ_KW, is_async)

        out = r._eng_prompts[0]
        assert out["encoder_prompt"]["media_io_kwargs"] == REQ_KW
        assert out["decoder_prompt"]["media_io_kwargs"] == REQ_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_mixed_batch(self, is_async):
        prompts = [
            {"multi_modal_data": {"image": "<img>"}},
            {"prompt_token_ids": [1, 2]},
            {"prompt_embeds": [[0.1, 0.2]]},
        ]
        r = self._run(prompts, REQ_KW, is_async)
        assert r._eng_prompts[0]["media_io_kwargs"] == REQ_KW
        assert "media_io_kwargs" not in r._eng_prompts[1]
        assert "media_io_kwargs" not in r._eng_prompts[2]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_media_io_kwargs_not_forwarded_to_extras(self, is_async):
        r = self._run([{"multi_modal_data": {"image": "<img>"}}], REQ_KW, is_async)
        assert "media_io_kwargs" not in r._applied_extras[-1]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_prompt_extras_is_not_mutated(self, is_async):
        renderer = _StubRenderer()
        prompt: TokensPrompt = {
            "prompt_token_ids": [1],
            "multi_modal_data": {"image": "<img>"},
        }
        prompt_extras = {"media_io_kwargs": REQ_KW, "cache_salt": "request-1"}

        if is_async:
            asyncio.run(
                renderer.render_cmpl_async(
                    [prompt],
                    object(),
                    prompt_extras=prompt_extras,
                )
            )
        else:
            renderer.render_cmpl(
                [prompt],
                object(),
                prompt_extras=prompt_extras,
            )

        assert prompt_extras == {
            "media_io_kwargs": REQ_KW,
            "cache_salt": "request-1",
        }
        assert renderer._applied_extras[-1] == {"cache_salt": "request-1"}

    def test_empty_prompts_raises(self):
        with pytest.raises(VLLMValidationError):
            self._run([], REQ_KW, is_async=False)
        with pytest.raises(VLLMValidationError):
            self._run([], REQ_KW, is_async=True)

    def test_sync_async_parity(self):
        prompt = {
            "multi_modal_data": {"image": "<img>"},
            "media_io_kwargs": DEFAULT_KW,
        }
        sync_r = self._run([prompt], REQ_KW, is_async=False)
        async_r = self._run([{**prompt}], REQ_KW, is_async=True)
        assert (
            sync_r._eng_prompts[0]["media_io_kwargs"]
            == async_r._eng_prompts[0]["media_io_kwargs"]
        )


# =========================================================================
# Prompt dict isolation
# =========================================================================
class TestPromptDictIsolation:
    """`render_prompt` must copy caller-owned prompt dicts.

    The downstream merge / extras stages write keys such as
    `media_io_kwargs` into the prompt. Returning the caller's object
    unchanged would leak those writes back, so reusing the same dict for a
    second request would pick up the first request's keys.
    """

    @staticmethod
    def _run(prompts, media_io_kwargs=_UNSET, is_async=False):
        renderer = _StubRenderer()
        extras = _make_extras(media_io_kwargs)
        tok_params = object()
        if is_async:
            asyncio.run(
                renderer.render_cmpl_async(prompts, tok_params, prompt_extras=extras)
            )
        else:
            renderer.render_cmpl(prompts, tok_params, prompt_extras=extras)
        return renderer

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_reused_prompt_dict_not_polluted(self, is_async):
        prompt = {"multi_modal_data": {"image": "<img>"}}

        r1 = self._run([prompt], REQ_KW, is_async)
        # The caller's dict must be left alone.
        assert "media_io_kwargs" not in prompt
        assert r1._eng_prompts[-1]["media_io_kwargs"] == REQ_KW

        # A second request on the same dict, without request-level kwargs,
        # must not inherit the first request's media_io_kwargs.
        r2 = self._run([prompt], _UNSET, is_async)
        assert "media_io_kwargs" not in prompt
        assert "media_io_kwargs" not in r2._eng_prompts[-1]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_reused_enc_dec_prompt_dict_not_polluted(self, is_async):
        enc = {"multi_modal_data": {"image": "<img>"}}
        dec = {"multi_modal_uuids": {"image": ["uuid-0"]}}
        prompt = {"encoder_prompt": enc, "decoder_prompt": dec}

        r1 = self._run([prompt], REQ_KW, is_async)
        # Nested prompt dicts must not be mutated either.
        assert "media_io_kwargs" not in prompt
        assert "media_io_kwargs" not in enc
        assert "media_io_kwargs" not in dec
        assert r1._eng_prompts[-1]["encoder_prompt"]["media_io_kwargs"] == REQ_KW
        assert r1._eng_prompts[-1]["decoder_prompt"]["media_io_kwargs"] == REQ_KW

        r2 = self._run([prompt], _UNSET, is_async)
        assert "media_io_kwargs" not in r2._eng_prompts[-1]["encoder_prompt"]
        assert "media_io_kwargs" not in r2._eng_prompts[-1]["decoder_prompt"]


# =========================================================================
# Chat path
# =========================================================================
class TestChatPath:
    """`render_chat` / `render_chat_async` contract.

    Request-level `media_io_kwargs` are merged into `ChatParams` before
    `render_messages` runs; the popped key must not reach
    `_apply_prompt_extras`.  Priority is request > chat-params default.
    """

    @staticmethod
    def _run(media_io_kwargs=_UNSET, chat_media_io_kwargs=None, is_async=False):
        renderer = _StubRenderer()
        chat_params = _make_chat_params(chat_media_io_kwargs)
        extras = _make_extras(media_io_kwargs)
        conversations = [[{"role": "user", "content": "hi"}]]
        tok_params = object()
        if is_async:
            asyncio.run(
                renderer.render_chat_async(
                    conversations,
                    chat_params,
                    tok_params,
                    prompt_extras=extras,
                )
            )
        else:
            renderer.render_chat(
                conversations, chat_params, tok_params, prompt_extras=extras
            )
        return renderer

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_request_kwargs_merged_into_chat_params(self, is_async):
        r = self._run(REQ_KW, is_async=is_async)
        assert r._chat_params_seen[-1].media_io_kwargs == REQ_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_request_overrides_chat_defaults(self, monkeypatch, is_async):
        calls = _install_merge_spy(monkeypatch)
        r = self._run(REQ_KW, chat_media_io_kwargs=DEFAULT_KW, is_async=is_async)

        assert r._chat_params_seen[-1].media_io_kwargs == MERGED_KW
        assert calls[-1] == (DEFAULT_KW, REQ_KW)

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_unset_request_keeps_chat_defaults(self, is_async):
        r = self._run(_UNSET, chat_media_io_kwargs=DEFAULT_KW, is_async=is_async)
        assert r._chat_params_seen[-1].media_io_kwargs == DEFAULT_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_empty_dict_is_noop(self, is_async):
        r = self._run({}, chat_media_io_kwargs=DEFAULT_KW, is_async=is_async)
        assert r._chat_params_seen[-1].media_io_kwargs == DEFAULT_KW

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_media_io_kwargs_not_forwarded_to_extras(self, is_async):
        r = self._run(REQ_KW, is_async=is_async)
        assert "media_io_kwargs" not in r._applied_extras[-1]

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_prompt_extras_is_not_mutated(self, is_async):
        renderer = _StubRenderer()
        chat_params = _make_chat_params()
        prompt_extras = {"media_io_kwargs": REQ_KW, "cache_salt": "request-1"}
        conversations = [[{"role": "user", "content": "hi"}]]

        if is_async:
            asyncio.run(
                renderer.render_chat_async(
                    conversations,
                    chat_params,
                    object(),
                    prompt_extras=prompt_extras,
                )
            )
        else:
            renderer.render_chat(
                conversations,
                chat_params,
                object(),
                prompt_extras=prompt_extras,
            )

        assert prompt_extras == {
            "media_io_kwargs": REQ_KW,
            "cache_salt": "request-1",
        }
        assert renderer._applied_extras[-1] == {"cache_salt": "request-1"}

    def test_sync_async_parity(self):
        sync_r = self._run(REQ_KW, chat_media_io_kwargs=DEFAULT_KW, is_async=False)
        async_r = self._run(REQ_KW, chat_media_io_kwargs=DEFAULT_KW, is_async=True)
        assert (
            sync_r._chat_params_seen[-1].media_io_kwargs
            == async_r._chat_params_seen[-1].media_io_kwargs
        )

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_resolved_kwargs_used_as_base(self, monkeypatch, is_async):
        """`chat_params.media_io_kwargs` is already server+request merged.

        It must be passed as the *base* (defaults) side of
        `merge_media_io_kwargs`, with prompt-level kwargs as overrides.
        Passing it as the request side would make `VideoMediaIO.merge_kwargs`
        strip server-only keys (`device`, `max_width`, `backend`) and log a
        bogus "Stripping request-level ..." warning; the prompt would then
        carry only the request-level keys, diverging from the engine's view
        and corrupting the mm cache hash.
        """
        calls = _install_merge_spy(monkeypatch)
        resolved = {
            "video": {
                "device": "cuda",
                "max_width": 1280,
                "backend": "pynvvideocodec",
                "num_frames": 8,
            }
        }
        prompt = {"multi_modal_data": {"video": "<video>"}}
        renderer = _StubRenderer(rendered_prompt=prompt)
        chat_params = _make_chat_params(resolved)

        if is_async:
            asyncio.run(
                renderer.render_chat_async(
                    [[{"role": "user", "content": "hi"}]], chat_params, object()
                )
            )
        else:
            renderer.render_chat(
                [[{"role": "user", "content": "hi"}]], chat_params, object()
            )

        # Base = resolved chat params, overrides = prompt-level (absent here).
        assert calls[-1] == (resolved, None)

        out = renderer._eng_prompts[-1]["media_io_kwargs"]["video"]
        # Server-only keys must survive; the request-side key stays.
        assert out["device"] == "cuda"
        assert out["max_width"] == 1280
        assert out["backend"] == "pynvvideocodec"
        assert out["num_frames"] == 8

    @pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
    def test_prompt_level_kwargs_are_overrides(self, monkeypatch, is_async):
        """Prompt-level `media_io_kwargs` override the resolved chat config."""
        calls = _install_merge_spy(monkeypatch)
        resolved = {"video": {"device": "cuda", "num_frames": 8}}
        prompt_kw = {"video": {"num_frames": 16}}
        prompt = {
            "multi_modal_data": {"video": "<video>"},
            "media_io_kwargs": prompt_kw,
        }
        renderer = _StubRenderer(rendered_prompt=prompt)
        chat_params = _make_chat_params(resolved)

        if is_async:
            asyncio.run(
                renderer.render_chat_async(
                    [[{"role": "user", "content": "hi"}]], chat_params, object()
                )
            )
        else:
            renderer.render_chat(
                [[{"role": "user", "content": "hi"}]], chat_params, object()
            )

        # (defaults=resolved chat kwargs, overrides=prompt-level)
        assert calls[-1] == (resolved, prompt_kw)

        out = renderer._eng_prompts[-1]["media_io_kwargs"]["video"]
        assert out == {"device": "cuda", "num_frames": 16}
        # Prompt-level dict itself must not be mutated.
        assert prompt_kw == {"video": {"num_frames": 16}}


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_chat_media_io_kwargs_scope_uuid_cache_key(is_async):
    """Same video UUID with different frame counts must not share a cache key."""
    video_data_url = encode_video_url(VideoAsset("baby_reading").np_ndarrays)
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=4096,
        mm_processor_cache_gb=4.0,
    )
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": video_data_url},
                    "uuid": "shared-video-uuid",
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]
    chat_params_8 = ChatParams(media_io_kwargs={"video": {"num_frames": 8}})
    chat_params_16 = ChatParams(media_io_kwargs={"video": {"num_frames": 16}})

    if is_async:

        async def get_hash(chat_params):
            _, inputs = await renderer.render_chat_async([[*messages]], chat_params)
            return inputs[0]["mm_hashes"]["video"][0]

        hash_8 = asyncio.run(get_hash(chat_params_8))
        hash_16 = asyncio.run(get_hash(chat_params_16))
    else:
        _, inputs_8 = renderer.render_chat([[*messages]], chat_params_8)
        _, inputs_16 = renderer.render_chat([[*messages]], chat_params_16)
        hash_8 = inputs_8[0]["mm_hashes"]["video"][0]
        hash_16 = inputs_16[0]["mm_hashes"]["video"][0]

    assert hash_8 != hash_16
