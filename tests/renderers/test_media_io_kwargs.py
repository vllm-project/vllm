# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import dataclasses
import sys

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import ChatParams

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

    def __init__(self):
        # Deliberately no super().__init__: none of the tested paths touch
        # the resources, tokenizer, or cache created there.
        self._applied_extras: list[dict] = []
        self._chat_params_seen: list[ChatParams] = []
        self._eng_prompts: list = []

    # -- chat downstream stages --------------------------------------
    def render_messages(self, messages, params):
        self._chat_params_seen.append(params)
        return [], {"prompt_token_ids": [1]}

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

    def test_sync_async_parity(self):
        sync_r = self._run(REQ_KW, chat_media_io_kwargs=DEFAULT_KW, is_async=False)
        async_r = self._run(REQ_KW, chat_media_io_kwargs=DEFAULT_KW, is_async=True)
        assert (
            sync_r._chat_params_seen[-1].media_io_kwargs
            == async_r._chat_params_seen[-1].media_io_kwargs
        )
