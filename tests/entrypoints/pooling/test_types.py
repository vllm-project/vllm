# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.pooling.types import OfflineInputsContext
from vllm.exceptions import VLLMValidationError


def _make_tokenize_params() -> MagicMock:
    """A stand-in for ``TokenizeParams`` that records ``with_kwargs`` calls."""
    tp = MagicMock(name="TokenizeParams")
    tp.with_kwargs = MagicMock(
        name="TokenizeParams.with_kwargs",
        side_effect=lambda **kw: MagicMock(name="TokenizeParams.clone", _kw=kw),
    )
    return tp


def _make_pooling_params(extra_kwargs: dict[str, Any] | None = None) -> MagicMock:
    pp = MagicMock(name="PoolingParams")
    pp.extra_kwargs = extra_kwargs
    return pp


def _make_context(**overrides: Any) -> OfflineInputsContext:
    kwargs: dict[str, Any] = {"pooling_task": "embed"}
    kwargs.update(overrides)
    return OfflineInputsContext(**kwargs)


def _make_tokenizer() -> MagicMock:
    return MagicMock(name="Tokenizer")


def _make_model_config() -> MagicMock:
    return MagicMock(name="ModelConfig")


def _build_chat_params(
    ctx: OfflineInputsContext,
    *,
    params: MagicMock | None = None,
    default_template: str | None = None,
    default_format: str = "string",
) -> MagicMock:
    """Call ``ctx.build_chat_params`` with the real resolver."""
    if params is None:
        params = _make_pooling_params()

    with patch("vllm.entrypoints.pooling.types.ChatParams") as chat_params_cls:
        chat_params_cls.return_value = MagicMock(name="ChatParams")
        ctx.build_chat_params(
            params=params,
            default_template=default_template,
            default_template_content_format=default_format,
            tokenizer=_make_tokenizer(),
            model_config=_make_model_config(),
        )
        return chat_params_cls


class TestBuildTokParams:
    """Priority: explicit add_special_tokens > tokenization_kwargs > default."""

    def test_no_overrides_returns_default_unchanged(self):
        default = _make_tokenize_params()
        result = _make_context().build_tok_params(default)
        assert result is default
        default.with_kwargs.assert_not_called()

    def test_tokenization_kwargs_only(self):
        default = _make_tokenize_params()
        _make_context(tokenization_kwargs={"max_length": 7}).build_tok_params(default)
        default.with_kwargs.assert_called_once_with(max_length=7)

    def test_add_special_tokens_none_keeps_tokenization_kwargs(self):
        default = _make_tokenize_params()
        _make_context(
            tokenization_kwargs={"max_length": 7}, add_special_tokens=None
        ).build_tok_params(default)
        default.with_kwargs.assert_called_once_with(max_length=7)

    def test_add_special_tokens_true_overrides_tokenization_kwargs(self):
        default = _make_tokenize_params()
        _make_context(
            tokenization_kwargs={"add_special_tokens": False, "max_length": 7},
            add_special_tokens=True,
        ).build_tok_params(default)
        default.with_kwargs.assert_called_once_with(
            add_special_tokens=True, max_length=7
        )

    def test_add_special_tokens_false_is_still_applied(self):
        # Regression guard: the guard is ``is not None``, not truthiness.
        default = _make_tokenize_params()
        _make_context(
            tokenization_kwargs={"add_special_tokens": True},
            add_special_tokens=False,
        ).build_tok_params(default)
        default.with_kwargs.assert_called_once_with(add_special_tokens=False)

    def test_empty_kwargs_and_none_add_special_tokens_fast_path(self):
        default = _make_tokenize_params()
        result = _make_context(
            tokenization_kwargs={}, add_special_tokens=None
        ).build_tok_params(default)
        assert result is default
        default.with_kwargs.assert_not_called()


class TestBuildChatTemplateKwargs:
    """Priority: explicit tri-state > self kwargs > params.extra_kwargs."""

    def test_default_tristate_does_not_inject(self):
        # Both tri-state fields default to ``None`` (not ``False``), so
        # they are not materialised and do not override params.
        result = _make_context().build_chat_template_kwargs(_make_pooling_params())
        assert "add_generation_prompt" not in result
        assert "continue_final_message" not in result

    def test_explicit_none_does_not_inject(self):
        result = _make_context(
            add_generation_prompt=None, continue_final_message=None
        ).build_chat_template_kwargs(_make_pooling_params())
        assert "add_generation_prompt" not in result
        assert "continue_final_message" not in result

    def test_explicit_true_overrides_params(self):
        params = _make_pooling_params(
            extra_kwargs={"chat_template_kwargs": {"add_generation_prompt": False}}
        )
        result = _make_context(add_generation_prompt=True).build_chat_template_kwargs(
            params
        )
        assert result["add_generation_prompt"] is True

    def test_explicit_false_overrides_params(self):
        params = _make_pooling_params(
            extra_kwargs={"chat_template_kwargs": {"add_generation_prompt": True}}
        )
        result = _make_context(add_generation_prompt=False).build_chat_template_kwargs(
            params
        )
        assert result["add_generation_prompt"] is False

    def test_continue_final_message_alone_is_allowed(self):
        result = _make_context(continue_final_message=True).build_chat_template_kwargs(
            _make_pooling_params()
        )
        assert result["continue_final_message"] is True
        assert "add_generation_prompt" not in result

    def test_conflict_at_context_level_raises(self):
        with pytest.raises(VLLMValidationError):
            _make_context(
                add_generation_prompt=True, continue_final_message=True
            ).build_chat_template_kwargs(_make_pooling_params())

    def test_conflict_in_params_extra_kwargs_raises(self):
        params = _make_pooling_params(
            extra_kwargs={
                "chat_template_kwargs": {
                    "add_generation_prompt": True,
                    "continue_final_message": True,
                }
            }
        )
        with pytest.raises(VLLMValidationError):
            _make_context(
                add_generation_prompt=None, continue_final_message=None
            ).build_chat_template_kwargs(params)


class TestBuildChatParams:
    """``OfflineInputsContext.build_chat_params`` -> ``ChatParams`` assembly."""

    def test_defaults_are_wired_into_chat_params(self):
        ctx = _make_context()
        chat_params_cls = _build_chat_params(
            ctx,
            default_template="default_tpl",
            default_format="string",
        )
        kwargs = chat_params_cls.call_args.kwargs
        assert kwargs["chat_template"] == "default_tpl"
        assert kwargs["chat_template_content_format"] == "string"
        assert kwargs["chat_template_kwargs"] == {}
        assert kwargs["media_io_kwargs"] is None

    def test_explicit_context_template_wins(self):
        ctx = _make_context(chat_template="ctx_tpl")
        chat_params_cls = _build_chat_params(ctx, default_template="default_tpl")
        assert chat_params_cls.call_args.kwargs["chat_template"] == "ctx_tpl"
