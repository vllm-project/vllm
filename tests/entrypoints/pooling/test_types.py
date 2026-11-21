# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestOptionsMixin,
    EmbeddingTokenizeParamsMixin,
    FixedMaxLenTokenizeParamsMixin,
    PoolingBasicRequestMixin,
)
from vllm.entrypoints.pooling.types import OfflineInputsContext
from vllm.exceptions import VLLMValidationError


@dataclass
class _DummyTokenizeParams:
    add_special_tokens: bool | None = True
    max_total_tokens: int | None = None
    max_output_tokens: int | None = None
    truncate_prompt_tokens: int | None = None
    truncation_side: str | None = None
    do_lower_case: bool = False
    max_total_tokens_param: str = "max_model_len"
    max_output_tokens_param: str | None = None
    kwargs: dict[str, Any] | None = None

    def with_kwargs(self, **kw: Any) -> _DummyTokenizeParams:
        merged = dict(self.kwargs or {})
        merged.update(kw)
        return replace(self, kwargs=merged)


class _OnlineFixedLenRequest(PoolingBasicRequestMixin, FixedMaxLenTokenizeParamsMixin):
    pass


class _OnlineEmbeddingRequest(PoolingBasicRequestMixin, EmbeddingTokenizeParamsMixin):
    padding: Literal["max_length", "do_not_pad"] | None = None


def _make_tokenize_params(**overrides: Any) -> _DummyTokenizeParams:
    return _DummyTokenizeParams(**overrides)


def _make_pooling_params(extra_kwargs: dict[str, Any] | None = None) -> MagicMock:
    pp = MagicMock(name="PoolingParams")
    pp.extra_kwargs = extra_kwargs
    return pp


def _make_model_config(
    *,
    max_model_len: int = 512,
    encoder_config: dict[str, Any] | None = None,
    pooler_config: Any = None,
) -> MagicMock:
    mc = MagicMock(name="ModelConfig")
    mc.max_model_len = max_model_len
    mc.encoder_config = encoder_config
    mc.pooler_config = pooler_config
    return mc


def _make_offline_context(**overrides: Any) -> OfflineInputsContext:
    kwargs: dict[str, Any] = {"pooling_task": "embed"}
    kwargs.update(overrides)
    return OfflineInputsContext(**kwargs)


def _make_online_request(**overrides: Any) -> ChatRequestOptionsMixin:
    return ChatRequestOptionsMixin(**overrides)


def _call_offline_build_chat_params(
    ctx: OfflineInputsContext,
    *,
    params: MagicMock | None = None,
    default_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    default_format: str = "string",
) -> dict[str, Any]:
    if params is None:
        params = _make_pooling_params()

    with patch("vllm.entrypoints.pooling.types.ChatParams") as chat_params_cls:
        chat_params_cls.return_value = MagicMock(name="ChatParams")
        ctx.build_chat_params(
            params=params,
            default_template=default_template,
            chat_template_kwargs=chat_template_kwargs,
            default_template_content_format=default_format,
        )
        assert chat_params_cls.call_args is not None
        return dict(chat_params_cls.call_args.kwargs)


def _call_online_build_chat_params(
    req: ChatRequestOptionsMixin,
    *,
    default_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    default_format: str = "string",
) -> dict[str, Any]:
    with patch("vllm.entrypoints.pooling.base.protocol.ChatParams") as chat_params_cls:
        chat_params_cls.return_value = MagicMock(name="ChatParams")
        req.build_chat_params(
            default_template=default_template,
            chat_template_kwargs=chat_template_kwargs,
            default_template_content_format=default_format,
        )
        assert chat_params_cls.call_args is not None
        return dict(chat_params_cls.call_args.kwargs)


# =============================================================================
# build_chat_params tests (Online & Offline)
# Note:
# - Online requests do not have `params` (`PoolingParams`) or `"auto"` format fallback.
# - Offline additionally merges `params.extra_kwargs` (`chat_template_kwargs`)
#   above passed-in `chat_template_kwargs` (e.g. saved ST kwargs) and below
#   context `self.chat_template_kwargs`, and resolves `"auto"` format to default.
# =============================================================================


class TestOnlineBuildChatParams:
    """Online (`ChatRequestOptionsMixin.build_chat_params`) tests."""

    def test_defaults_apply_false_flags(self):
        req = _make_online_request()
        kwargs = _call_online_build_chat_params(
            req,
            default_template="default_tpl",
            default_format="string",
        )
        assert kwargs["chat_template"] == "default_tpl"
        assert kwargs["chat_template_content_format"] == "string"
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }
        assert kwargs["media_io_kwargs"] is None

    def test_passed_chat_template_kwargs_override_false_defaults(self):
        req = _make_online_request()
        kwargs = _call_online_build_chat_params(
            req,
            chat_template_kwargs={
                "add_generation_prompt": True,
                "from_saved": 1,
            },
        )
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": True,
            "continue_final_message": False,
            "from_saved": 1,
        }

    def test_request_chat_template_kwargs_override_passed_kwargs(self):
        req = _make_online_request(
            add_generation_prompt=None,
            continue_final_message=None,
            chat_template_kwargs={
                "add_generation_prompt": False,
                "custom_arg": "foo",
            },
        )
        kwargs = _call_online_build_chat_params(
            req,
            chat_template_kwargs={"add_generation_prompt": True, "from_saved": 1},
        )
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": False,
            "continue_final_message": False,
            "from_saved": 1,
            "custom_arg": "foo",
        }

    def test_explicit_flags_override_chat_template_kwargs_and_defaults(self):
        req = _make_online_request(
            add_generation_prompt=False,
            continue_final_message=True,
            chat_template_kwargs={
                "add_generation_prompt": True,
                "continue_final_message": False,
            },
        )
        kwargs = _call_online_build_chat_params(
            req,
            chat_template_kwargs={"add_generation_prompt": True},
        )
        assert kwargs["chat_template_kwargs"]["add_generation_prompt"] is False
        assert kwargs["chat_template_kwargs"]["continue_final_message"] is True

    def test_explicit_template_and_media_io_kwargs_are_forwarded(self):
        media_kwargs = {"image": {"resize": True}}
        req = _make_online_request(
            chat_template="req_tpl",
            media_io_kwargs=media_kwargs,
        )
        kwargs = _call_online_build_chat_params(req, default_template="default_tpl")
        assert kwargs["chat_template"] == "req_tpl"
        assert kwargs["media_io_kwargs"] == media_kwargs

    def test_conflict_in_explicit_fields_raises(self):
        with pytest.raises(VLLMValidationError):
            _make_online_request(
                add_generation_prompt=True,
                continue_final_message=True,
            )

    def test_conflict_in_merged_chat_template_kwargs_raises(self):
        req = _make_online_request(
            add_generation_prompt=None,
            continue_final_message=None,
            chat_template_kwargs={
                "add_generation_prompt": True,
                "continue_final_message": True,
            },
        )
        with pytest.raises(VLLMValidationError):
            _call_online_build_chat_params(req)

    def test_conflict_between_passed_kwargs_and_explicit_flag_raises(self):
        req = _make_online_request(continue_final_message=True)
        with pytest.raises(VLLMValidationError):
            _call_online_build_chat_params(
                req,
                chat_template_kwargs={"add_generation_prompt": True},
            )


class TestOfflineBuildChatParams:
    """Offline (`OfflineInputsContext.build_chat_params`) tests."""

    def test_defaults_apply_false_flags(self):
        ctx = _make_offline_context()
        kwargs = _call_offline_build_chat_params(
            ctx,
            default_template="default_tpl",
            default_format="string",
        )
        assert kwargs["chat_template"] == "default_tpl"
        assert kwargs["chat_template_content_format"] == "string"
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }
        assert kwargs["media_io_kwargs"] is None

    def test_passed_chat_template_kwargs_override_false_defaults(self):
        ctx = _make_offline_context()
        kwargs = _call_offline_build_chat_params(
            ctx,
            chat_template_kwargs={
                "add_generation_prompt": True,
                "from_saved": 1,
            },
        )
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": True,
            "continue_final_message": False,
            "from_saved": 1,
        }

    def test_priority_order(
        self,
    ):
        params = _make_pooling_params(
            extra_kwargs={
                "chat_template_kwargs": {
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                    "from_params": 1,
                }
            }
        )
        ctx = _make_offline_context(
            add_generation_prompt=None,
            continue_final_message=False,
            chat_template_kwargs={
                "add_generation_prompt": False,
                "from_self": 2,
            },
        )
        kwargs = _call_offline_build_chat_params(
            ctx,
            params=params,
            chat_template_kwargs={
                "add_generation_prompt": True,
                "from_passed": 0,
            },
        )
        assert kwargs["chat_template_kwargs"] == {
            "add_generation_prompt": False,
            "continue_final_message": False,
            "from_passed": 0,
            "from_params": 1,
            "from_self": 2,
        }

    def test_explicit_template_and_non_auto_format_win(self):
        ctx = _make_offline_context(
            chat_template="ctx_tpl",
            chat_template_content_format="openai",
        )
        kwargs = _call_offline_build_chat_params(
            ctx,
            default_template="default_tpl",
            default_format="string",
        )
        assert kwargs["chat_template"] == "ctx_tpl"
        assert kwargs["chat_template_content_format"] == "openai"

    def test_auto_format_falls_back_to_default_format(self):
        ctx = _make_offline_context(chat_template_content_format="auto")
        kwargs = _call_offline_build_chat_params(ctx, default_format="string")
        assert kwargs["chat_template_content_format"] == "string"

    def test_conflict_at_context_level_raises(self):
        ctx = _make_offline_context(
            add_generation_prompt=True,
            continue_final_message=True,
        )
        with pytest.raises(VLLMValidationError):
            _call_offline_build_chat_params(ctx)

    def test_conflict_in_merged_kwargs_raises(self):
        params = _make_pooling_params(
            extra_kwargs={"chat_template_kwargs": {"continue_final_message": True}}
        )
        ctx = _make_offline_context(
            add_generation_prompt=True,
            continue_final_message=None,
        )
        with pytest.raises(VLLMValidationError):
            _call_offline_build_chat_params(ctx, params=params)


# =============================================================================
# build_tok_params tests (Online & Offline)
# Note:
# - Online requests do not have `tokenization_kwargs`.
# - Offline additionally merges `params.extra_kwargs`
# (`chat_template_kwargs`) below `self.chat_template_kwargs`.
# =============================================================================


class TestOnlineBuildTokParams:
    """Online (`build_tok_params`) tests for fixed-len and embedding mixins."""

    def test_fixed_len_defaults_preserve_default_special_tokens(self):
        req = _OnlineFixedLenRequest()
        model_cfg = _make_model_config(max_model_len=1024)
        default_tp = _make_tokenize_params(add_special_tokens=True)

        with patch(
            "vllm.entrypoints.pooling.base.protocol.replace",
            side_effect=replace,
        ):
            result = req.build_tok_params(model_cfg, default_tp)

        assert result.max_total_tokens == 1024
        assert result.max_output_tokens == 0
        assert result.add_special_tokens is True
        assert result.do_lower_case is False
        assert getattr(result, "kwargs", None) is None

    def test_explicit_add_special_tokens_and_truncation_and_padding(self):
        req = _OnlineFixedLenRequest(
            add_special_tokens=False,
            truncate_prompt_tokens=128,
            truncation_side="left",
            padding="max_length",
        )
        model_cfg = _make_model_config(
            max_model_len=256,
            encoder_config={"do_lower_case": True},
        )
        default_tp = _make_tokenize_params(add_special_tokens=True)

        with patch(
            "vllm.entrypoints.pooling.base.protocol.replace",
            side_effect=replace,
        ):
            result = req.build_tok_params(model_cfg, default_tp)

        assert result.add_special_tokens is False
        assert result.truncate_prompt_tokens == 128
        assert result.truncation_side == "left"
        assert result.do_lower_case is True
        assert getattr(result, "kwargs", None) == {"padding": "max_length"}

    def test_embedding_chunked_processing_sets_max_embed_len_and_pad_tokens(self):
        pooler_cfg = MagicMock(
            enable_chunked_processing=True,
            max_embed_len=128,
        )
        model_cfg = _make_model_config(
            max_model_len=512,
            pooler_config=pooler_cfg,
        )
        req = _OnlineEmbeddingRequest(padding="max_length")
        default_tp = _make_tokenize_params()

        with patch(
            "vllm.entrypoints.pooling.base.protocol.replace",
            side_effect=replace,
        ):
            result = req.build_tok_params(model_cfg, default_tp)

        assert result.max_total_tokens == 128
        assert getattr(result, "kwargs", None) == {
            "padding": "max_length",
            "pad_prompt_tokens": 128,
        }

    def test_embedding_non_chunked_uses_max_embed_len_when_set(self):
        pooler_cfg = MagicMock(
            enable_chunked_processing=False,
            max_embed_len=200,
        )
        model_cfg = _make_model_config(
            max_model_len=512,
            pooler_config=pooler_cfg,
        )
        req = _OnlineEmbeddingRequest(padding="max_length")
        default_tp = _make_tokenize_params()

        with patch(
            "vllm.entrypoints.pooling.base.protocol.replace",
            side_effect=replace,
        ):
            result = req.build_tok_params(model_cfg, default_tp)

        assert result.max_total_tokens == 200
        assert getattr(result, "kwargs", None) == {"padding": "max_length"}
        assert "pad_prompt_tokens" not in (getattr(result, "kwargs", None) or {})


class TestOfflineBuildTokParams:
    """Offline (`OfflineInputsContext.build_tok_params`) tests."""

    def test_no_overrides_returns_default_unchanged(self):
        default_tp = _make_tokenize_params()
        result = _make_offline_context().build_tok_params(default_tp)
        assert result is default_tp
        assert getattr(result, "kwargs", None) is None

    def test_tokenization_kwargs_only(self):
        default_tp = _make_tokenize_params()
        result = _make_offline_context(
            tokenization_kwargs={"max_length": 32, "truncation": True}
        ).build_tok_params(default_tp)
        assert getattr(result, "kwargs", None) == {
            "max_length": 32,
            "truncation": True,
        }

    def test_add_special_tokens_none_keeps_tokenization_kwargs(self):
        default_tp = _make_tokenize_params()
        result = _make_offline_context(
            tokenization_kwargs={"max_length": 32},
            add_special_tokens=None,
        ).build_tok_params(default_tp)
        assert getattr(result, "kwargs", None) == {"max_length": 32}

    @pytest.mark.parametrize("flag", [True, False])
    def test_explicit_add_special_tokens_overrides_tokenization_kwargs(
        self, flag: bool
    ):
        default_tp = _make_tokenize_params()
        result = _make_offline_context(
            tokenization_kwargs={"add_special_tokens": not flag, "max_length": 16},
            add_special_tokens=flag,
        ).build_tok_params(default_tp)
        assert getattr(result, "kwargs", None) == {
            "add_special_tokens": flag,
            "max_length": 16,
        }
