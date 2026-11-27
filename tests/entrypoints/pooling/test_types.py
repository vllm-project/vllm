# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.pooling.base.protocol import (
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


def _make_online_request(**overrides: Any) -> _OnlineFixedLenRequest:
    return _OnlineFixedLenRequest(**overrides)


def _call_offline_build_prompt_extras(
    ctx: OfflineInputsContext,
    *,
    params: Any = None,
) -> dict[str, Any]:
    return ctx.build_prompt_extras(params)


def _call_online_build_prompt_extras(
    req: PoolingBasicRequestMixin,
) -> dict[str, Any]:
    return req.build_prompt_extras()


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


# =============================================================================
# Multimodal & Caching prompt_extras tests (Online & Offline)
# =============================================================================


class TestOnlineBuildPromptExtras:
    """Request media options are carried to the renderer."""

    def test_empty_request_returns_empty(self):
        req = _make_online_request()
        result = _call_online_build_prompt_extras(req)
        assert result == {}

    def test_explicit_fields_preserve_request_media_io_kwargs(self):
        req = _make_online_request(
            cache_salt="salt-abc-123",
            media_io_kwargs={"image": {"resize": [224, 224]}},
            mm_processor_kwargs={"do_rescale": False},
        )

        result = _call_online_build_prompt_extras(req)
        assert result == {
            "cache_salt": "salt-abc-123",
            "mm_processor_kwargs": {"do_rescale": False},
            "media_io_kwargs": {"image": {"resize": [224, 224]}},
        }


class TestOfflineBuildPromptExtras:
    """Offline (`OfflineInputsContext.build_prompt_extras`) precedence tests."""

    def test_empty_inputs_returns_empty_without_key_error(self):
        ctx = _make_offline_context()
        result = _call_offline_build_prompt_extras(ctx, params=None)

        assert result == {}

    def test_context_fields_override_pooling_params_extra_kwargs(self):
        params = _make_pooling_params(
            extra_kwargs={
                "cache_salt": "from-params",
                "mm_processor_kwargs": {"from_params": True},
                "media_io_kwargs": {"image": {"source": "params"}},
            }
        )
        ctx = _make_offline_context(
            cache_salt="from-context",
            mm_processor_kwargs={"from_context": True},
            media_io_kwargs={"image": {"source": "context"}},
        )

        result = _call_offline_build_prompt_extras(ctx, params=params)

        # Context `media_io_kwargs` takes precedence over `params.extra_kwargs`.
        assert result == {
            "cache_salt": "from-context",
            "mm_processor_kwargs": {"from_context": True},
            "media_io_kwargs": {"image": {"source": "context"}},
        }

    def test_none_context_fields_fall_back_to_extra_kwargs(self):
        params = _make_pooling_params(
            extra_kwargs={
                "cache_salt": "from-params",
                "mm_processor_kwargs": {"from_params": 1},
                "media_io_kwargs": {"image": {"from_params": "yes"}},
            }
        )
        ctx = _make_offline_context(
            cache_salt=None,
            mm_processor_kwargs=None,
            media_io_kwargs=None,
        )

        result = _call_offline_build_prompt_extras(ctx, params=params)

        assert result == {
            "cache_salt": "from-params",
            "mm_processor_kwargs": {"from_params": 1},
            "media_io_kwargs": {"image": {"from_params": "yes"}},
        }


# =============================================================================
# Request validation tests (cache_salt bounds & invalid inputs)
# =============================================================================


class TestOnlineRequestValidation:
    """Validation rules for `cache_salt` on online pooling requests."""

    @pytest.mark.parametrize("salt", ["a", "x" * 128])
    def test_valid_cache_salt_lengths_succeed(self, salt: str):
        req = _make_online_request(cache_salt=salt)
        assert req.cache_salt == salt

    @pytest.mark.parametrize("bad_salt", ["", "x" * 129])
    def test_invalid_cache_salt_length_raises_error(self, bad_salt: str):
        with pytest.raises(VLLMValidationError):
            _make_online_request(cache_salt=bad_salt)

    @pytest.mark.parametrize("bad_salt", ["a@b", "a/b", "a\\b", "a\x00b"])
    def test_forbidden_cache_salt_chars_raise_error(self, bad_salt: str):
        with pytest.raises(VLLMValidationError):
            _make_online_request(cache_salt=bad_salt)
