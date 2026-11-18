# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.pooling.protocol import (
    PoolingChatRequest,
    PoolingCompletionRequest,
)
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor
from vllm.entrypoints.pooling.types import OfflineEncodeInputsContext
from vllm.entrypoints.pooling.utils import merge_chat_template_kwargs
from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams, TokenizeParams


@pytest.fixture
def processor() -> PoolingIOProcessor:
    return object.__new__(PoolingIOProcessor)


def test_rejects_untrusted_request_chat_template(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._validate_chat_template("template", None, False)

    assert str(exc_info.value) == (
        "Chat template is passed with request, but "
        "--trust-request-chat-template is not set. "
        "Refused request with untrusted chat template."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_pooling_params(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._params_to_seq([PoolingParams()], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and params (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_lora_requests(processor: PoolingIOProcessor):
    with pytest.raises(VLLMValidationError) as exc_info:
        processor._lora_request_to_seq([None], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and lora_request (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_conflicting_pooling_task(processor: PoolingIOProcessor):
    processor.model_config = SimpleNamespace(is_encoder_decoder=False)
    processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=None)
    )
    ctx = OfflineEncodeInputsContext(
        pooling_task="embed",
        tokenization_kwargs=None,
        lora_request=None,
        priorities=None,
        prompts=[[1]],
        pooling_params=PoolingParams(task="classify"),
    )

    with pytest.raises(VLLMValidationError) as exc_info:
        processor.get_request_factory_offline(ctx)

    assert str(exc_info.value) == (
        "You cannot overwrite param.task='classify' with pooling_task='embed'!"
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_qwen3_reranker_warns_without_chat_template(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.get_model_cls", lambda *_: MagicMock()
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.interfaces.supports_score_template",
        lambda *_: False,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.is_mistral_tokenizer",
        lambda *_: False,
    )

    mock_logger = MagicMock()
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.logger",
        mock_logger,
    )

    model_config = MagicMock()
    model_config.architecture = "Qwen3ForSequenceClassification"
    model_config.model = "Qwen/Qwen3-Reranker-0.6B"
    model_config.hf_config.is_original_qwen3_reranker = True
    model_config.use_sep_token = False
    model_config.is_multimodal_model = False
    model_config.max_model_len = 8192

    vllm_config = MagicMock(model_config=model_config)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    chat_template_config = MagicMock(chat_template=None)

    CrossEncoderIOProcessor(
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args
    assert "Qwen/Qwen3-Reranker-0.6B" in warning_call[0][1]
    assert "qwen3_reranker.jinja" in warning_call[0][2]


def test_qwen3_vl_reranker_warns_with_vl_template(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.get_model_cls", lambda *_: MagicMock()
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.interfaces.supports_score_template",
        lambda *_: False,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.is_mistral_tokenizer",
        lambda *_: False,
    )

    mock_logger = MagicMock()
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.logger",
        mock_logger,
    )

    model_config = MagicMock()
    model_config.architecture = "Qwen3VLForSequenceClassification"
    model_config.model = "Qwen/Qwen3-VL-Reranker-2B"
    model_config.hf_config.is_original_qwen3_reranker = True
    model_config.use_sep_token = False
    model_config.is_multimodal_model = True
    model_config.max_model_len = 8192

    vllm_config = MagicMock(model_config=model_config)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    chat_template_config = MagicMock(chat_template=None)

    CrossEncoderIOProcessor(
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args
    assert "Qwen/Qwen3-VL-Reranker-2B" in warning_call[0][1]
    assert "qwen3_vl_reranker.jinja" in warning_call[0][2]


def test_qwen3_reranker_no_warning_when_template_provided(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.get_model_cls", lambda *_: MagicMock()
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.interfaces.supports_score_template",
        lambda *_: False,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.is_mistral_tokenizer",
        lambda *_: False,
    )

    mock_logger = MagicMock()
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.scoring.io_processor.logger",
        mock_logger,
    )

    model_config = MagicMock()
    model_config.architecture = "Qwen3ForSequenceClassification"
    model_config.model = "Qwen/Qwen3-Reranker-0.6B"
    model_config.hf_config.is_original_qwen3_reranker = True
    model_config.use_sep_token = False
    model_config.is_multimodal_model = False
    model_config.max_model_len = 8192

    vllm_config = MagicMock(model_config=model_config)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    chat_template_config = MagicMock(chat_template="custom template")

    CrossEncoderIOProcessor(
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    mock_logger.warning.assert_not_called()


def _make_score_llm(monkeypatch) -> PoolingOfflineMixin:
    """A scoring LLM with no model behind it.

    Input validation runs for real; only the rendering, engine and
    post-processing boundaries are stubbed out. The pooling task is filled in
    before any of them, so the mutation is observable without inference.
    """
    proc = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    proc.is_multimodal_model = False
    proc.architecture = "CrossEncoder"
    monkeypatch.setattr(proc, "get_request_factory_offline", lambda ctx: (None, 0))
    monkeypatch.setattr(proc, "post_process_offline", lambda ctx: [])

    llm = PoolingOfflineMixin.__new__(PoolingOfflineMixin)
    llm.runner_type = "pooling"
    llm.pooling_task = "classify"  # SCORE_TYPE_MAP -> "cross-encoder"
    llm.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_labels=1))
    llm.pooling_io_processors = {"cross-encoder": proc}
    monkeypatch.setattr(llm, "_run_tiling_engine", lambda *args, **kwargs: [])
    return llm


def test_score_leaves_caller_pooling_params_untouched(monkeypatch):
    llm = _make_score_llm(monkeypatch)

    params = PoolingParams()
    llm.score("query", "doc", pooling_params=params)

    assert params.task is None, (
        "LLM.score() wrote the model's own pooling task into the caller's "
        "PoolingParams; reusing that object with another pooling model now "
        "fails with 'You cannot overwrite ...' before any inference runs."
    )


def _make_chat_request(**overrides):
    """Build a chat request without running the full engine wiring."""
    defaults = dict(
        messages=[{"role": "user", "content": "hi"}],
        chat_template=None,
        chat_template_kwargs=None,
        chat_template_content_format="auto",
        add_generation_prompt=None,
        continue_final_message=None,
        add_special_tokens=None,
        media_io_kwargs=None,
    )
    defaults.update(overrides)
    return PoolingChatRequest(**defaults)


def _make_completion_request(**overrides):
    defaults = dict(
        input="hi",
        add_special_tokens=None,
    )
    defaults.update(overrides)
    return PoolingCompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# merge_chat_template_kwargs: tri-state top-level flags
# ---------------------------------------------------------------------------


class TestMergeChatTemplateKwargs:
    """Top-level typed flags are tri-state: True/False override, None defers."""

    @pytest.mark.parametrize(
        "add_generation_prompt, continue_final_message, expected",
        [
            # None on both sides -> nothing injected.
            (None, None, {}),
            # Explicit True/False are injected as-is.
            (True, None, {"add_generation_prompt": True}),
            (False, None, {"add_generation_prompt": False}),
            (None, True, {"continue_final_message": True}),
            (None, False, {"continue_final_message": False}),
            (
                True,
                False,
                {"add_generation_prompt": True, "continue_final_message": False},
            ),
            (
                False,
                True,
                {"add_generation_prompt": False, "continue_final_message": True},
            ),
        ],
    )
    def test_none_defers_and_bool_overrides(
        self, add_generation_prompt, continue_final_message, expected
    ):
        merged = merge_chat_template_kwargs(
            None,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        assert merged == expected

    def test_none_keeps_existing_kwargs_key(self):
        merged = merge_chat_template_kwargs(
            {"add_generation_prompt": True, "foo": "bar"},
            add_generation_prompt=None,
            continue_final_message=None,
        )
        assert merged == {"add_generation_prompt": True, "foo": "bar"}

    def test_explicit_flag_overrides_existing_kwargs_key(self):
        merged = merge_chat_template_kwargs(
            {"add_generation_prompt": True},
            add_generation_prompt=False,
            continue_final_message=None,
        )
        assert merged == {"add_generation_prompt": False}

    def test_does_not_mutate_input_kwargs(self):
        original = {"add_generation_prompt": True}
        merge_chat_template_kwargs(
            original,
            add_generation_prompt=False,
            continue_final_message=None,
        )
        assert original == {"add_generation_prompt": True}

    @pytest.mark.parametrize(
        "kwargs, add_generation_prompt, continue_final_message",
        [
            # Both top-level flags True.
            (None, True, True),
            # Both from chat_template_kwargs.
            (
                {"add_generation_prompt": True, "continue_final_message": True},
                None,
                None,
            ),
            # Mixed: one from kwargs, one from top-level.
            ({"continue_final_message": True}, True, None),
            ({"add_generation_prompt": True}, None, True),
        ],
    )
    def test_rejects_conflicting_flags(
        self, kwargs, add_generation_prompt, continue_final_message
    ):
        with pytest.raises(VLLMValidationError) as exc_info:
            merge_chat_template_kwargs(
                kwargs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
        assert "Cannot set both `continue_final_message` and " in str(exc_info.value)
        assert "`add_generation_prompt` to True." in str(exc_info.value)

    def test_explicit_false_disables_kwargs_true(self):
        # Top-level False wins over kwargs True, so no conflict is raised.
        merged = merge_chat_template_kwargs(
            {"add_generation_prompt": True},
            add_generation_prompt=False,
            continue_final_message=None,
        )
        assert merged == {"add_generation_prompt": False}


# ---------------------------------------------------------------------------
# ChatRequestOptionsMixin.build_chat_params: chat_template_content_format
# ---------------------------------------------------------------------------


class TestBuildChatParamsContentFormat:
    """`auto`/None defer to the engine default; explicit string/openai win."""

    @pytest.mark.parametrize(
        "request_format, engine_default, expected",
        [
            # auto + auto -> auto is passed through for the renderer to resolve.
            ("auto", "auto", "auto"),
            # auto + concrete engine default -> engine default wins.
            ("auto", "string", "string"),
            ("auto", "openai", "openai"),
            # Explicit request value wins.
            ("string", "auto", "string"),
            ("openai", "auto", "openai"),
            ("string", "openai", "string"),
            ("openai", "string", "openai"),
        ],
    )
    def test_content_format_resolution(self, request_format, engine_default, expected):

        request = _make_chat_request(
            chat_template_content_format=request_format,
        )
        params = request.build_chat_params(
            default_template=None,
            default_template_content_format=engine_default,
        )
        assert isinstance(params, ChatParams)
        assert params.chat_template_content_format == expected

    def test_auto_passed_through_when_both_auto(self):
        request = _make_chat_request(chat_template_content_format="auto")
        params = request.build_chat_params(None, "auto")
        # `auto` must reach the renderer, which resolves it via
        # resolve_chat_template_content_format.
        assert params.chat_template_content_format == "auto"

    def test_chat_template_falls_back_to_default(self):
        request = _make_chat_request(chat_template=None)
        params = request.build_chat_params("default.jinja", "auto")
        assert params.chat_template == "default.jinja"

    def test_request_chat_template_overrides_default(self):
        request = _make_chat_request(chat_template="request.jinja")
        params = request.build_chat_params("default.jinja", "auto")
        assert params.chat_template == "request.jinja"


# ---------------------------------------------------------------------------
# ChatRequestOptionsMixin: tri-state add_special_tokens / flags
# ---------------------------------------------------------------------------


class TestChatRequestTriStateFields:
    """`add_generation_prompt`, `continue_final_message`, `add_special_tokens`
    accept True/False/None, where None means 'use the default'."""

    @pytest.mark.parametrize("value", [True, False, None])
    def test_add_special_tokens_accepts_tri_state(self, value):
        request = _make_chat_request(add_special_tokens=value)
        assert request.add_special_tokens is value

    @pytest.mark.parametrize("value", [True, False, None])
    def test_add_generation_prompt_accepts_tri_state(self, value):
        request = _make_chat_request(add_generation_prompt=value)
        assert request.add_generation_prompt is value

    @pytest.mark.parametrize("value", [True, False, None])
    def test_continue_final_message_accepts_tri_state(self, value):
        request = _make_chat_request(continue_final_message=value)
        assert request.continue_final_message is value

    def test_defaults_are_none(self):
        request = _make_chat_request()
        assert request.add_special_tokens is None
        assert request.add_generation_prompt is None
        assert request.continue_final_message is None
        assert request.chat_template_content_format == "auto"

    def test_rejects_conflicting_generation_flags_at_model_validation(self):
        with pytest.raises(VLLMValidationError) as exc_info:
            _make_chat_request(
                add_generation_prompt=True,
                continue_final_message=True,
            )
        assert "Cannot set both `continue_final_message` and " in str(exc_info.value)


# ---------------------------------------------------------------------------
# OfflineEncodeInputsContext.build_tok_params: tri-state add_special_tokens
# ---------------------------------------------------------------------------


class TestOfflineBuildTokParams:
    """Offline encode inputs currently only take the completion path.

    `None` keeps the renderer/model default as-is; explicit True/False
    overrides; user `tokenization_kwargs` still win over the flag.
    """

    def _ctx(self, add_special_tokens=None, tokenization_kwargs=None):
        return OfflineEncodeInputsContext(
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
            lora_request=None,
            priorities=None,
            prompts=[[1]],
            pooling_params=PoolingParams(),
            add_special_tokens=add_special_tokens,
        )

    def test_none_keeps_default_instance_untouched(self):
        default = TokenizeParams(max_total_tokens=8192, add_special_tokens=False)
        ctx = self._ctx(add_special_tokens=None)
        tok_params = ctx.build_tok_params(default)
        # None must not call with_kwargs: the exact default instance is kept,
        # preserving multimodal models that default to add_special_tokens=False.
        assert tok_params is default
        assert tok_params.add_special_tokens is False

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_bool_overrides_default(self, value):
        default = TokenizeParams(max_total_tokens=8192, add_special_tokens=not value)
        ctx = self._ctx(add_special_tokens=value)
        tok_params = ctx.build_tok_params(default)
        assert tok_params.add_special_tokens is value

    def test_tokenization_kwargs_win_over_explicit_flag(self):
        default = TokenizeParams(max_total_tokens=8192, add_special_tokens=True)
        ctx = self._ctx(
            add_special_tokens=False,
            tokenization_kwargs={"add_special_tokens": True},
        )
        tok_params = ctx.build_tok_params(default)
        assert tok_params.add_special_tokens is True

    def test_empty_kwargs_returns_default(self):
        default = TokenizeParams(max_total_tokens=8192, add_special_tokens=False)
        ctx = self._ctx(add_special_tokens=None, tokenization_kwargs=None)
        assert ctx.build_tok_params(default) is default


# ---------------------------------------------------------------------------
# Online pooling: per-path default_add_special_tokens
# ---------------------------------------------------------------------------


class TestOnlineDefaultAddSpecialTokens:
    """Online chat path defaults to False; completion path keeps the model
    default. Both are passed explicitly to `build_tok_params`."""

    def test_chat_path_uses_false_default(self):
        request = _make_chat_request(add_special_tokens=None)
        # The mixin resolves None via the caller-provided default.
        resolved = request._resolve_add_special_tokens(False)
        assert resolved is False

    def test_completion_path_uses_model_default(self):
        request = _make_completion_request(add_special_tokens=None)
        # completion: None -> renderer.default_cmpl_tok_params.add_special_tokens
        resolved = request._resolve_add_special_tokens(False)  # model default False
        assert resolved is False

    @pytest.mark.parametrize("explicit", [True, False])
    def test_explicit_value_overrides_per_path_default(self, explicit):
        request = _make_chat_request(add_special_tokens=explicit)
        # Explicit value wins regardless of the per-path default.
        assert request._resolve_add_special_tokens(False) is explicit
        assert request._resolve_add_special_tokens(True) is explicit

    def test_none_with_none_default_falls_back_to_true(self):
        request = _make_chat_request(add_special_tokens=None)
        # Last-resort fallback when the caller also passes None.
        assert request._resolve_add_special_tokens(None) is True
