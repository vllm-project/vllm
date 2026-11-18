# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.embed.io_processor import TokenEmbedIOProcessor
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor
from vllm.entrypoints.pooling.types import (
    OfflineEncodeInputsContext,
    OfflineInputsContext,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def processor() -> PoolingIOProcessor:
    """An uninitialised `PoolingIOProcessor` for pure-method tests."""
    return object.__new__(PoolingIOProcessor)


def _make_offline_processor(monkeypatch: pytest.MonkeyPatch) -> PoolingIOProcessor:
    """A minimally wired `PoolingIOProcessor` for exercising the offline
    request factory without a real engine."""
    proc = PoolingIOProcessor.__new__(PoolingIOProcessor)
    proc.name = "embed"
    proc.model_config = SimpleNamespace(multimodal_config=None)
    proc.renderer = MagicMock()
    proc.renderer.default_cmpl_tok_params = TokenizeParams(max_total_tokens=None)
    proc.renderer.tokenizer = MagicMock()
    proc.chat_template = "engine_jinja"
    cast(Any, proc).chat_template_content_format = "string"
    proc.template_kwargs = None
    proc.tool_dicts = None

    # `is_mistral_tokenizer` would try to introspect the MagicMock tokenizer.
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.base.io_processor.is_mistral_tokenizer",
        lambda *_: False,
    )
    return proc


def _make_encode_ctx(
    *,
    pooling_task: str = "embed",
    inputs_seq=None,
    pooling_params=None,
    **kwargs,
) -> OfflineEncodeInputsContext:
    return OfflineEncodeInputsContext(
        pooling_task=pooling_task,
        tokenization_kwargs=None,
        lora_request=None,
        priorities=None,
        inputs_seq=inputs_seq if inputs_seq is not None else ["hello"],
        pooling_params=pooling_params,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_untrusted_request_chat_template(
        self, processor: PoolingIOProcessor
    ):
        with pytest.raises(VLLMValidationError) as exc_info:
            processor._validate_chat_template("template", None, False)

        assert str(exc_info.value) == (
            "Chat template is passed with request, but "
            "--trust-request-chat-template is not set. "
            "Refused request with untrusted chat template."
        )
        assert exc_info.value.parameter is None
        assert exc_info.value.value is None

    def test_rejects_mismatched_pooling_params(self, processor: PoolingIOProcessor):
        with pytest.raises(VLLMValidationError) as exc_info:
            processor._params_to_seq([PoolingParams()], num_requests=2)

        assert str(exc_info.value) == (
            "The lengths of prompts (2) and params (1) must be the same."
        )
        assert exc_info.value.parameter is None
        assert exc_info.value.value is None

    def test_rejects_mismatched_lora_requests(self, processor: PoolingIOProcessor):
        with pytest.raises(VLLMValidationError) as exc_info:
            processor._lora_request_to_seq([None], num_requests=2)

        assert str(exc_info.value) == (
            "The lengths of prompts (2) and lora_request (1) must be the same."
        )
        assert exc_info.value.parameter is None
        assert exc_info.value.value is None

    def test_rejects_conflicting_pooling_task(self, processor: PoolingIOProcessor):
        processor.model_config = SimpleNamespace(is_encoder_decoder=False)
        processor.renderer = SimpleNamespace(
            default_cmpl_tok_params=TokenizeParams(max_total_tokens=None)
        )
        ctx = OfflineEncodeInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            inputs_seq=[[1]],
            pooling_params=PoolingParams(task="classify"),
        )

        with pytest.raises(VLLMValidationError) as exc_info:
            processor.get_request_factory_offline(ctx)

        assert str(exc_info.value) == (
            "You cannot overwrite param.task='classify' with pooling_task='embed'!"
        )
        assert exc_info.value.parameter is None
        assert exc_info.value.value is None


# ---------------------------------------------------------------------------
# Qwen3 reranker without a template -> warn; with a template -> silent
# ---------------------------------------------------------------------------


class TestQwen3RerankerWarning:
    @staticmethod
    def _build_processor(
        monkeypatch: pytest.MonkeyPatch,
        *,
        architecture: str,
        model_name: str,
        is_multimodal: bool,
        chat_template: str | None,
    ) -> tuple[CrossEncoderIOProcessor, MagicMock]:
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
            "vllm.entrypoints.pooling.scoring.io_processor.logger", mock_logger
        )

        model_config = MagicMock()
        model_config.architecture = architecture
        model_config.model = model_name
        model_config.hf_config.is_original_qwen3_reranker = True
        model_config.use_sep_token = False
        model_config.is_multimodal_model = is_multimodal
        model_config.max_model_len = 8192

        vllm_config = MagicMock(model_config=model_config)
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        renderer = MagicMock()
        renderer.get_tokenizer.return_value = tokenizer
        chat_template_config = MagicMock(chat_template=chat_template)

        proc = CrossEncoderIOProcessor(
            vllm_config=vllm_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        return proc, mock_logger

    def test_qwen3_reranker_warns_without_chat_template(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _, mock_logger = self._build_processor(
            monkeypatch,
            architecture="Qwen3ForSequenceClassification",
            model_name="Qwen/Qwen3-Reranker-0.6B",
            is_multimodal=False,
            chat_template=None,
        )

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "Qwen/Qwen3-Reranker-0.6B" in warning_call[0][1]
        assert "qwen3_reranker.jinja" in warning_call[0][2]

    def test_qwen3_vl_reranker_warns_with_vl_template(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _, mock_logger = self._build_processor(
            monkeypatch,
            architecture="Qwen3VLForSequenceClassification",
            model_name="Qwen/Qwen3-VL-Reranker-2B",
            is_multimodal=True,
            chat_template=None,
        )

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "Qwen/Qwen3-VL-Reranker-2B" in warning_call[0][1]
        assert "qwen3_vl_reranker.jinja" in warning_call[0][2]

    def test_qwen3_reranker_no_warning_when_template_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _, mock_logger = self._build_processor(
            monkeypatch,
            architecture="Qwen3ForSequenceClassification",
            model_name="Qwen/Qwen3-Reranker-0.6B",
            is_multimodal=False,
            chat_template="custom template",
        )

        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# LLM.score() must not mutate the caller's PoolingParams
# ---------------------------------------------------------------------------


def _make_score_llm(monkeypatch: pytest.MonkeyPatch) -> PoolingOfflineMixin:
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
    llm.chat_template_config = SimpleNamespace(chat_template_content_format="auto")
    monkeypatch.setattr(llm, "_run_tiling_engine", lambda *args, **kwargs: [])
    return llm


class TestScorePoolingParamsIsolation:
    def test_score_leaves_caller_pooling_params_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        llm = _make_score_llm(monkeypatch)

        params = PoolingParams()
        llm.score("query", "doc", pooling_params=params)

        assert params.task is None, (
            "LLM.score() wrote the model's own pooling task into the caller's "
            "PoolingParams; reusing that object with another pooling model now "
            "fails with 'You cannot overwrite ...' before any inference runs."
        )


# ---------------------------------------------------------------------------
# `OfflineInputsContext.build_chat_params`
# ---------------------------------------------------------------------------


class TestBuildChatParams:
    def test_self_template_wins(self):
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template="user_jinja",
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template == "user_jinja"

    def test_self_content_format_wins(self):
        """The user-supplied content format must override the engine default."""
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template_content_format="openai",
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_content_format == "openai"

    def test_auto_falls_back_to_default(self):
        """`"auto"` is the "unspecified" sentinel: it must not swallow the
        engine default. Resolution happens later, in the renderer."""
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template_content_format="auto",
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_content_format == "string"

    def test_default_false_is_materialized(self):
        """Default `False` flags must still be written into`chat_template_kwargs`."""
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_kwargs["add_generation_prompt"] is False
        assert params.chat_template_kwargs["continue_final_message"] is False

    def test_explicit_kwargs_win_over_top_level_flag(self):
        """An explicit value in `chat_template_kwargs` wins over the top-level flag."""
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template_kwargs={"add_generation_prompt": False},
            add_generation_prompt=True,
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_kwargs["add_generation_prompt"] is False

    def test_top_level_true_fills_missing_kwarg(self):
        """A top-level `True` fills in the missing key; caller-supplied kwargs
        are preserved, and the companion flag is materialized as `False`."""
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template_kwargs={"existing": 1},
            add_generation_prompt=True,
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_kwargs == {
            "existing": 1,
            "add_generation_prompt": True,
            "continue_final_message": False,
        }

    def test_explicit_kwargs_win_over_default_false(self):
        """An explicit `True` in kwargs must survive the top-level default
        `False`.

        Guards `setdefault` precedence: the top-level default is a fallback,
        not an override.
        """
        ctx = OfflineInputsContext(
            pooling_task="embed",
            tokenization_kwargs=None,
            lora_request=None,
            priorities=None,
            chat_template_kwargs={"add_generation_prompt": True},
        )

        params = ctx.build_chat_params(
            default_template="engine_jinja",
            default_template_content_format="string",
        )

        assert params.chat_template_kwargs["add_generation_prompt"] is True


# ---------------------------------------------------------------------------
# `PoolingIOProcessor.get_request_factory_offline` must call `.with_defaults`
# ---------------------------------------------------------------------------


class TestOfflineFactoryDefaults:
    def test_with_defaults_is_applied(self, monkeypatch: pytest.MonkeyPatch):
        """Offline must apply the same `.with_defaults(...)` as online.

        Without this, Mistral tokenizers silently lose the `tokenize=True`
        flag, and multimodal models silently lose their engine-level
        `media_io_kwargs` on the offline path.
        """
        proc = _make_offline_processor(monkeypatch)

        ctx = _make_encode_ctx()
        fake_chat_params = MagicMock(name="ChatParams")
        monkeypatch.setattr(
            OfflineEncodeInputsContext,
            "build_chat_params",
            MagicMock(return_value=fake_chat_params),
        )

        proc.get_request_factory_offline(ctx)

        fake_chat_params.with_defaults.assert_called_once()
        call = fake_chat_params.with_defaults.call_args
        default_template_kwargs = call.args[0]

        # `merge_kwargs` drops `None` values, so `tools` (populated from
        # `self.tool_dicts`, which is `None` here) is absent while
        # `tokenize` survives because it is always a concrete bool.
        assert "tools" not in default_template_kwargs
        assert default_template_kwargs["tokenize"] is False
        assert call.kwargs["default_media_io_kwargs"] is None

    def test_with_defaults_passes_tools_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When `self.tool_dicts` is populated, it must reach the chat template.

        Counterpart to the "tools absent" assertion above: this guards that
        `self.tool_dicts` is actually forwarded (once it becomes non-`None`).
        """
        proc = _make_offline_processor(monkeypatch)
        proc.tool_dicts = [{"type": "function", "function": {"name": "f"}}]

        ctx = _make_encode_ctx()
        fake_chat_params = MagicMock(name="ChatParams")
        monkeypatch.setattr(
            OfflineEncodeInputsContext,
            "build_chat_params",
            MagicMock(return_value=fake_chat_params),
        )

        proc.get_request_factory_offline(ctx)

        call = fake_chat_params.with_defaults.call_args
        default_template_kwargs = call.args[0]
        assert default_template_kwargs["tools"] == [
            {"type": "function", "function": {"name": "f"}}
        ]

    def test_with_defaults_passes_media_io_kwargs_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When the model has engine-level `media_io_kwargs`, offline must
        forward them (mirroring the online path)."""
        proc = _make_offline_processor(monkeypatch)
        mm_config = SimpleNamespace(media_io_kwargs={"image": {"timeout": 5}})
        proc.model_config = SimpleNamespace(multimodal_config=mm_config)

        ctx = _make_encode_ctx()
        fake_chat_params = MagicMock(name="ChatParams")
        monkeypatch.setattr(
            OfflineEncodeInputsContext,
            "build_chat_params",
            MagicMock(return_value=fake_chat_params),
        )

        proc.get_request_factory_offline(ctx)

        call = fake_chat_params.with_defaults.call_args
        assert call.kwargs["default_media_io_kwargs"] == {"image": {"timeout": 5}}

    def test_offline_factory_accepts_request_chat_template(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Offline encode/score must accept per-call chat templates without
        requiring `trust_request_chat_template=True`; the serving-side trust
        gate only applies to the online factory."""
        proc = _make_offline_processor(monkeypatch)

        ctx = _make_encode_ctx(chat_template="user_jinja")
        build_calls: list[str] = []

        original_build = OfflineEncodeInputsContext.build_chat_params

        def _build(self_ctx, *args, **kwargs):
            build_calls.append("called")
            return original_build(self_ctx, *args, **kwargs)

        monkeypatch.setattr(OfflineEncodeInputsContext, "build_chat_params", _build)

        # Must not raise: offline callers are trusted library users.
        factory, num_requests = proc.get_request_factory_offline(ctx)
        assert num_requests == 1

        # The caller's template must be forwarded into ChatParams building.
        assert build_calls == ["called"], (
            "offline factory should build ChatParams from the caller's template"
        )


# ---------------------------------------------------------------------------
# Late-chunking + chat must be rejected up front
# ---------------------------------------------------------------------------


class TestLateChunkingChatInput:
    def test_rejects_chat_input_with_late_chunking(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = TokenEmbedIOProcessor.__new__(TokenEmbedIOProcessor)

        # Skip the base factory's rendering work; we only want the validation
        # that happens right after it.
        monkeypatch.setattr(
            PoolingIOProcessor,
            "get_request_factory_offline",
            lambda self, ctx: (lambda: iter([]), 1),
        )

        fake_params = MagicMock(spec=PoolingParams)
        fake_params.late_chunking_params = MagicMock()  # non-None -> opted in

        ctx = _make_encode_ctx(
            pooling_task="token_embed",
            inputs_seq=[[{"role": "user", "content": "hello"}]],
            pooling_params=fake_params,
        )

        with pytest.raises(
            VLLMValidationError, match="Late chunking does not support chat inputs"
        ):
            proc.get_request_factory_offline(ctx)

    def test_completion_input_with_late_chunking_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = TokenEmbedIOProcessor.__new__(TokenEmbedIOProcessor)
        monkeypatch.setattr(
            PoolingIOProcessor,
            "get_request_factory_offline",
            lambda self, ctx: (lambda: iter([]), 1),
        )
        proc.vllm_config = SimpleNamespace()  # unused on this branch

        fake_params = MagicMock(spec=PoolingParams)
        fake_params.late_chunking_params = MagicMock()

        ctx = _make_encode_ctx(
            pooling_task="token_embed",
            inputs_seq=["plain text"],
            pooling_params=fake_params,
        )

        # Should not raise.
        proc.get_request_factory_offline(ctx)
