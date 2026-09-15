# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import OfflineEncodeInputsContext
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams


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
