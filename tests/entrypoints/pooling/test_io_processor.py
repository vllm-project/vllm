# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor
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


@pytest.mark.parametrize(
    "explicit_chat_template,tokenizer_chat_template",
    [
        (None, "unrelated chat template"),
        ("explicit", "unrelated chat template"),
        (None, {"default": "unrelated chat template", "score": "score template"}),
    ],
)
def test_saved_sentence_transformers_chat_template(
    monkeypatch, tmp_path, explicit_chat_template, tokenizer_chat_template
):
    """The chat template named by the Sentence Transformers config must exist,
    unless an explicit chat template takes precedence. It is resolved the same
    way by every processor of the model (e.g. /score and /rerank)."""
    import json
    from unittest.mock import MagicMock

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
    st_files = {
        "config_sentence_transformers.json": {"model_type": "CrossEncoder"},
        "modules.json": [
            {"path": "", "type": "sentence_transformers.models.Transformer"}
        ],
        "sentence_bert_config.json": {
            "transformer_task": "sequence-classification",
            "modality_config": {"message": {"format": "flat"}},
            "processing_kwargs": {"chat_template": {"chat_template": "score"}},
        },
    }
    for name, content in st_files.items():
        (tmp_path / name).write_text(json.dumps(content))

    model_config = MagicMock(model=str(tmp_path), revision=None)
    model_config.hf_config.is_original_qwen3_reranker = False
    from tokenizers import Tokenizer, models
    from transformers import TokenizersBackend

    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, "[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = tokenizer_chat_template
    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer

    def create():
        return CrossEncoderIOProcessor(
            vllm_config=MagicMock(model_config=model_config),
            renderer=renderer,
            chat_template_config=MagicMock(chat_template=explicit_chat_template),
        )

    if explicit_chat_template is not None:
        assert create().saved_chat_template is None
    elif isinstance(tokenizer_chat_template, str):
        with pytest.raises(ValueError, match="no 'score' chat template"):
            create()
    else:
        assert [create().saved_chat_template for _ in range(2)] == [
            "score template",
            "score template",
        ]
