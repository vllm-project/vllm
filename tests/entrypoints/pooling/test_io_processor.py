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


@pytest.fixture
def late_chunk_processor(processor):
    from unittest.mock import Mock

    from vllm.config import PoolerConfig

    processor.model_config = SimpleNamespace(
        is_encoder_decoder=False,
        architecture="NomicBertModel",
        model_impl="auto",
        is_matryoshka=False,
        hf_config=SimpleNamespace(),
        pooler_config=PoolerConfig(seq_pooling_type="MEAN", tok_pooling_type="ALL"),
    )
    processor.vllm_config = SimpleNamespace(
        model_config=processor.model_config,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        lora_config=None,
    )
    processor.renderer = Mock()
    processor.renderer.default_cmpl_tok_params = TokenizeParams(max_total_tokens=32)
    processor.renderer.render_cmpl.return_value = [
        {
            "type": "token",
            "prompt_token_ids": [101, 1, 2, 102],
            "prompt_token_offsets": [(0, 0), (0, 1), (2, 3), (0, 0)],
        }
    ]
    return processor


def _late_chunk_context(prompts="a b", **kwargs):
    return OfflineEncodeInputsContext(
        pooling_task="token_embed",
        tokenization_kwargs=kwargs or None,
        lora_request=None,
        priorities=None,
        prompts=prompts,
        pooling_params=PoolingParams(late_chunk_size=2),
    )


def test_late_chunk_render_requests_offsets_once_and_keeps_params_isolated(
    late_chunk_processor,
):
    processor = late_chunk_processor
    ctx = _late_chunk_context()
    factory, count = processor.get_request_factory_offline(ctx)
    render_params = next(factory())
    result = processor.render(render_params)
    assert count == 1
    processor.renderer.render_cmpl.assert_called_once()
    assert processor.renderer.render_cmpl.call_args.kwargs[
        "tok_params"
    ].return_token_offsets
    assert not render_params["tok_params"].return_token_offsets
    assert ctx.pooling_params.task is None
    assert "prompt_token_offsets" not in result["prompts"]
    assert [c.char_range for c in result["late_chunking"].chunks] == [(0, 1), (2, 3)]
    assert result["late_chunking"].input_tokens == 4


@pytest.mark.parametrize(
    "prompts", ["", [1, 2], {"prompt": "abc", "multi_modal_data": {}}]
)
def test_late_chunking_rejects_unsupported_input_before_render(
    late_chunk_processor, prompts
):
    processor = late_chunk_processor
    factory, _ = processor.get_request_factory_offline(_late_chunk_context(prompts))
    with pytest.raises(VLLMValidationError, match="plain-text"):
        processor.render(next(factory()))
    processor.renderer.render_cmpl.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncate_prompt_tokens": 2},
        {"truncation": True},
        {"pad_prompt_tokens": 8},
        {"padding": True},
        {"do_lower_case": True},
    ],
)
def test_late_chunking_rejects_text_changes_before_render(late_chunk_processor, kwargs):
    processor = late_chunk_processor
    with pytest.raises(VLLMValidationError, match="does not support"):
        factory, _ = processor.get_request_factory_offline(
            _late_chunk_context(**kwargs)
        )
        processor.render(next(factory()))
    processor.renderer.render_cmpl.assert_not_called()


@pytest.mark.parametrize("setting", ["prefix_cache", "chunked_prefill", "lora"])
def test_late_chunking_rejects_unsupported_execution_before_render(
    late_chunk_processor, setting
):
    processor = late_chunk_processor
    if setting == "prefix_cache":
        processor.vllm_config.cache_config.enable_prefix_caching = True
    elif setting == "chunked_prefill":
        processor.vllm_config.scheduler_config.enable_chunked_prefill = True
    else:
        processor.vllm_config.lora_config = object()
    factory, _ = processor.get_request_factory_offline(_late_chunk_context())
    with pytest.raises(VLLMValidationError, match="does not support"):
        processor.render(next(factory()))
    processor.renderer.render_cmpl.assert_not_called()


def test_late_chunk_ranges_keep_unicode_overlaps_and_special_only_chunks():
    from vllm.entrypoints.pooling.late_chunking import build_late_chunking_metadata

    text = "中 😀 e\u0301"
    offsets = [(0, 0), (0, 1), (2, 3), (2, 3), (4, 6), (0, 0)]
    metadata = build_late_chunking_metadata(text, len(offsets), offsets, 1)
    assert [c.char_range for c in metadata.chunks] == [
        None,
        (0, 1),
        (2, 3),
        (2, 3),
        (4, 6),
        None,
    ]
    assert [
        text[slice(*c.char_range)] if c.char_range else None for c in metadata.chunks
    ] == [None, "中", "😀", "😀", "e\u0301", None]
    chunks = build_late_chunking_metadata(text, len(offsets), offsets, 4).chunks
    assert [(c.token_range, c.char_range) for c in chunks] == [
        ((0, 4), (0, 3)),
        ((4, 6), (4, 6)),
    ]


@pytest.mark.parametrize(
    "offsets", [None, [(0, 1)], [(-1, 1), (0, 0)], [(0, 4), (0, 0)], [(2, 3), (0, 1)]]
)
def test_late_chunk_ranges_reject_missing_or_invalid_offsets(offsets):
    from vllm.entrypoints.pooling.late_chunking import build_late_chunking_metadata

    with pytest.raises(VLLMValidationError, match="offsets"):
        build_late_chunking_metadata("abc", 2, offsets, 2)
