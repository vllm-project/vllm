# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Executor, Future
from types import SimpleNamespace

import pytest

from vllm import PoolingParams, PoolingRequestOutput
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


class _InlineExecutor(Executor):
    """Executor that renders on the calling thread, for deterministic tests."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class _FakeEngine:
    """Engine that finishes one request per step, in arrival order."""

    def __init__(self, max_num_seqs: int):
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs)
        )
        self.in_core: list[str] = []
        self.aborted: list[str] = []
        self.num_added = 0

    def has_unfinished_requests(self) -> bool:
        return bool(self.in_core)

    def step(self) -> list[PoolingRequestOutput]:
        if not self.in_core:
            return []
        request_id = self.in_core.pop(0)
        return [
            PoolingRequestOutput(
                request_id=request_id,
                outputs=None,
                prompt_token_ids=[],
                num_cached_tokens=0,
                finished=True,
            )
        ]

    def abort_request(self, request_ids) -> None:
        self.aborted.extend(request_ids)


def _make_tiling_llm(
    monkeypatch,
    engine: _FakeEngine,
    on_add=None,
) -> PoolingOfflineMixin:
    """A pooling LLM whose engine and rendering are stubbed out.

    `_run_tiling_engine` runs for real so that the interleaving between
    preprocessing and `add_request` is observable.
    """
    llm = PoolingOfflineMixin.__new__(PoolingOfflineMixin)
    llm.llm_engine = engine
    monkeypatch.setattr(llm, "_executor", _InlineExecutor(), raising=False)

    def render_and_add_requests(prompts, params, lora_requests, priorities):
        if on_add is not None:
            on_add()
        request_ids = [str(engine.num_added + i) for i in range(len(prompts))]
        engine.num_added += len(prompts)
        engine.in_core.extend(request_ids)
        return request_ids

    monkeypatch.setattr(llm, "_render_and_add_requests", render_and_add_requests)
    return llm


def _render_param(index: int) -> dict:
    return {
        "prompts": index,
        "params": None,
        "lora_requests": None,
        "priorities": 0,
    }


def test_tiling_engine_adds_before_preprocessing_every_prompt(monkeypatch):
    engine = _FakeEngine(max_num_seqs=4)
    num_requests = 1000
    max_requests_in_core = 2 * 4
    produced = 0
    produced_at_first_add = None

    def request_factory():
        nonlocal produced
        for i in range(num_requests):
            produced += 1
            yield _render_param(i)

    def on_add():
        nonlocal produced_at_first_add
        if produced_at_first_add is None:
            produced_at_first_add = produced

    llm = _make_tiling_llm(monkeypatch, engine, on_add=on_add)
    io_processor = SimpleNamespace(render=lambda render_params: render_params)

    outputs = llm._run_tiling_engine(
        io_processor, request_factory, num_requests, use_tqdm=False
    )

    assert len(outputs) == num_requests
    assert [int(o.request_id) for o in outputs] == sorted(
        int(o.request_id) for o in outputs
    )
    assert produced_at_first_add is not None
    assert produced_at_first_add <= max_requests_in_core, (
        "The whole prompt list was preprocessed before the engine received "
        "its first request."
    )


def test_tiling_engine_aborts_added_requests_on_bad_prompt(monkeypatch):
    engine = _FakeEngine(max_num_seqs=1)
    llm = _make_tiling_llm(monkeypatch, engine)

    def request_factory():
        for i in range(3):
            yield _render_param(i)
        raise ValueError("bad prompt")

    io_processor = SimpleNamespace(render=lambda render_params: render_params)

    with pytest.raises(ValueError, match="bad prompt"):
        llm._run_tiling_engine(io_processor, request_factory, 5, use_tqdm=False)

    assert engine.aborted, (
        "Requests already added to the engine were not aborted when a later "
        "prompt failed to be preprocessed."
    )


def test_offline_factory_does_not_mutate_caller_pooling_params(
    processor: PoolingIOProcessor,
):
    processor.model_config = SimpleNamespace(is_encoder_decoder=False)
    processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=None)
    )
    params = PoolingParams()
    ctx = OfflineEncodeInputsContext(
        pooling_task="embed",
        tokenization_kwargs=None,
        lora_request=None,
        priorities=None,
        prompts=[[1], [2]],
        pooling_params=params,
    )

    factory, num_requests = processor.get_request_factory_offline(ctx)
    assert num_requests == 2

    requests = list(factory())
    assert [request["params"].task for request in requests] == ["embed", "embed"]
    assert params.task is None
