# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from vllm import PoolingParams
from vllm.config import PoolerConfig
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.late_chunking import build_late_chunking_metadata
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.typing import OfflineEncodeInputsContext
from vllm.exceptions import VLLMValidationError
from vllm.outputs import (
    LateChunk,
    LateChunkingMetadata,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
)
from vllm.renderers import TokenizeParams


@pytest.fixture
def late_chunk_processor():
    processor = object.__new__(PoolingIOProcessor)
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
    with pytest.raises(VLLMValidationError, match="offsets"):
        build_late_chunking_metadata("abc", 2, offsets, 2)


def _mock_chunk_tiling_engine(outputs):
    llm = mock.Mock(spec=PoolingOfflineMixin)
    llm._run_tiling_engine = PoolingOfflineMixin._run_tiling_engine.__get__(llm)
    llm._executor = SimpleNamespace(map=map)
    llm.llm_engine = mock.Mock()
    llm.llm_engine.vllm_config.scheduler_config.max_num_seqs = 2
    llm.llm_engine.has_unfinished_requests.return_value = False
    llm.llm_engine.step.side_effect = outputs
    llm._render_and_add_requests = mock.Mock(return_value=["0-internal", "1-internal"])
    requests = [
        {
            "prompts": {"type": "token", "prompt_token_ids": [1, 2]},
            "params": PoolingParams(task="token_embed", late_chunk_size=2),
            "lora_requests": None,
            "priorities": 0,
            "late_chunking": LateChunkingMetadata(
                chunk_size=2,
                input_tokens=2,
                chunks=[LateChunk((0, 2), (i, i + 2))],
            ),
        }
        for i in range(2)
    ]
    return llm, requests


def _chunk_output(request_id, **kwargs):
    return PoolingRequestOutput(
        str(request_id), PoolingOutput(torch.ones(1, 4)), [1, 2], 0, True, **kwargs
    )


def test_late_chunk_mapping_follows_request_ids_and_preserves_request_errors():
    error = RequestError("test_error", "original error")
    failed = _chunk_output(1, error=error)
    # A failed request need not have a valid chunk tensor.
    failed.outputs.data = failed.outputs.data[:0]
    llm, requests = _mock_chunk_tiling_engine([[failed, _chunk_output(0)]])
    outputs = llm._run_tiling_engine(
        SimpleNamespace(render=lambda x: x), lambda: iter(requests), 2, use_tqdm=False
    )
    assert outputs[0].late_chunking is requests[0]["late_chunking"]
    assert outputs[1].late_chunking is None
    assert outputs[1].error is error
    llm.llm_engine.abort_request.assert_not_called()


@pytest.mark.parametrize(
    "error, abort_expected",
    [
        pytest.param(RuntimeError("step failed"), True, id="runtime-error"),
        pytest.param(KeyboardInterrupt(), False, id="keyboard-interrupt"),
    ],
)
def test_late_chunk_mapping_propagates_errors_and_is_not_reused(error, abort_expected):
    llm, requests = _mock_chunk_tiling_engine(error)
    processor = SimpleNamespace(render=lambda x: x)
    with pytest.raises(type(error)):
        llm._run_tiling_engine(processor, lambda: iter(requests), 2, use_tqdm=False)
    if abort_expected:
        llm.llm_engine.abort_request.assert_called_once()
        assert set(llm.llm_engine.abort_request.call_args.args[0]) == {"0", "1"}
    else:
        llm.llm_engine.abort_request.assert_not_called()
    for request in requests:
        del request["late_chunking"]
        request["params"] = PoolingParams(task="token_embed")
    llm.llm_engine.step.side_effect = [[_chunk_output(1), _chunk_output(0)]]
    outputs = llm._run_tiling_engine(
        processor, lambda: iter(requests), 2, use_tqdm=False
    )
    assert all(output.late_chunking is None for output in outputs)


def test_late_chunk_mapping_rejects_successful_output_with_wrong_row_count():
    first = _chunk_output(0)
    first.outputs.data = first.outputs.data[:0]
    llm, requests = _mock_chunk_tiling_engine([[first, _chunk_output(1)]])
    with pytest.raises(ValueError, match="does not match"):
        llm._run_tiling_engine(
            SimpleNamespace(render=lambda x: x),
            lambda: iter(requests),
            2,
            use_tqdm=False,
        )
    assert set(llm.llm_engine.abort_request.call_args.args[0]) == {"0", "1"}
