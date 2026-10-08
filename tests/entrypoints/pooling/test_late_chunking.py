# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from vllm import PoolingParams
from vllm.config import PoolerConfig
from vllm.entrypoints.pooling.embed.io_processor import TokenEmbedIOProcessor
from vllm.entrypoints.pooling.late_chunking import build_late_chunking_metadata
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.typing import (
    OfflineEncodeInputsContext,
    OfflineOutputsContext,
)
from vllm.exceptions import VLLMValidationError
from vllm.outputs import (
    LateChunk,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
)
from vllm.pooling_params import LateChunkingParams
from vllm.renderers import TokenizeParams


@pytest.fixture
def late_chunk_processor():
    processor = object.__new__(TokenEmbedIOProcessor)
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
    processor.renderer.render_cmpl.side_effect = lambda **_: [
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
        pooling_params=PoolingParams(
            late_chunking_params=LateChunkingParams(chunk_size=2)
        ),
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
    assert ctx.pooling_params.late_chunking_params.metadata is None
    assert "prompt_token_offsets" not in result["prompts"]
    assert [
        c.char_range for c in result["params"].late_chunking_params.metadata.chunks
    ] == [(0, 1), (2, 3)]
    assert result["params"].late_chunking_params.metadata.input_tokens == 4


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


def _mock_chunk_llm(processor, outputs):
    llm = mock.Mock(spec=PoolingOfflineMixin)
    llm.encode = PoolingOfflineMixin.encode.__get__(llm)
    llm._run_tiling_engine = PoolingOfflineMixin._run_tiling_engine.__get__(llm)
    llm.pooling_io_processors = {"token_embed": processor}
    llm._executor = SimpleNamespace(map=map)
    llm.llm_engine = mock.Mock()
    llm.llm_engine.vllm_config.scheduler_config.max_num_seqs = 2
    llm.llm_engine.has_unfinished_requests.return_value = False
    llm.llm_engine.step.side_effect = outputs
    llm._render_and_add_requests = mock.Mock(
        side_effect=lambda **kw: [
            str(i) + "-internal" for i in range(len(kw["params"]))
        ]
    )
    return llm


def _chunk_output(request_id, rows=2, **kwargs):
    return PoolingRequestOutput(
        str(request_id),
        PoolingOutput(torch.ones(rows, 4)),
        [101, 1, 2, 102],
        0,
        True,
        **kwargs,
    )


def test_late_chunk_mapping_preserves_order_mixed_requests_and_errors(
    late_chunk_processor,
):
    error = RequestError("test_error", "original error")
    failed = _chunk_output(1, rows=0, error=error)
    llm = _mock_chunk_llm(
        late_chunk_processor, [[_chunk_output(2, rows=4), failed, _chunk_output(0)]]
    )
    params = [
        PoolingParams(late_chunking_params=LateChunkingParams(2)),
        PoolingParams(late_chunking_params=LateChunkingParams(1)),
        PoolingParams(),
    ]
    outputs = llm.encode(
        ["a b"] * 3, pooling_task="token_embed", pooling_params=params, use_tqdm=False
    )
    assert outputs[0].late_chunking.chunks == [
        LateChunk((0, 2), (0, 1)),
        LateChunk((2, 4), (2, 3)),
    ]
    assert outputs[1].late_chunking is None
    assert outputs[1].error is error
    assert outputs[2].late_chunking is None
    assert params[0].late_chunking_params is not None
    assert params[0].late_chunking_params.metadata is None
    llm.llm_engine.abort_request.assert_not_called()


@pytest.mark.parametrize(
    "error, abort_expected",
    [
        pytest.param(RuntimeError("step failed"), True, id="runtime-error"),
        pytest.param(KeyboardInterrupt(), False, id="keyboard-interrupt"),
    ],
)
def test_late_chunk_mapping_propagates_errors_and_is_not_reused(
    late_chunk_processor, error, abort_expected
):
    llm = _mock_chunk_llm(late_chunk_processor, error)
    params = PoolingParams(late_chunking_params=LateChunkingParams(2))
    with pytest.raises(type(error)):
        llm.encode(
            ["a b"] * 2,
            pooling_task="token_embed",
            pooling_params=params,
            use_tqdm=False,
        )
    if abort_expected:
        llm.llm_engine.abort_request.assert_called_once()
        assert set(llm.llm_engine.abort_request.call_args.args[0]) == {"0", "1"}
    else:
        llm.llm_engine.abort_request.assert_not_called()
    llm.llm_engine.step.side_effect = [
        [_chunk_output(1, rows=4), _chunk_output(0, rows=4)]
    ]
    outputs = llm.encode(["a b"] * 2, pooling_task="token_embed", use_tqdm=False)
    assert all(output.late_chunking is None for output in outputs)
    assert params.late_chunking_params is not None
    assert params.late_chunking_params.metadata is None


def test_late_chunk_mapping_rejects_successful_output_with_wrong_row_count(
    late_chunk_processor,
):
    llm = _mock_chunk_llm(late_chunk_processor, [[_chunk_output(0, rows=0)]])
    with pytest.raises(ValueError, match="does not match"):
        llm.encode(
            "a b",
            pooling_task="token_embed",
            pooling_params=PoolingParams(late_chunking_params=LateChunkingParams(2)),
            use_tqdm=False,
        )
    # All engine requests have finished before post-processing validates the output.
    llm.llm_engine.abort_request.assert_not_called()


def test_late_chunk_contexts_remain_isolated_when_rendering_interleaves(
    late_chunk_processor,
):
    processor = late_chunk_processor
    first, second = _late_chunk_context(), _late_chunk_context()
    shared = PoolingParams(late_chunking_params=LateChunkingParams(2))
    first.pooling_params = second.pooling_params = shared
    first_factory, _ = processor.get_request_factory_offline(first)
    second_factory, _ = processor.get_request_factory_offline(second)
    first_request, second_request = next(first_factory()), next(second_factory())
    processor.render(second_request)
    assert first.late_chunking[0].metadata is None
    processor.render(first_request)
    first_output = processor.post_process_offline(
        OfflineOutputsContext([_chunk_output(0)], late_chunking=first.late_chunking)
    )[0]
    second_output = processor.post_process_offline(
        OfflineOutputsContext([_chunk_output(0)], late_chunking=second.late_chunking)
    )[0]
    assert first_output.late_chunking == second_output.late_chunking
    assert first_output.late_chunking is not second_output.late_chunking
    assert shared.late_chunking_params is not None
    assert shared.late_chunking_params.metadata is None
