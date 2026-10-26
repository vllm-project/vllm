# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from fastapi.responses import ORJSONResponse
from transformers import BertConfig, BertTokenizerFast

from vllm import PoolingParams
from vllm.config import DeviceConfig, ModelConfig, PoolerConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.embed.io_processor import (
    JinaRankingTokenEmbedIOProcessor,
    TokenEmbedIOProcessor,
)
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.token_embed.late_chunking import (
    build_late_chunking_metadata,
)
from vllm.entrypoints.pooling.token_embed.protocol import (
    TokenEmbeddingRequest,
    TokenEmbeddingResponse,
)
from vllm.entrypoints.pooling.token_embed.serving import ServingTokenEmbedding
from vllm.entrypoints.pooling.typing import (
    OfflineEncodeInputsContext,
    OfflineOutputsContext,
    PoolingServeContext,
)
from vllm.entrypoints.pooling.utils import encode_pooling_output_float_or_ndarray
from vllm.exceptions import GenerationError, VLLMValidationError
from vllm.outputs import (
    EmbeddingRequestOutput,
    LateChunk,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
)
from vllm.pooling_params import LateChunkingParams
from vllm.renderers import TokenizeParams
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.renderers.hf import HfRenderer
from vllm.v1.engine.input_processor import InputProcessor


@pytest.fixture
def late_chunk_processor():
    processor = object.__new__(TokenEmbedIOProcessor)
    processor.model_config = SimpleNamespace(
        is_encoder_decoder=False,
        encoder_config={},
        max_model_len=32,
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


def test_late_chunk_render_requests_offsets_once(late_chunk_processor):
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
    assert "prompt_token_offsets" not in result["prompts"]
    assert "late_chunking" not in result
    assert [c.char_range for c in ctx.late_chunking[0].metadata.chunks] == [
        (0, 1),
        (2, 3),
    ]
    assert ctx.late_chunking[0].metadata.input_tokens == 4


@pytest.mark.parametrize("as_sequence", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_jina_ranking_rejects_chunking_before_tokenization(
    late_chunk_processor, as_sequence, chunk_size
):
    processor = object.__new__(JinaRankingTokenEmbedIOProcessor)
    processor.__dict__.update(late_chunk_processor.__dict__)
    ctx = _late_chunk_context(prompts=["document", "query"])
    params = PoolingParams(
        late_chunking_params=LateChunkingParams(chunk_size) if chunk_size else None
    )
    ctx.pooling_params = [params] if as_sequence else params
    factory, count = processor.get_request_factory_offline(ctx)
    assert count == 1
    request = next(factory())
    if chunk_size is not None:
        with pytest.raises(VLLMValidationError, match="JinaForRanking.*late chunking"):
            processor.render(request)
        late_chunk_processor.renderer.render_cmpl.assert_not_called()
    else:
        result = processor.render(request)
        assert result["prompts"]["type"] == "token"
        assert result["prompts"]["prompt_token_ids"] == [101, 1, 2, 102]
        late_chunk_processor.renderer.render_cmpl.assert_called_once()
        assert not request["tok_params"].return_token_offsets


@pytest.mark.parametrize(
    "prompts, kwargs, error",
    [
        pytest.param("", {}, "plain-text", id="empty"),
        pytest.param([1, 2], {}, "plain-text", id="token-ids"),
        pytest.param(
            {"prompt": "abc", "multi_modal_data": {}},
            {},
            "plain-text",
            id="multimodal",
        ),
    ]
    + [
        pytest.param("abc", {key: value}, "does not support", id=key)
        for key, value in (
            ("truncate_prompt_tokens", 2),
            ("truncation", True),
            ("pad_prompt_tokens", 8),
            ("padding", True),
            ("do_lower_case", True),
        )
    ],
)
def test_late_chunking_rejects_unsupported_input_before_render(
    late_chunk_processor, prompts, kwargs, error
):
    processor = late_chunk_processor
    with pytest.raises(VLLMValidationError, match=error):
        factory, _ = processor.get_request_factory_offline(
            _late_chunk_context(prompts, **kwargs)
        )
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
    llm.pooling_task = "token_embed"
    llm.embed = PoolingOfflineMixin.embed.__get__(llm)
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
    failed = _chunk_output(3, rows=0, error=error)
    llm = _mock_chunk_llm(late_chunk_processor, RuntimeError("step failed"))
    params = [
        PoolingParams(late_chunking_params=LateChunkingParams(2)),
        PoolingParams(),
        PoolingParams(late_chunking_params=LateChunkingParams(1)),
        PoolingParams(late_chunking_params=LateChunkingParams(2)),
    ]
    # A failed encode call must not leave metadata for the following batch.
    with pytest.raises(RuntimeError, match="step failed"):
        llm.encode(
            ["a b"] * 2,
            pooling_task="token_embed",
            pooling_params=params[:2],
            use_tqdm=False,
        )
    llm.llm_engine.step.side_effect = [
        [
            _chunk_output(2, rows=4),
            failed,
            _chunk_output(1, rows=4),
            _chunk_output(0),
        ]
    ]
    outputs = llm.embed(["a b"] * 4, pooling_params=params, use_tqdm=False)
    assert outputs[0].late_chunking.chunks == [
        LateChunk((0, 2), (0, 1)),
        LateChunk((2, 4), (2, 3)),
    ]
    assert outputs[1].late_chunking is None
    assert len(outputs[2].late_chunking.chunks) == 4
    assert outputs[3].late_chunking is None
    assert outputs[3].error is error


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
    first, second = _late_chunk_context("xy z"), _late_chunk_context()
    shared = PoolingParams(late_chunking_params=LateChunkingParams(2))
    first.pooling_params = second.pooling_params = shared
    first_factory, _ = processor.get_request_factory_offline(first)
    second_factory, _ = processor.get_request_factory_offline(second)
    first_request, second_request = next(first_factory()), next(second_factory())
    processor.render(second_request)
    assert first.late_chunking[0].metadata is None
    processor.renderer.render_cmpl.side_effect = lambda **_: [
        {
            "type": "token",
            "prompt_token_ids": [101, 1, 2, 102],
            "prompt_token_offsets": [(0, 0), (0, 2), (3, 4), (0, 0)],
        }
    ]
    processor.render(first_request)
    first_output = processor.post_process_offline(
        OfflineOutputsContext([_chunk_output(0)], late_chunking=first.late_chunking)
    )[0]
    second_output = processor.post_process_offline(
        OfflineOutputsContext([_chunk_output(0)], late_chunking=second.late_chunking)
    )[0]
    assert first_output.late_chunking.chunks[0].char_range == (0, 2)
    assert second_output.late_chunking.chunks[0].char_range == (0, 1)


def test_plain_token_embed_skips_chunk_metadata(late_chunk_processor):
    ctx = _late_chunk_context()
    ctx.pooling_params = PoolingParams()
    factory, _ = late_chunk_processor.get_request_factory_offline(ctx)
    request = next(factory())
    result = late_chunk_processor.render(request)
    assert not ctx.late_chunking
    assert result["params"].late_chunking_params is None
    assert not late_chunk_processor.renderer.render_cmpl.call_args.kwargs[
        "tok_params"
    ].return_token_offsets


def _online_chunk_context(texts, chunk_size=2):
    request = TokenEmbeddingRequest(
        input=texts,
        late_chunking_params=(
            {"chunk_size": chunk_size} if chunk_size is not None else None
        ),
    )
    return PoolingServeContext(
        request=request,
        model_name="model",
        request_id="same-client-id",
        pooling_params=request.to_pooling_params(),
        lora_request=None,
        priorities=0,
        prompt_extras=None,
    )


@pytest.mark.parametrize("task", ["embed", "token_embed"])
def test_embed_preserves_output_types(task):
    llm = mock.Mock(spec=PoolingOfflineMixin)
    llm.pooling_task = task
    output = _chunk_output(0)
    if task == "embed":
        output.outputs.data = output.outputs.data[0]
    llm.encode.return_value = [output]

    result = PoolingOfflineMixin.embed(llm, "text")[0]
    assert llm.encode.call_args.kwargs["pooling_task"] == task
    if task == "token_embed":
        assert result is output
    else:
        assert isinstance(result, EmbeddingRequestOutput)
        assert result.outputs.embedding == output.outputs.data.tolist()


@pytest.mark.parametrize(
    "runner, task, supported_tasks, error",
    [
        ("pooling", "classify", ("classify",), "--convert embed"),
        (
            "pooling",
            "classify",
            ("classify", "embed"),
            r'PoolerConfig\(task="embed"\)',
        ),
        ("generate", None, ("generate",), "--runner pooling"),
    ],
)
def test_embed_preserves_task_validation(runner, task, supported_tasks, error):
    llm = mock.Mock(spec=PoolingOfflineMixin)
    llm.runner_type = runner
    llm.pooling_task = task
    llm.supported_tasks = supported_tasks
    llm.encode = PoolingOfflineMixin.encode.__get__(llm)
    llm._verify_pooling_task = PoolingOfflineMixin._verify_pooling_task.__get__(llm)

    with pytest.raises(ValueError, match=error):
        PoolingOfflineMixin.embed(llm, "text", use_tqdm=False)
    llm._run_tiling_engine.assert_not_called()


@pytest.fixture
def token_embedding_serving(tmp_path):
    BertConfig(
        architectures=["BertModel"],
        hidden_size=4,
        num_attention_heads=1,
        num_hidden_layers=1,
        intermediate_size=8,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        runner="pooling",
        dtype="float32",
        max_model_len=32,
        pooler_config=PoolerConfig(task="token_embed"),
    )
    config = VllmConfig(
        model_config=model_config, device_config=DeviceConfig(device="cpu")
    )
    renderer = HfRenderer(config, tokenizer=BertTokenizerFast(vocab={"[UNK]": 0}))
    engine = Mock(
        spec=EngineClient,
        model_config=model_config,
        vllm_config=config,
        renderer=renderer,
        input_processor=InputProcessor(config, renderer),
    )
    try:
        yield ServingTokenEmbedding(
            engine,
            OpenAIServingModels(engine, []),
            request_logger=None,
            chat_template_config=ChatTemplateConfig(),
        )
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_token_embedding_response_preserves_ndarray_and_optional_ranges(
    late_chunk_processor, token_embedding_serving, chunk_size, dtype
):
    ctx = _online_chunk_context("a b", chunk_size=chunk_size)
    processor = late_chunk_processor
    ctx.engine_inputs = [
        processor.render(r) for r in processor.get_request_factory_online(ctx)
    ]
    output = _chunk_output(0, rows=4)
    # Exercise the existing encoder's contiguous-copy and dtype fallback paths.
    data = (torch.arange(16).reshape(4, 4).to(dtype) / 10).T
    output.outputs.data = data
    ctx.final_res_batch = [output]
    processor.post_process_online(ctx)
    response = token_embedding_serving._build_response(ctx)
    body = json.loads(response.body)
    TokenEmbeddingResponse.model_validate(body)
    item = body["data"][0]
    expected = ORJSONResponse(
        {"embedding": encode_pooling_output_float_or_ndarray(output)}
    )
    assert item["embedding"] == json.loads(expected.body)["embedding"]
    torch.testing.assert_close(torch.tensor(item["embedding"], dtype=dtype), data)
    if chunk_size is None:
        assert "late_chunking" not in item
        assert not ctx.late_chunking
        assert not processor.renderer.render_cmpl.call_args.kwargs[
            "tok_params"
        ].return_token_offsets
    else:
        assert item["late_chunking"]["chunks"][0]["char_range"] is None


def test_online_chunk_ranges_are_per_document(
    late_chunk_processor, token_embedding_serving
):
    processor = late_chunk_processor
    ctx = _online_chunk_context(["a b", "xy z"])

    def render_cmpl(**kwargs):
        text = kwargs["prompts"][0]["prompt"]
        split = text.index(" ")
        return [
            {
                "type": "token",
                "prompt_token_ids": [101, 1, 2, 102],
                "prompt_token_offsets": [
                    (0, 0),
                    (0, split),
                    (split + 1, len(text)),
                    (0, 0),
                ],
            }
        ]

    processor.renderer.render_cmpl.side_effect = render_cmpl
    requests = processor.get_request_factory_online(ctx)
    # Render out of order, as the online thread pool can do.
    second = processor.render(requests[1])
    first = processor.render(requests[0])
    ctx.engine_inputs = [first, second]
    ctx.final_res_batch = [_chunk_output(0), _chunk_output(1)]
    processor.post_process_online(ctx)

    body = json.loads(token_embedding_serving._build_response(ctx).body)
    assert body["usage"]["prompt_tokens"] == 8
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert [item["embedding"] for item in body["data"]] == [
        output.outputs.data.tolist() for output in ctx.final_res_batch
    ]
    assert [
        item["late_chunking"]["chunks"][0]["char_range"] for item in body["data"]
    ] == [[0, 1], [0, 2]]
    assert first["params"] is second["params"] is ctx.pooling_params


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_online_chunk_mapping_survives_reordered_completion_and_failure(
    late_chunk_processor, outcome
):
    processor = late_chunk_processor
    serving = object.__new__(ServingTokenEmbedding)
    ctx = _online_chunk_context(["a b", "a b"])
    ctx.engine_inputs = [
        processor.render(r) for r in processor.get_request_factory_online(ctx)
    ]
    entered = asyncio.Event()
    closed = asyncio.Event()
    error = RequestError("test_error", "original error")

    async def results():
        try:
            yield 1, _chunk_output(1)
            if outcome == "cancel":
                entered.set()
                await asyncio.Event().wait()
            yield 0, _chunk_output(0, error=error if outcome == "error" else None)
        finally:
            closed.set()

    ctx.result_generator = results()
    task = asyncio.create_task(serving._collect_batch(ctx))
    if outcome == "cancel":
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(GenerationError, match="original error"):
            await task
        await ctx.result_generator.aclose()
    else:
        await task
        processor.post_process_online(ctx)
        assert [result.request_id for result in ctx.final_res_batch] == ["0", "1"]
        assert all(result.late_chunking is not None for result in ctx.final_res_batch)
    assert closed.is_set()

    # A later request with the same caller ID must get its own mapping.
    following = _online_chunk_context("a b", chunk_size=1)
    following.engine_inputs = [
        processor.render(r) for r in processor.get_request_factory_online(following)
    ]
    following.final_res_batch = [_chunk_output(0, rows=4)]
    processor.post_process_online(following)
    assert following.final_res_batch[0].late_chunking.chunk_size == 1
    assert len(following.final_res_batch[0].late_chunking.chunks) == 4
