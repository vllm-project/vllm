# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models
from transformers import TokenizersBackend

from tests.parser.engine.replay_harness import _test_request, collect_output
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.deepseek_v41 import DeepSeekV41Parser
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser.qwen3 import Qwen3Parser
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer


def _byte_fallback_fixture(holdback):
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["<unk>"] = 256
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.decoder = decoders.ByteFallback()
    backend.add_special_tokens(
        [AddedToken("<think>", special=True), AddedToken("</think>", special=True)]
    )
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    request = EngineCoreRequest(
        request_id="byte-fallback-counts",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=SamplingParams(
            skip_special_tokens=False,
            stop=["Z" * (holdback + 1)] if holdback else None,
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return tokenizer, FastIncrementalDetokenizer(tokenizer, request)


def _stream_tokens(parser, detokenizer, ids, chunk_size):
    request = _test_request()
    deltas = []
    chunk_size = chunk_size or len(ids)
    for start in range(0, len(ids), chunk_size):
        batch = ids[start : start + chunk_size]
        finished = start + chunk_size >= len(ids)
        detokenizer.update(batch, stop_terminated=False)
        text = detokenizer.get_next_output_text(finished=finished, delta=True)
        deltas.append(parser.parse_delta(text, batch, request, finished=finished))
    return collect_output(deltas)


@pytest.mark.parametrize("parser_cls", [DeepSeekV4Parser, Qwen3Parser, Glm47MoeParser])
@pytest.mark.parametrize("chunk_size", [1, 2, None])
@pytest.mark.parametrize("holdback", [0, 3, 40])
def test_byte_fallback_reasoning_usage_matches_batch(parser_cls, chunk_size, holdback):
    tokenizer, detokenizer = _byte_fallback_fixture(holdback)
    vocab = tokenizer.get_vocab()
    ids = (
        [vocab["<think>"]]
        + list("中".encode())
        + [vocab["</think>"]]
        + list("文".encode())
    )
    parser = parser_cls(tokenizer)
    output = _stream_tokens(parser, detokenizer, ids, chunk_size)
    assert output.reasoning == "中"
    assert output.content == "文"
    assert parser.count_reasoning_tokens(ids) == 3

    batch_parser = parser_cls(tokenizer)
    reasoning, content, _ = batch_parser.parse(
        tokenizer.decode(ids, skip_special_tokens=False),
        _test_request(),
        model_output_token_ids=ids,
    )
    assert (reasoning, content) == (output.reasoning, output.content)
    assert batch_parser.count_reasoning_tokens(ids) == 3


@pytest.mark.parametrize("chunk_size", [1, 2, None])
@pytest.mark.parametrize("holdback", [0, 3, 40])
@pytest.mark.parametrize("reasoning_tail", [False, True])
def test_invisible_tail_is_counted_in_its_phase(chunk_size, holdback, reasoning_tail):
    tokenizer, detokenizer = _byte_fallback_fixture(holdback)
    vocab = tokenizer.get_vocab()
    ids = [vocab["<think>"]]
    if not reasoning_tail:
        ids += list("中".encode()) + [vocab["</think>"]]
    ids += list("文".encode())[:2]
    parser = DeepSeekV4Parser(tokenizer)
    output = _stream_tokens(parser, detokenizer, ids, chunk_size)
    assert output.reasoning == ("" if reasoning_tail else "中")
    assert output.content == ""
    assert parser.count_reasoning_tokens(ids) == (2 if reasoning_tail else 3)


def test_byte_fallback_inside_text_tool_marker_is_not_reasoning():
    tokenizer, detokenizer = _byte_fallback_fixture(0)
    text = (
        '中<｜DSML｜ calls><｜DSML｜ invoke name="emit">'
        '<｜DSML｜ parameter name="value" string="true">ok</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    ids = [tokenizer.get_vocab()["<think>"]] + list(text.encode())
    parser = DeepSeekV41Parser(tokenizer)
    output = _stream_tokens(parser, detokenizer, ids, 1)
    assert output.reasoning == "中"
    assert output.content == ""
    assert output.tool_calls == [{"name": "emit", "arguments": '{"value": "ok"}'}]
    assert parser.count_reasoning_tokens(ids) == 3


@pytest.mark.parametrize("chunk_size", [1, 2, None])
@pytest.mark.parametrize("holdback", [0, 3, 40])
def test_incomplete_utf8_before_reasoning_end(chunk_size, holdback):
    tokenizer, detokenizer = _byte_fallback_fixture(holdback)
    vocab = tokenizer.get_vocab()
    ids = (
        [vocab["<think>"]]
        + list("中".encode())[:2]
        + [vocab["</think>"]]
        + list("文".encode())
    )
    parser = DeepSeekV4Parser(tokenizer)
    output = _stream_tokens(parser, detokenizer, ids, chunk_size)
    assert output.reasoning == "\ufffd\ufffd"
    assert output.content == "文"
    assert parser.count_reasoning_tokens(ids) == 2


@pytest.mark.parametrize("terminal", ["THINK_END", DROP_TERMINAL])
@pytest.mark.parametrize("resolve", [False, True])
def test_deferred_boundaries_preserve_invisible_prefix_counts(terminal, resolve):
    tokenizer, _ = _byte_fallback_fixture(0)
    end_id = tokenizer.get_vocab()["</think>"]
    scanner = TokenIDScanner({end_id: terminal}, tokenizer)
    assert scanner.scan("", [0xE4, 0xB8, end_id, 0xE6, end_id, 0x41]) == []
    items = scanner.scan("</think></think>", []) if resolve else scanner.flush_pending()
    assert items == [
        TextChunk("", token_count=2),
        PreLexedTerminal(terminal, end_id, "</think>"),
        TextChunk("", token_count=1),
        PreLexedTerminal(terminal, end_id, "</think>"),
        TextChunk("", token_count=1),
    ]
