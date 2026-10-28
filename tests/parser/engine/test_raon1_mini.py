# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon template markers must agree with extraction and the grammar gate."""

import json
import string

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager


@pytest.fixture
def tokenizer():
    vocab = {char: i + 1 for i, char in enumerate(string.printable)}
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **vocab}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    return TokenizersBackend(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        additional_special_tokens=[
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
    )


@pytest.mark.parametrize("end_marker", ["</think>", "<tool_call>"])
def test_reasoning_gate_matches_extraction(tokenizer, end_marker):
    cls = ParserManager.get_parser(
        reasoning_parser_name="raon_1",
        tool_parser_name="raon_1",
        enable_auto_tools=True,
        tokenizer=tokenizer,
    )
    assert cls is not None
    parser = cls(tokenizer)
    request = ChatCompletionRequest(model="raon", messages=[])
    suffix = (
        "answer"
        if end_marker == "</think>"
        else (
            "<function=lookup><parameter=city>Seoul</parameter></function></tool_call>"
        )
    )
    text = "reasoning" + end_marker + suffix
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert isinstance(parser, DelegatingParser)
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "reasoning"
    assert content == ("answer" if end_marker == "</think>" else end_marker + suffix)
    assert parser.is_reasoning_end(ids)
    assert not parser.is_reasoning_end(tokenizer.encode("reasoning"))


def test_xml_tools_coerce_types_from_request_schema(tokenizer):
    cls = ParserManager.get_parser(
        tool_parser_name="raon_1", enable_auto_tools=True, tokenizer=tokenizer
    )
    request = ChatCompletionRequest(
        model="raon",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "limit": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            }
        ],
    )
    assert cls is not None
    parser = cls(tokenizer, tools=request.tools)
    assert isinstance(parser, DelegatingParser)
    result = parser.extract_tool_calls(
        "<tool_call>\n<function=lookup>\n<parameter=city>\nSeoul\n</parameter>\n"
        "<parameter=limit>3</parameter><parameter=enabled>true</parameter>"
        "</function>\n</tool_call>",
        request,
    )
    assert result.tools_called
    assert result.tool_calls[0].function.name == "lookup"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "city": "Seoul",
        "limit": 3,
        "enabled": True,
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_streamed_markup_matches_complete_call(tokenizer, chunk_size):
    cls = ParserManager.get_parser(
        reasoning_parser_name="raon_1",
        tool_parser_name="raon_1",
        enable_auto_tools=True,
        tokenizer=tokenizer,
    )
    assert cls is not None
    request = ChatCompletionRequest(
        model="raon",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )
    parser = cls(tokenizer, tools=request.tools)
    text = (
        "reasoning</think><tool_call>\n<function=lookup>\n"
        "<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    reasoning, names, arguments = [], [], []
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        delta = parser.parse_delta(
            tokenizer.decode(chunk),
            chunk,
            request,
            finished=start + len(chunk) == len(ids),
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning.append(delta.reasoning)
        for call in delta.tool_calls or []:
            if call.function:
                if call.function.name:
                    names.append(call.function.name)
                if call.function.arguments:
                    arguments.append(call.function.arguments)
    assert "".join(reasoning) == "reasoning"
    assert "".join(names) == "lookup"
    assert json.loads("".join(arguments)) == {"city": "Seoul"}
