# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.parser_manager import ParserManager

MODEL = "google/gemma-4-E2B-it"
REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"
FUNCTION = {
    "name": "record_color",
    "description": "Record a color.",
    "parameters": {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["blue"]}},
        "required": ["color"],
        "additionalProperties": False,
    },
    "strict": True,
}


@pytest.fixture
def gemma4_parser_cls():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="gemma4",
        reasoning_parser_name="gemma4",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls


@pytest.fixture
def gemma4_tokenizer():
    special_tokens = [
        "<|tool_call>",
        "<tool_call|>",
        "<|channel>",
        "<channel|>",
        '<|"|>',
    ]
    vocab = {token: i for i, token in enumerate(["<unk>", *special_tokens])}
    return TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        additional_special_tokens=special_tokens,
    )


def _chat_choice(mode):
    if mode == "named":
        return {"type": "function", "function": {"name": "record_color"}}
    return mode


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("mode", ["auto", "required", "named"])
def test_gemma4_strict_tool_grammar(
    monkeypatch, gemma4_parser_cls, gemma4_tokenizer, api, mode
):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    if api == "chat":
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [],
                "tools": [{"type": "function", "function": FUNCTION}],
                "tool_choice": _chat_choice(mode),
            }
        )
    else:
        request = ResponsesRequest.model_validate(
            {
                "input": "Record a color.",
                "tools": [{"type": "function", **FUNCTION}],
                "tool_choice": {"type": "function", "name": "record_color"}
                if mode == "named"
                else mode,
            }
        )
    parser = gemma4_parser_cls(gemma4_tokenizer, tools=request.tools)
    adjusted = parser.adjust_request(request)
    params = adjusted.extract_structured_outputs()
    assert params is not None
    assert params.structural_tag is not None
    assert adjusted.skip_special_tokens is False

    grammar = Grammar.from_structural_tag(params.structural_tag)
    valid = '<|tool_call>call:record_color{color:<|"|>blue<|"|>}<tool_call|>'
    assert _is_grammar_accept_string(grammar, valid)
    assert not _is_grammar_accept_string(grammar, valid.replace("blue", "red"))
    assert not _is_grammar_accept_string(
        grammar, valid.replace("record_color", "other")
    )
    assert _is_grammar_accept_string(grammar, "Plain text.") == (mode == "auto")


@pytest.mark.parametrize(
    ("mode", "strict", "enabled"),
    [("auto", False, "1"), ("none", True, "1"), ("required", True, "0")],
)
def test_gemma4_skips_unrequested_tool_grammar(
    monkeypatch, gemma4_parser_cls, gemma4_tokenizer, mode, strict, enabled
):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", enabled)
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [],
            "tools": [{"type": "function", "function": {**FUNCTION, "strict": strict}}],
            "tool_choice": mode,
        }
    )
    parser = gemma4_parser_cls(gemma4_tokenizer, tools=request.tools)
    assert parser.adjust_request(request).extract_structured_outputs() is None


@pytest.fixture(scope="module")
def gemma4_server():
    args = [
        "--revision",
        REVISION,
        "--tool-call-parser",
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--language-model-only",
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(
        MODEL, args, env_dict={"VLLM_ENFORCE_STRICT_TOOL_CALLING": "1"}
    ) as server:
        yield server


@pytest.mark.parametrize("mode", ["required", "named"])
@pytest.mark.parametrize("stream", [False, True])
def test_gemma4_strict_tool_grammar_server(gemma4_server, mode, stream):
    with gemma4_server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Call record_color with color set to red."}
            ],
            tools=[{"type": "function", "function": FUNCTION}],
            tool_choice=_chat_choice(mode),
            parallel_tool_calls=False,
            temperature=0,
            seed=42,
            max_completion_tokens=96,
            stream=stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if stream:
            name_parts = []
            argument_parts = []
            for chunk in response:
                for choice in chunk.choices:
                    for call in choice.delta.tool_calls or []:
                        assert call.index == 0
                        if call.function:
                            name_parts.append(call.function.name or "")
                            argument_parts.append(call.function.arguments or "")
            name = "".join(name_parts)
            arguments = "".join(argument_parts)
        else:
            calls = response.choices[0].message.tool_calls
            assert calls and len(calls) == 1
            name = calls[0].function.name
            arguments = calls[0].function.arguments

    assert name == "record_color"
    assert json.loads(arguments) == {"color": "blue"}
