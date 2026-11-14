# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Whitespace-cap behaviour of the tool-call structural-tag grammar (issue #67029).

xgrammar's default admits an unbounded run of whitespace at every JSON value
position of a grammar-constrained tool call, so a grammar-masked token can
turn into generation that loops on whitespace until max_tokens. The registry
caps the run via VLLM_TOOL_CALL_MAX_WHITESPACE (default 16).

These tests exercise both builder families: the xgrammar builtin templates
(llama) and the vLLM-owned builders (hermes).
"""

import json

import pytest
from xgrammar import Grammar, StructuralTag
from xgrammar.testing import _is_grammar_accept_string

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


@pytest.fixture
def sample_tools_strict() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


def _llama_required_call(spaces_at_value: int) -> str:
    # xgrammar's builtin llama template with tool_choice="required": the
    # grammar starts directly at the JSON call body.
    return (
        '{"name": "get_weather", "parameters": {"city": '
        + " " * spaces_at_value
        + '"x"}}'
    )


def _hermes_required_call(tag: StructuralTag, spaces_at_value: int) -> str:
    # Derive the call shape from the tag itself: a hermes tool call is
    # begin + JSON arguments + end, with the whitespace probe inside the
    # arguments' first string value.
    fmt = tag.model_dump()["format"]
    assert fmt["type"] == "tags_with_separator"
    node = fmt["tags"][0]
    assert node["type"] == "tag"
    return node["begin"] + '{"city": ' + " " * spaces_at_value + '"x"}' + node["end"]


def _build_tag(model: str, tools: list[ChatCompletionToolsParam]) -> StructuralTag:
    tag = get_model_structural_tag(
        model=model,
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )
    assert tag is not None
    return tag


@pytest.mark.parametrize("model", ["llama", "hermes"])
def test_tool_call_whitespace_run_is_capped(
    model: str,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    """A whitespace run at an argument value position is bounded.

    A compact call and an indent-4 pretty-printed call are still accepted;
    a run longer than the cap is rejected instead of being admitted until
    max_tokens.
    """
    tag = _build_tag(model, sample_tools_strict)
    grammar = Grammar.from_structural_tag(json.dumps(tag.model_dump()))

    def call(spaces: int) -> str:
        if model == "llama":
            return _llama_required_call(spaces)
        return _hermes_required_call(tag, spaces)

    assert _is_grammar_accept_string(grammar, call(4))
    assert not _is_grammar_accept_string(grammar, call(20))


@pytest.mark.parametrize("model", ["llama", "hermes"])
def test_tool_call_whitespace_cap_can_be_disabled(
    model: str,
    sample_tools_strict: list[ChatCompletionToolsParam],
    monkeypatch: pytest.MonkeyPatch,
):
    """VLLM_TOOL_CALL_MAX_WHITESPACE=none restores the unbounded default."""
    monkeypatch.setattr(envs, "VLLM_TOOL_CALL_MAX_WHITESPACE", None)
    tag = _build_tag(model, sample_tools_strict)
    grammar = Grammar.from_structural_tag(json.dumps(tag.model_dump()))

    def call(spaces: int) -> str:
        if model == "llama":
            return _llama_required_call(spaces)
        return _hermes_required_call(tag, spaces)

    assert _is_grammar_accept_string(grammar, call(20))


def test_tool_call_whitespace_env_parsing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_TOOL_CALL_MAX_WHITESPACE", "8")
    assert envs.maybe_tool_call_max_whitespace() == 8

    monkeypatch.setenv("VLLM_TOOL_CALL_MAX_WHITESPACE", "none")
    assert envs.maybe_tool_call_max_whitespace() is None

    monkeypatch.setenv("VLLM_TOOL_CALL_MAX_WHITESPACE", "0")
    with pytest.raises(ValueError, match="VLLM_TOOL_CALL_MAX_WHITESPACE"):
        envs.maybe_tool_call_max_whitespace()

    monkeypatch.delenv("VLLM_TOOL_CALL_MAX_WHITESPACE")
    assert envs.maybe_tool_call_max_whitespace() == 16
