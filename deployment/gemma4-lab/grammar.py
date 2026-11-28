# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-native tool branch for vLLM 0.32.0's tools-or-response composition."""

from xgrammar import StructuralTag
from xgrammar.builtin_structural_tag import get_gemma_4_structural_tag
from xgrammar.structural_tag import (
    ConstStringFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
)


def bound_reply_whitespace(tag):
    """The 0.32 delegating parser composes OR(native_tools, request_schema)."""
    fmt = tag["format"]
    if fmt["type"] == "or" and len(fmt["elements"]) == 2:
        answer = fmt["elements"][1]
        if answer["type"] == "json_schema" and answer.get("style", "json") == "json":
            answer["max_whitespace_cnt"] = 1
            return True
    return False


def resolve_tool_token_ids(fmt, vocab):
    """The 0.32 API preflight validates without tokenizer_info; it needs numeric IDs."""
    if isinstance(fmt, list):
        for node in fmt:
            resolve_tool_token_ids(node, vocab)
    elif isinstance(fmt, dict) and fmt.get("type") != "json_schema":
        if fmt.get("type") == "token" and fmt["token"] in (
            "<|tool_call>",
            "<tool_call|>",
        ):
            fmt["token"] = vocab[fmt["token"]]
        else:
            for node in fmt.values():
                resolve_tool_token_ids(node, vocab)


def gemma_tools(tools, builtin_tools, choice, reasoning, token_suffix):
    if builtin_tools or token_suffix or reasoning:
        raise ValueError(
            "The lab only supports function tools with native thinking off"
        )
    tag = get_gemma_4_structural_tag(tools, tool_choice=choice, reasoning="disabled")
    if choice == "required":
        # vLLM ORs this branch with the answer schema. No free text before a call.
        start, end = "<|tool_call>", "<tool_call|>"
        options = []
        for t in tag.format.tags:
            assert t.begin.startswith(start) and t.end == end
            options.append(
                SequenceFormat(
                    elements=[
                        ConstStringFormat(value=t.begin[len(start) :]),
                        t.content,
                    ]
                )
            )
        native = TagFormat(
            begin=TokenFormat(token=start),
            content=OrFormat(elements=options),
            end=TokenFormat(token=end),
        )
        return StructuralTag(
            format=TagsWithSeparatorFormat(
                tags=[native], separator="", at_least_one=True
            )
        )
    return tag
