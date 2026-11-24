# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XGrammar formats for Muse's recipient-routed assistant messages."""

from __future__ import annotations

from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
)

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_ASSISTANT_MESSAGE_SEPARATOR = f"{_START}assistant "
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_FUNCTION_CALLS_CLOSE = "</atem:function_calls>"
_INVOKE_CLOSE = "</atem:invoke>"
_PARAMETER_CLOSE = "</atem:parameter>"
_MESSAGE_CONTENT_EXCLUDES = [_EOM, _EOT, _START, _MESSAGE]
_PARAMETER_VALUE_EXCLUDES = [
    _PARAMETER_CLOSE,
    _INVOKE_CLOSE,
    _FUNCTION_CALLS_OPEN,
    _FUNCTION_CALLS_CLOSE,
    "<atem:invoke",
    "<atem:parameter",
    _EOM,
    _EOT,
    _START,
    _MESSAGE,
]


def _muse_message(recipient: str, content: Format, end_token: str) -> TagFormat:
    """Build one recipient-routed Muse assistant message."""
    return TagFormat(
        begin=f"to={recipient}{_MESSAGE}",
        content=content,
        end=end_token,
    )


def _muse_free_text_message(recipient: str) -> TagFormat:
    return _muse_message(
        recipient,
        AnyTextFormat(excludes=_MESSAGE_CONTENT_EXCLUDES),
        _EOM,
    )


def _muse_user_message(content: Format) -> TagFormat:
    return _muse_message("user", content, _EOT)


def _muse_assistant_message_sequence(
    continuation_messages: list[TagFormat],
    terminal_messages: list[TagFormat],
) -> SequenceFormat:
    """Repeat continuation messages, then require a final answer or tool call."""
    return SequenceFormat(
        elements=[
            ConstStringFormat(value=" "),
            StarFormat(
                content=SequenceFormat(
                    elements=[
                        OrFormat(elements=continuation_messages),
                        ConstStringFormat(value=_ASSISTANT_MESSAGE_SEPARATOR),
                    ]
                )
            ),
            OrFormat(elements=terminal_messages),
        ]
    )


def muse_answer_structural_tag(answer: Format) -> StructuralTag:
    """Constrain the content of Muse's sole user-facing answer message."""
    return StructuralTag(
        format=SequenceFormat(
            elements=[
                ConstStringFormat(value=" "),
                OptionalFormat(
                    content=SequenceFormat(
                        elements=[
                            _muse_free_text_message("self"),
                            ConstStringFormat(value=_ASSISTANT_MESSAGE_SEPARATOR),
                        ]
                    )
                ),
                _muse_user_message(answer),
            ]
        )
    )


def _muse_parameter_element() -> SequenceFormat:
    """A parameter with a free-form name and value inside valid ATEM tags."""
    return SequenceFormat(
        elements=[
            ConstStringFormat(value='<atem:parameter name="'),
            RegexFormat(pattern=r'[^"<>]+'),
            ConstStringFormat(value='">'),
            AnyTextFormat(excludes=_PARAMETER_VALUE_EXCLUDES),
            ConstStringFormat(value=_PARAMETER_CLOSE),
        ]
    )


def _muse_tool_call_message(
    name: str,
    *,
    end_token: str,
) -> TagFormat:
    """Constrain one ATEM call and its routed recipient to the same name."""
    invoke = TagFormat(
        begin=f'<atem:invoke name="{name}">\n',
        content=StarFormat(
            content=SequenceFormat(
                elements=[
                    _muse_parameter_element(),
                    ConstStringFormat(value="\n"),
                ]
            )
        ),
        end=_INVOKE_CLOSE,
    )
    content = SequenceFormat(
        elements=[
            ConstStringFormat(value=f"{_FUNCTION_CALLS_OPEN}\n"),
            invoke,
            ConstStringFormat(value=f"\n{_FUNCTION_CALLS_CLOSE}"),
        ]
    )
    return _muse_message(
        name,
        content,
        end_token,
    )


def muse_tool_structural_tag(
    tools: list[FunctionToolParam],
    tool_choice: str,
) -> StructuralTag:
    """Build Muse's grammar for interleaved messages and tool calls."""
    tool_calls_eom = [
        _muse_tool_call_message(
            tool.function.name,
            end_token=_EOM,
        )
        for tool in tools
    ]
    tool_calls_eot = [
        _muse_tool_call_message(
            tool.function.name,
            end_token=_EOT,
        )
        for tool in tools
    ]

    continuation_messages = [
        _muse_free_text_message("self"),
        _muse_free_text_message("commentary"),
        *tool_calls_eom,
    ]
    terminal_messages = tool_calls_eot

    if tool_choice == "auto":
        terminal_messages.append(
            _muse_user_message(AnyTextFormat(excludes=_MESSAGE_CONTENT_EXCLUDES))
        )
    elif tool_choice not in ("required", "forced"):
        raise ValueError(f"Unsupported Muse tool choice: {tool_choice}")

    return StructuralTag(
        format=_muse_assistant_message_sequence(
            continuation_messages,
            terminal_messages,
        )
    )
