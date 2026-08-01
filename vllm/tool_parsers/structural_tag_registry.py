# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.tool import Tool as ResponsesTool
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from openai.types.responses.tool_choice_function import ToolChoiceFunction
from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)

ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ResponsesToolChoice
    | None
)
AllowedToolRef: TypeAlias = dict[str, object]
SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
StructuralTagBuilder: TypeAlias = Callable[
    [
        list[FunctionToolParam],
        list[BuiltinToolParam],
        SimplifiedToolChoice,
        bool,
    ],
    StructuralTag,
]

# Keep this list in sync with xgrammar.builtin_structural_tag. It is used for
# vLLM-side validation and for documenting the xgrammar builtin surface that
# can be requested by tool parsers through ``structural_tag_model``.
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset(
    {
        "llama",
        "kimi",
        "deepseek_r1",
        "deepseek_v3_1",
        "qwen_3_5",
        "qwen_3_coder",
        "qwen_3",
        "deepseek_v3_2",
        "glm_4_7",
        "deepseek_v4",
    }
)
VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes", "kimi_k3"})
SUPPORTED_STRUCTURAL_TAG_MODELS = (
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
)

_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


def register_vllm_structural_tag(model: str):
    """Register a vLLM-owned structural tag builder."""

    def decorator(func: StructuralTagBuilder) -> StructuralTagBuilder:
        _VLLM_STRUCTURAL_TAG_REGISTRY[model] = func
        return func

    return decorator


def _any_tool_strict(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
) -> bool:
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.strict is True:
            return True
        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False


def get_model_structural_tag(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""

    if not tools or tool_choice == "none":
        return None

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None

    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

    if model in _VLLM_STRUCTURAL_TAG_REGISTRY:
        function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
            dumped_tools,
            dumped_tool_choice,
        )
        return _VLLM_STRUCTURAL_TAG_REGISTRY[model](
            function_tools,
            builtin_tools,
            simplified_tool_choice,
            reasoning,
        )

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        supported = sorted(SUPPORTED_STRUCTURAL_TAG_MODELS)
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    structural_tag = get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
    )
    if model == "deepseek_v4" and not reasoning:
        _allow_deepseek_v4_chat_think_end(structural_tag)
    return structural_tag


def _allow_deepseek_v4_chat_think_end(structural_tag: StructuralTag) -> None:
    """Allow DSv4 chat-mode outputs to emit an extra ``</think>``."""
    fmt = structural_tag.format
    if getattr(fmt, "type", None) != "triggered_tags":
        return
    excludes = getattr(fmt, "excludes", None)
    if not isinstance(excludes, list):
        return
    fmt.excludes = [exclude for exclude in excludes if exclude != "</think>"]


def _dump_tool_for_xgrammar(
    tool: ChatCompletionToolsParam | ResponsesTool,
) -> dict[str, Any]:
    """Convert tool objects to xgrammar's Chat Completions tool protocol."""

    if isinstance(tool, FunctionTool):
        function: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            function["description"] = tool.description
        if tool.parameters is not None:
            function["parameters"] = tool.parameters
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}
    dumped_tool = tool.model_dump(mode="json", exclude_none=True)
    if isinstance(tool, ChatCompletionToolsParam):
        return dumped_tool
    return dict(dumped_tool)


def _dump_tool_choice_for_xgrammar(
    tool_choice: ToolChoice,
) -> dict[str, Any] | str | None:
    """Convert tool_choice objects to xgrammar's expected protocol."""

    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)

    if isinstance(tool_choice, ToolChoiceFunction):
        return {
            "type": "function",
            "function": {"name": tool_choice.name},
        }

    if isinstance(tool_choice, ToolChoiceAllowed):
        return {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": tool_choice.mode,
                "tools": [
                    _dump_allowed_tool_ref_for_xgrammar(tool)
                    for tool in tool_choice.tools
                ],
            },
        }

    return tool_choice.model_dump(mode="json", exclude_none=True)


def _dump_allowed_tool_ref_for_xgrammar(tool_ref: AllowedToolRef) -> AllowedToolRef:
    if (
        tool_ref.get("type") == "function"
        and "function" not in tool_ref
        and "name" in tool_ref
    ):
        return {
            "type": "function",
            "function": {"name": tool_ref["name"]},
        }
    return tool_ref


def get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


def _hermes_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    arguments_field_prefix = '", "arguments": '
    formats = [
        # <tool_call>
        # {"name": "t1", "arguments": {"q": "v"}}
        # </tool_call>
        ('<tool_call>\n{"name": "', "}\n</tool_call>"),
        # <tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>
        ('<tool_call>{"name": "', "}</tool_call>"),
    ]

    return [
        TagFormat(
            begin=begin + tool.function.name + arguments_field_prefix,
            content=JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function)
            ),
            end=end,
        )
        for tool in tools
        for begin, end in formats
    ]


@register_vllm_structural_tag("hermes")
def get_hermes_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    tool_call_trigger = "<tool_call>"

    if tool_choice == "auto":
        tags = _hermes_tool_tags(tools)
        suffix_tag = (
            TriggeredTagsFormat(triggers=[tool_call_trigger], tags=tags)
            if tags
            else AnyTextFormat()
        )
    elif tool_choice == "forced":
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
            stop_after_first=True,
        )
    else:
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
        )

    return StructuralTag(format=suffix_tag)


def _minimax_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=f'<invoke name="{tool.function.name}">\n',
            content=JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function),
                style="minimax_xml",
            ),
            end="</invoke>\n",
        )
        for tool in tools
    ]


@register_vllm_structural_tag("minimax")
def get_minimax_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    tool_call_begin = "<minimax:tool_call>\n"
    tool_call_end = "</minimax:tool_call>"
    tool_call_trigger = "<minimax:tool_call>"

    tags = _minimax_tool_tags(tools)

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=[tool_call_trigger],
                tags=[
                    TagFormat(
                        begin=tool_call_begin,
                        content=TagsWithSeparatorFormat(
                            tags=tags,
                            separator="",
                            at_least_one=True,
                        ),
                        end=tool_call_end,
                    )
                ],
                excludes=["<think>", "</think>"],
            )
            if tags
            else AnyTextFormat(excludes=["<think>", "</think>"])
        )
    elif tool_choice == "forced":
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )
    else:
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )

    return StructuralTag(format=suffix_tag)


# ---------------------------------------------------------------------------
# Kimi K3 (XTML channel format)
# ---------------------------------------------------------------------------
# K3 assistant output after the reasoning gate (``<|close|>think<|sep|>``):
#   <|open|>response<|sep|> <content> <|close|>response<|sep|>
#   [ <|open|>tools<|sep|>
#       <|open|>call tool="NAME" index="1"<|sep|>
#         <|open|>argument key="K" type="TYPE"<|sep|>VALUE<|close|>argument<|sep|>
#       <|close|>call<|sep|> ...
#     <|close|>tools<|sep|> ]
# See ``encoding_k3.py`` (_render_assistant_segments / _open_tag / _attr) and
# ``kimi_k3_tool_parser.py`` for the exact byte-level encoding this mirrors.
_K3_OPEN = "<|open|>"
_K3_CLOSE = "<|close|>"
_K3_SEP = "<|sep|>"
_K3_RESPONSE_OPEN = f"{_K3_OPEN}response{_K3_SEP}"
_K3_RESPONSE_CLOSE = f"{_K3_CLOSE}response{_K3_SEP}"
_K3_TOOLS_OPEN = f"{_K3_OPEN}tools{_K3_SEP}"
_K3_TOOLS_CLOSE = f"{_K3_CLOSE}tools{_K3_SEP}"
_K3_CALL_CLOSE = f"{_K3_CLOSE}call{_K3_SEP}"
_K3_ARG_CLOSE = f"{_K3_CLOSE}argument{_K3_SEP}"
# The model closes the assistant turn with <|close|>message<|sep|> right before
# the end-of-message token. It is generated (not part of the prompt prefix), so
# the tag must permit it or the FSM would mask the model's natural terminator.
_K3_MESSAGE_CLOSE = f"{_K3_CLOSE}message{_K3_SEP}"

# JSON-schema type -> K3 XTML ``type=`` attribute value. Mirrors
# ``encoding_k3._xtml_type`` (integer collapses onto number).
_K3_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}


def _k3_escape_attr(value: str) -> str:
    """Mirror ``encoding_k3._escape_attr_value`` (``&`` then ``"``)."""
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


_K3_STRING_ATOM = r"(?:[^<]|<[^|])"
"""One raw-string character: anything but the ambiguous "<|" marker prefix.
Allows '<' inside values (e.g. HTML snippets); a value *ending* in '<' or
containing a literal "<|" is not expressible and falls back to AnyText via
the pattern checks below never matching those cases at build time (schemas
cannot know values, so the only build-time effect is the length bound)."""


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    """Length/pattern constraint for the raw string channel, if expressible.

    The XTML string channel emits values raw (not JSON-quoted), so
    JSONSchemaFormat cannot enforce string constraints there; unconstrained
    AnyText lets maxLength/pattern violations through (observed on the walle
    verifier: over-long junk strings pass the grammar and fail validation).
    xgrammar's regex engine has no lookahead, so the close marker is kept
    unambiguous by excluding the "<|" prefix from value characters.

    Returns a regex for the value, or None to keep permissive AnyText.
    """
    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if not isinstance(max_len, int) or max_len < 0 or max_len > 4096:
        return None
    if not isinstance(min_len, int) or min_len < 0 or min_len > max_len:
        min_len = 0
    return _K3_STRING_ATOM + f"{{{min_len},{max_len}}}"


def _k3_argument_tag(
    key: str,
    schema: dict[str, Any],
    root_defs: dict[str, Any] | None = None,
) -> TagFormat | None:
    """Build one ``argument`` XTML tag for property ``key``.

    ``string`` values are emitted raw (bounded by the close marker); every other
    JSON type is emitted as JSON and validated against the property schema. A
    property whose type is a union / missing is left permissive (any XTML type,
    raw value) so a valid call is never rejected.

    ``root_defs`` carries the tool parameters' root-level ``$defs`` /
    ``definitions``: slicing a property out of the parameters document orphans
    its ``#/$defs/...`` references, so those tables must be re-attached to keep
    the embedded schema self-contained.
    """
    prop = schema if isinstance(schema, dict) else {}
    json_type = prop.get("type")
    xtml_type = (
        _K3_JSON_TO_XTML_TYPE.get(json_type) if isinstance(json_type, str) else None
    )
    if xtml_type is None:
        # Unknown / union type: constrain the key but keep the value permissive.
        return None
    begin = (
        f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" type="{xtml_type}"{_K3_SEP}'
    )
    if xtml_type == "string":
        # Raw string channel: JSONSchemaFormat can't apply (values are not
        # JSON-quoted), but an enum/const of strings is a finite set that can
        # be enforced exactly with const-string alternation. Enum semantics
        # are exclusive, so this never over-rejects. Fall back to permissive
        # AnyText for open-ended strings or non-representable enums.
        enum_values = prop.get("enum")
        if enum_values is None and isinstance(prop.get("const"), str):
            enum_values = [prop["const"]]
        if (
            isinstance(enum_values, list)
            and enum_values
            and len(enum_values) <= 256
            and all(isinstance(v, str) for v in enum_values)
            and not any("<|" in v for v in enum_values)
        ):
            branches = [ConstStringFormat(value=v) for v in enum_values]
            content: Any = (
                branches[0] if len(branches) == 1 else OrFormat(elements=branches)
            )
        elif (bounded := _k3_bounded_string_regex(prop)) is not None:
            content = RegexFormat(pattern=bounded)
        else:
            content = AnyTextFormat(excludes=[_K3_CLOSE])
    else:
        embedded = prop
        if root_defs:
            embedded = dict(prop)
            for defs_key, defs_value in root_defs.items():
                embedded.setdefault(defs_key, defs_value)
        content = JSONSchemaFormat(json_schema=embedded)
    return TagFormat(begin=begin, content=content, end=_K3_ARG_CLOSE)


def _k3_permissive_argument_tag() -> TagFormat:
    """A key/type-agnostic ``argument`` tag: any attributes, raw value.

    Used as a fallback so tools with union/loose schemas still get the XTML
    skeleton constrained without over-rejecting the value.
    """
    return TagFormat(
        begin=_K3_OPEN + "argument ",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^<]*" + _K3_SEP.replace("|", r"\|")),
                AnyTextFormat(excludes=[_K3_CLOSE]),
            ]
        ),
        end=_K3_ARG_CLOSE,
    )


def _k3_arguments_block(parameters: dict[str, Any] | bool) -> Any:
    """Build ``argument`` tags for a tool's parameter schema.

    Require at least one tag when the root schema declares required properties.
    Otherwise, keep accepting zero-or-more tags. Arguments remain order-agnostic
    and non-unique.
    """
    if not isinstance(parameters, dict):
        return StarFormat(content=_k3_permissive_argument_tag())
    props = parameters.get("properties")
    if not isinstance(props, dict) or not props:
        # No declared properties: allow any argument blocks (or none).
        return StarFormat(content=_k3_permissive_argument_tag())
    root_defs = {
        defs_key: parameters[defs_key]
        for defs_key in ("$defs", "definitions")
        if isinstance(parameters.get(defs_key), dict)
    }
    tags: list[TagFormat] = []
    for key, prop in props.items():
        tag = _k3_argument_tag(key, prop, root_defs)
        tags.append(tag if tag is not None else _k3_permissive_argument_tag())
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=list(tags))
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _k3_call_tag(tool: FunctionToolParam) -> TagFormat:
    """One ``call`` tag: ``<|open|>call tool="N" index="<digits>"<|sep|> args``."""
    function = tool.function
    parameters = get_function_parameters(function)
    begin = f'{_K3_OPEN}call tool="{_k3_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[0-9]+"),
                ConstStringFormat(value=f'"{_K3_SEP}'),
                _k3_arguments_block(parameters),
            ]
        ),
        end=_K3_CALL_CLOSE,
    )


def _k3_response_prefix() -> list[Any]:
    """The response channel that always precedes the tools channel.

    ``response`` is generated in thinking mode (prefix ends at
    ``<|open|>think<|sep|>``) but is part of the generation prefix in
    non-thinking mode, so its open marker is optional. The body is bounded by
    the response close marker.
    """
    return [
        OptionalFormat(content=ConstStringFormat(value=_K3_RESPONSE_OPEN)),
        TagFormat(begin="", content=AnyTextFormat(), end=_K3_RESPONSE_CLOSE),
    ]


def _k3_tools_channel(tools: list[FunctionToolParam]) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[_k3_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


@register_vllm_structural_tag("kimi_k3")
def get_kimi_k3_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    trailer = OptionalFormat(content=ConstStringFormat(value=_K3_MESSAGE_CLOSE))

    if not tools:
        return StructuralTag(
            format=SequenceFormat(elements=[*_k3_response_prefix(), trailer])
        )

    if tool_choice == "auto":
        tools_part: Any = OptionalFormat(content=_k3_tools_channel(tools))
    elif tool_choice == "forced":
        # K3 rejects named tool choice upstream; treat defensively as a single
        # mandatory call of the first tool.
        tools_part = _k3_tools_channel(tools[:1])
    else:  # required
        tools_part = _k3_tools_channel(tools)

    return StructuralTag(
        format=SequenceFormat(elements=[*_k3_response_prefix(), tools_part, trailer])
    )
