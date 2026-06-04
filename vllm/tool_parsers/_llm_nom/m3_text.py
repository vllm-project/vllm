# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from typing_extensions import Self

from .data_type import AtomDataType, FunctionCallParameterDataType
from .base import FunctionCallDict
from .generator import (
    GeneratorParser,
    default_tool_call_output_key,
    json_dumps,
)


class M3TextParser(GeneratorParser):
    """
    Parser for MiniMax M3 models.

    M3 uses a namespace token `]<]minimax[>[` as delimiter before each tag.
    Parameters use actual XML tag names (not `<parameter name="...">`), and can be nested.
    Complex arguments, as nested XML tags, are buffered and emitted as complete JSON;
    Simple arguments, are streamed character-by-character if possible.

    Example raw output::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="func1">]<]minimax[>[<p1>value1]<]minimax[>[</p1>]<]minimax[>[<p2>]<]minimax[>[<item>]<]minimax[>[<k>val]<]minimax[>[</k>]<]minimax[>[</item>]<]minimax[>[</p2>]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>
    """

    def __init__(
        self,
        *,
        with_reasoning: bool = True,
        reasoning_prefix: str = "<mm:think>",
        reasoning_suffix: str = "</mm:think>",
        functions: Optional[Dict] = None,
        tool_call_xml_tag_name: str = "tool_call",
        tool_call_namespace_token: str = "]<]minimax[>[",
        always_nullable: bool = True,
        reasoning_field: str = "reasoning",
        content_field: str = "content",
        tool_call_output_key: Callable[
            [int, Dict], Dict
        ] = default_tool_call_output_key,
    ):
        self.with_reasoning = with_reasoning
        self._reasoning_tokens: Optional[int] = None
        self._reasoning_prefix = reasoning_prefix
        self._reasoning_suffix = reasoning_suffix
        self._reasoning_suffix_without_newline = reasoning_suffix.lstrip()
        self._functions = functions
        self._tool_call_namespace_token = tool_call_namespace_token
        self._tool_call_start = (
            f"{self._tool_call_namespace_token}<{tool_call_xml_tag_name}>"
        )
        self._tool_call_end = (
            f"{self._tool_call_namespace_token}</{tool_call_xml_tag_name}>"
        )
        self._invoke_prefix = f'{self._tool_call_namespace_token}<invoke name="'
        self._invoke_suffix = '">'
        self._end_of_invoke = f"{self._tool_call_namespace_token}</invoke>"
        self._parameter_prefix = f"{self._tool_call_namespace_token}<"
        self._parameter_suffix = f"{self._tool_call_namespace_token}</"
        self._always_nullable = always_nullable
        self._reasoning_field = reasoning_field
        self._content_field = content_field
        self._tool_call_output_key = tool_call_output_key
        super().__init__()

    def _get_function(self, function_name: str) -> Optional[Dict]:
        if isinstance(self._functions, dict):
            return self._functions.get(function_name)

    def count_reasoning_tokens(self) -> Optional[int]:
        if self.with_reasoning:
            if self._reasoning_tokens is None:
                return self._count_consumed_tokens()
            else:
                return self._reasoning_tokens

    def _process(self) -> Generator[dict, None, None]:
        if self.with_reasoning:
            if self._reasoning_prefix:
                with_reasoning = yield from self._literal(
                    self._reasoning_prefix, should_raise=False
                )
                if with_reasoning:
                    yield from self._take_any(
                        until=self._reasoning_suffix, key=self._reasoning_field
                    )
                    self._reasoning_tokens = self._count_consumed_tokens()
                else:
                    self._reasoning_tokens = 0
                    # If reasoning is disabled, we may find the suffix at the beginning without the prefix.
                    yield from self._literal(
                        self._reasoning_suffix_without_newline,
                        should_raise=False,
                    )
            else:
                yield from self._take_any(
                    until=self._reasoning_suffix, key=self._reasoning_field
                )
                self._reasoning_tokens = self._count_consumed_tokens()

        if not self._functions:
            yield from self._take_any(key=self._content_field)
        else:
            yield from self._take_any(
                until=self._tool_call_start,
                key=self._content_field,
                should_consume_suffix=False,
            )
            yield from self._literal(self._tool_call_start)
            yield from self._literal("\n", should_raise=False)

            # NOTE: Only ONE `<tool_call>` block is supported by design.
            # Multiple parallel calls must share a single wrapper and use
            # multiple `<invoke>` tags inside it. A second `<tool_call>` after
            # the first `</tool_call>` will cause `update()` to raise
            # PatternMismatched (pattern exhausted).
            tool_call_index = 0
            while True:
                if tool_call_index:
                    yield from self._literal("\n", should_raise=False)
                tried = yield from self._literal(
                    (self._invoke_prefix, self._tool_call_end),
                    should_raise=False,
                )
                if tried is None or tried == self._tool_call_end:
                    break

                function_name = yield from self._take_any(until=self._invoke_suffix)
                self._append_delta(
                    self._tool_call_output_key(
                        tool_call_index, {"name": function_name, "arguments": "{"}
                    )
                )
                function = self._get_function(function_name)
                is_first_parameter = True
                while True:
                    tried = yield from self._literal(
                        (self._end_of_invoke, self._parameter_prefix),
                        should_raise=False,
                    )
                    if tried == self._end_of_invoke:
                        self._append_delta(
                            self._tool_call_output_key(
                                tool_call_index, {"arguments": "}"}
                            )
                        )
                        break
                    if tried is None:
                        # Consume and ignore, hoping it may recover after this invoke ends.
                        yield from self._take_any(until=self._end_of_invoke)
                        self._append_delta(
                            self._tool_call_output_key(
                                tool_call_index, {"arguments": "}"}
                            )
                        )
                        break
                    parameter_name = yield from self._take_any(until=">")
                    parameter_name_to_arguments = "{}{}: ".format(
                        "" if is_first_parameter else ", ",
                        json_dumps(parameter_name),
                    )
                    self._append_delta(
                        self._tool_call_output_key(
                            tool_call_index,
                            {"arguments": parameter_name_to_arguments},
                        )
                    )
                    parameter_suffix = "{}{}>".format(
                        self._parameter_suffix, parameter_name
                    )
                    parameter_data_type = (
                        FunctionCallParameterDataType.get_schema_of_parameter(
                            function, parameter_name
                        )
                    )
                    tried = yield from self._literal(
                        self._parameter_prefix,
                        should_raise=False,
                        should_consume=False,
                    )
                    if tried is not None:
                        param_body_str = yield from self._take_any(
                            until=parameter_suffix
                        )
                        if param_body_str:
                            # nested XML -> object
                            # NOTE: The namespace token has the highest semantic
                            # priority. Once the model emits `]<]minimax[>[<` here,
                            # we MUST treat the body as nested XML, even if the
                            # schema says this parameter should be a primitive.
                            # The model is asserting "this is a JSON level
                            # transition" — schema mismatches are reported back via
                            # the agent loop, not silently rewritten here.
                            param = self._parse_parameter(
                                param_body_str, parameter_data_type
                            )
                        else:
                            param = parameter_data_type.convert("")
                        self._append_delta(
                            self._tool_call_output_key(
                                tool_call_index,
                                {"arguments": json_dumps(param)},
                            )
                        )
                    else:
                        # no more nested XML -> string / number / boolean
                        yield from self._take_data_type_as_json(
                            until=parameter_suffix,
                            key=lambda value: self._tool_call_output_key(
                                tool_call_index, {"arguments": value}
                            ),
                            data_type=parameter_data_type,
                            always_nullable=False,
                            should_consume_suffix=True,
                        )
                    is_first_parameter = False
                tool_call_index += 1

    def _parse_parameter(
        self, body: str, parameter_data_type: FunctionCallParameterDataType
    ) -> dict:
        chunks = body.split(self._tool_call_namespace_token)
        # NOTE: Array detection is intentionally a strict "schema says array
        # AND first child is <item>" check. We do NOT promote a uniform
        # `<x><x>...` body to an array on schema mismatch — leave it as
        # `{"x": [...]}` so the agent loop can spot and correct the model.
        if (
            AtomDataType.array in parameter_data_type.candidates
            and len(chunks) > 1
            and chunks[1].startswith("<item>")
        ):
            root = []
        else:
            root = {}
        stack: List[_StackItem] = [
            _StackItem(tag=None, value=root, texts=None, data_type=parameter_data_type)
        ]

        # Ignore the first chunk inside the parameter.
        # It should be empty, since we've tried `self._parameter_prefix` and failed before entering this function.
        for chunk_index in range(1, len(chunks)):
            chunk = chunks[chunk_index]
            # There are 7 = 3 + 3 + 1 non-empty categories of chunks.
            if chunk.startswith("</"):
                gt_offset = chunk.find(">", 2)
                if gt_offset == -1:
                    # 1. `</tag`
                    tag = chunk[2:]
                    value = None
                elif gt_offset == len(chunk) - 1:
                    # 2. `</tag>`
                    tag = chunk[2:-1]
                    value = None
                else:
                    # 3. `</tag>value`
                    tag = chunk[2:gt_offset]
                    value = chunk[gt_offset + 1 :]
                while len(stack) > 1:
                    item = stack.pop()
                    stack[-1].append(item)
                    if item.tag == tag:
                        break
                if value:
                    stack[-1].append_text(value)
            elif chunk.startswith("<"):
                gt_offset = chunk.find(">", 1)
                if gt_offset == -1:
                    # 4. `<tag`
                    tag = chunk[1:]
                    value = None
                elif gt_offset == len(chunk) - 1:
                    # 5. `<tag>`
                    tag = chunk[1:-1]
                    value = None
                else:
                    # 6. `<tag>value`
                    tag = chunk[1:gt_offset]
                    value = chunk[gt_offset + 1 :]
                sub_data_type = stack[-1].get_data_type_of_property(tag)
                if (
                    sub_data_type
                    and AtomDataType.array in sub_data_type.candidates
                    and len(chunks) > chunk_index + 1
                    and chunks[chunk_index + 1].startswith("<item>")
                ):
                    sub = []
                elif sub_data_type and AtomDataType.object in sub_data_type.candidates:
                    sub = {}
                else:
                    sub = None
                stack.append(
                    _StackItem(
                        tag=tag,
                        value=sub,
                        texts=[value] if value else None,
                        data_type=sub_data_type,
                    )
                )
            elif chunk:
                # 7. `value`
                stack[-1].append_text(chunk)

        while len(stack) > 1:
            item = stack.pop()
            stack[-1].append(item)

        return stack[0].get_value()

    def stringify_function_calls(self, function_calls: List[FunctionCallDict]) -> str:
        if not function_calls:
            return ""
        parts = [self._tool_call_start + "\n"]
        for function_call in function_calls:
            parts.append(
                self._invoke_prefix + function_call["name"] + self._invoke_suffix
            )
            arguments = function_call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            parts.append(self._stringify_parameter(arguments))
            parts.append(self._end_of_invoke + "\n")
        parts.append(self._tool_call_end)
        return "".join(parts)

    def _stringify_parameter(self, parameter: Any) -> str:
        # NOTE: null values are simply ignored.
        # This is limited due to the training philosophy of MiniMax M3.
        # In the training process, the model will NOT see any null values in tool calls.
        # Thus, we have to skip them here, in order to avoid OOD.
        # This cost is unwillingly accepted:
        #     `["a", null, "c"]` becomes `<items>a</items><items>c</items>`,
        #     even the size of the array is changed.
        if isinstance(parameter, dict):
            return "".join(
                f"{self._tool_call_namespace_token}<{key}>{self._stringify_parameter(value)}{self._tool_call_namespace_token}</{key}>"
                for key, value in parameter.items()
                if value is not None
            )
        elif isinstance(parameter, list):
            return "".join(
                f"{self._tool_call_namespace_token}<item>{self._stringify_parameter(value)}{self._tool_call_namespace_token}</item>"
                for value in parameter
                if value is not None
            )
        elif isinstance(parameter, str):
            return parameter
        elif parameter is None:
            # should be unreachable
            return ""
        else:
            return json.dumps(parameter, ensure_ascii=False)


@dataclass(slots=True)
class _StackItem:
    tag: Optional[str]
    value: Optional[Union[Dict, List]]
    texts: Optional[List[str]]
    data_type: Optional[FunctionCallParameterDataType]

    def get_value(self) -> Any:
        if self.value is None:
            if self.texts:
                value = "".join(self.texts)
            else:
                value = ""
            if self.data_type:
                value = self.data_type.convert(value)
            return value
        elif self.texts and isinstance(self.value, dict):
            extra_text_key = "$text"
            while extra_text_key in self.value:
                extra_text_key = "$" + extra_text_key
            self.value[extra_text_key] = "".join(self.texts)
            return self.value
        else:
            return self.value

    def append(self, item: Self) -> None:
        if self.value is None:
            self.value = {item.tag: item.get_value()}
        elif isinstance(self.value, dict):
            if item.tag in self.value:
                # NOTE: Duplicate tag inside an object is collapsed into a
                # list to preserve all values, even if the schema declares
                # the key as a singleton. We don't drop the data and we
                # don't try to "fix" the schema mismatch silently — the agent
                # loop should surface the inconsistency back to the model.
                value = self.value[item.tag]
                if isinstance(value, list):
                    value.append(item.get_value())
                else:
                    self.value[item.tag] = [value, item.get_value()]
            else:
                self.value[item.tag] = item.get_value()
        elif isinstance(self.value, list):
            # We expect `item.tag` to be `"item"`, but if it's not, we should still accept it.
            self.value.append(item.get_value())
        else:
            raise NotImplementedError()

    def append_text(self, value: str) -> None:
        if isinstance(self.value, list):
            if self.data_type:
                data_type = self.data_type.get_data_type_of_item(index=len(self.value))
                if data_type:
                    value = data_type.convert(value)
            self.value.append(value)
        elif self.texts is None:
            self.texts = [value]
        else:
            self.texts.append(value)

    def get_data_type_of_property(
        self, tag: str
    ) -> Optional[FunctionCallParameterDataType]:
        if self.data_type:
            if isinstance(self.value, list):
                # We expect `tag` to be `"item"`, but if it's not, we should still accept it.
                return self.data_type.get_data_type_of_item(index=len(self.value))
            elif isinstance(self.value, dict):
                return self.data_type.get_data_type_of_property(tag)
            else:
                return None
