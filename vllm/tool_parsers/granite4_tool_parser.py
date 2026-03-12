# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import io
import json
import operator
from ast import literal_eval
from collections.abc import Sequence
from typing import Any

import regex as re
from lark import Lark, Transformer, UnexpectedCharacters
from lark.lexer import Lexer, Token

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)

hermes_grammar = r"""

    output: (text+|tagged_tool_call)+
    tagged_tool_call: TC_START tool_call TC_END
    tool_call: _LBRACE tool_name _COMMA tool_arguments _RBRACE
             | _LBRACE tool_arguments _COMMA tool_name _RBRACE
    tool_name: NAME _COLON string
    tool_arguments: ARGUMENTS _COLON value
    text: TEXT -> text

    ?value: dict
          | list
          | string
          | FLOAT             -> number
          | TRUE              -> true
          | FALSE             -> false
          | NULL              -> null

    list : _LBRACK [value (_COMMA value)*] _RBRACK
    dict : _LBRACE [pair (_COMMA pair)*] _RBRACE
    pair : string _COLON value
    string: STRING -> string

    %declare TC_START TC_END NAME ARGUMENTS STRING FLOAT TRUE FALSE NULL
    %declare _LBRACE _RBRACE _LBRACK _RBRACK _COMMA _COLON TEXT SKIP
    %ignore SKIP
"""


class HermesLexer(Lexer):
    def tc_start(self):
        return r"<tool_call>"

    def tc_end(self):
        return r"</tool_call>"

    PATTERNS = [
        ("STRING", r'"(?:[^"\\]|\\.)*"'),
        ("FLOAT", r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?(?=[ \t\n\r\}\],]+)"),
        ("TRUE", r"true"),
        ("FALSE", r"false"),
        ("NULL", r"null"),
        ("_LBRACE", r"\{"),
        ("_RBRACE", r"\}"),
        ("_LBRACK", r"\["),
        ("_RBRACK", r"\]"),
        ("_COMMA", r","),
        ("_COLON", r":"),
        ("WS", r"[ \t\n\r]+"),
    ]

    def __init__(self, lexer_conf):
        delimiters = [("TC_START", self.tc_start()), ("TC_END", self.tc_end())]
        pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in delimiters + self.PATTERNS
        )
        self.regex = re.compile(pattern)
        self.start_regex = re.compile(f"(?P<TC_START>{self.tc_start()})")
        self.buffer = ""
        self.done = False

    def wait_for_content(self, stream):
        try:
            while True:
                self.buffer = stream.send(self.buffer)
                if not self.buffer:
                    yield Token("SKIP", "")
                else:
                    return False
        except StopIteration:
            return True

    def consume_buffer(self, length: int | None = None) -> str:
        length = len(self.buffer) if length is None else length
        text = self.buffer[:length]
        self.buffer = self.buffer[length:]
        return text

    def consume_text(self, stream):
        while True:
            self.done = yield from self.wait_for_content(stream)
            if self.done:
                return
            match = self.start_regex.search(self.buffer, partial=True)
            if match is not None and match.end() > match.start():
                text = self.consume_buffer(match.start())
                if text:
                    yield Token("TEXT", text)
                if match.partial:
                    if not text:  # No need to send another token
                        yield Token("SKIP", "")
                else:
                    return
            else:
                yield Token("TEXT", self.consume_buffer())

    def consume_tool_call(self, stream):
        self.dict_dept = 0
        while True:
            self.done = yield from self.wait_for_content(stream)
            if self.done:
                return
            while self.buffer:
                if (match := self.regex.match(self.buffer, partial=True)) is not None:
                    token_type = match.lastgroup
                    if match.partial:
                        yield Token("SKIP", "")  # Possible partial match
                        break
                    if token_type == "WS":  # skip whitespace
                        self.consume_buffer(match.end())
                        continue
                    value = match.group() if token_type in ["STRING", "FLOAT"] else ""
                    if token_type == "_LBRACE":
                        self.dict_dept += 1
                    elif token_type == "_RBRACE":
                        self.dict_dept -= 1
                    elif token_type == "STRING":
                        if value == '"name"' and self.dict_dept == 1:
                            token_type = "NAME"
                            value = ""
                        elif value == '"arguments"' and self.dict_dept == 1:
                            token_type = "ARGUMENTS"
                            value = ""
                    self.consume_buffer(match.end())
                    yield Token(token_type, value)
                    if token_type == "TC_END":
                        return
                else:
                    raise UnexpectedCharacters(self.buffer, 0, 0, 0)

    def lex(self, stream):
        self.buffer = ""
        self.done = False
        while not self.done:
            yield from self.consume_text(stream)
            yield from self.consume_tool_call(stream)


class HermesTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.parsed_tool_names = list[str]()
        self.parsed_tool_calls = list[dict[str, dict[str, Any]]]()

    def tool_name(self, items):
        self.parsed_tool_names.append(items[1])
        return items[1]

    def tool_arguments(self, items):
        return items[1]

    def tagged_tool_call(self, items):
        return items[1]

    def tool_call(self, items):
        tc = {"name": items[0], "arguments": items[1]}
        self.parsed_tool_calls.append(tc)
        self.validate()
        return tc

    def text(self, items):
        chunk = str(items[0])
        return chunk

    def dict(self, items):
        if items == [None]:
            return {}
        return dict(items)

    def list(self, items):
        if items == [None]:
            return []
        return list(items)

    pair = tuple
    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False

    def number(self, items):
        try:
            return int(items[0])
        except ValueError:
            return float(items[0])

    def string(self, items):
        return literal_eval(str(items[0]))

    def validate(self, comp=operator.le):
        seen_tools = len(self.parsed_tool_calls)
        seen_tool_names = len(self.parsed_tool_names)
        assert comp(seen_tools, seen_tool_names)
        assert 0 <= (seen_tool_names - seen_tools) <= 1
        for name, tc in zip(self.parsed_tool_names, self.parsed_tool_calls):
            assert name == tc["name"]

    def finish(self):
        self.validate(operator.eq)


# Avoid compiling the grammar every time. It's about an order of magnitude faster
# Loading from a warm cache takes 0.3 ms on my machine
@functools.cache
def _make_parser_proto(lexer_class=HermesLexer):
    new_hermes_parser = Lark(
        hermes_grammar,
        start="output",
        lexer=lexer_class,
        parser="lalr",
        transformer=HermesTransformer(),
    )
    binary_buffer = io.BytesIO()
    new_hermes_parser.save(binary_buffer)
    binary_buffer.seek(0)
    return binary_buffer


def make_parser(lexer_class=HermesLexer):
    serialized = _make_parser_proto(lexer_class)
    serialized.seek(0)
    return Lark.load(serialized)


class HermesToolCallParser:
    def __init__(
        self,
        tool_name_callback,
        tool_call_callback,
        text_callback,
        lexer_class=HermesLexer,
    ):
        self.parser = make_parser(lexer_class)

        self.tool_name_callback = tool_name_callback
        self.tool_call_callback = tool_call_callback
        self.text_callback = text_callback
        self.tool_names_found = 0
        self.tool_found = 0
        self.text_ptr = 0
        self.finished = False

        self.food = ""

        def generator():
            yield ""
            while not self.finished:
                food = self.food
                self.food = ""
                push_back = yield food
                self.food = push_back + self.food

        gen = generator()
        next(gen)

        self.interactive = self.parser.parse_interactive(gen)
        self.transformer = self.parser.parser.options.transformer
        self.token_stream = self.interactive.lexer_thread.lex(
            self.interactive.parser_state
        )
        lexer = lexer_class(None)
        self.start_token = lexer.tc_start()
        self.end_token = lexer.tc_end()

    def _invoke_callbacks(self):
        if len(self.transformer.parsed_tool_names) > self.tool_names_found:
            self.tool_name_callback(self.transformer.parsed_tool_names[-1])
            self.tool_names_found += 1
        if len(self.transformer.parsed_tool_calls) > self.tool_found:
            self.tool_call_callback(self.transformer.parsed_tool_calls[-1])
            self.tool_found += 1

    def feed(self, chunk: str):
        self.food += chunk
        while True:
            token = next(self.token_stream)
            if token.type == "TEXT":
                self.text_callback(str(token))
            if token.type == "SKIP":  # waiting for more input
                break
            self.interactive.result = self.interactive.feed_token(token)
            self._invoke_callbacks()

    def finish(self) -> list[tuple[str, Any]]:
        self.finished = True
        result = self.interactive.feed_eof()
        self._invoke_callbacks()
        return result


def dump_args(args: None | dict[str, Any] | str) -> str | None:
    if args is None or isinstance(args, str):
        return args
    else:
        return json.dumps(args, ensure_ascii=False)


class Granite4ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool = list[str]()
        self.streamed_tool_names = list[str]()
        self.streamed_text_chunks = list[str]()

        self.tool_names = list[str]()
        self.tool_calls = list[dict[str, dict[str, Any]]]()
        self.parsed_text_chunks = list[str]()

        self.parser = HermesToolCallParser(
            lambda x: self.tool_names.append(x),
            lambda x: self.tool_calls.append(x),
            lambda x: self.parsed_text_chunks.append(x),
        )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        msg = ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )
        try:
            self.parser.feed(model_output)
            self.parser.finish()
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        # function call args are JSON but as a string
                        arguments=dump_args(tc["arguments"]),
                    ),
                )
                for tc in self.tool_calls
            ]
            msg.tools_called = bool(tool_calls)
            msg.tool_calls = tool_calls
            msg.content = "".join(self.parsed_text_chunks) or None
        except Exception:
            logger.exception("Error in extracting tool call from response.")
        return msg

    def collect_unstreamed_text(self) -> str:
        n_streamed_chunks = len(self.streamed_text_chunks)
        n_parsed_chunks = len(self.parsed_text_chunks)
        unstreamed = []

        if n_streamed_chunks < n_parsed_chunks:
            n_unstreamed = n_parsed_chunks - n_streamed_chunks
            unstreamed = self.parsed_text_chunks[-n_unstreamed:]
            self.streamed_text_chunks.extend(unstreamed)
        ret = "".join(unstreamed)
        return ret

    def collect_tool_calls(self) -> list[DeltaToolCall]:
        tool_calls = list[DeltaToolCall]()

        n_parsed_names = len(self.tool_names)
        n_streamed_names = len(self.streamed_tool_names)
        n_streamed_args = len(self.streamed_args_for_tool)

        if n_streamed_names == n_streamed_args:
            if n_parsed_names > n_streamed_names:
                self.current_tool_id += 1
            else:
                return tool_calls
        else:
            assert n_streamed_names == n_streamed_args + 1

        for i in range(self.current_tool_id, n_parsed_names):
            self.current_tool_id = i
            function = DeltaFunctionCall()

            if i == len(self.streamed_tool_names):
                function.name = self.tool_names[i]
                self.streamed_tool_names.append(function.name)

            if i < len(self.tool_calls):
                tc = self.tool_calls[i]
                self.prev_tool_call_arr.append(tc)
                function.arguments = dump_args(tc.get("arguments"))
                self.streamed_args_for_tool.append(function.arguments or "")

            delta = DeltaToolCall(
                index=self.current_tool_id,
                function=function.model_dump(exclude_none=True),
            )
            if function.name:
                delta.type = "function"
                delta.id = make_tool_call_id()
            if function.name or function.arguments:
                tool_calls.append(delta)
        return tool_calls

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        try:
            msg = DeltaMessage()

            self.parser.feed(delta_text)
            msg.tool_calls = self.collect_tool_calls()
            msg.content = self.collect_unstreamed_text()

            if msg.content or msg.tool_calls:
                return msg
            else:
                return None  # nothing to stream

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
