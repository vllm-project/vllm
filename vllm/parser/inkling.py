# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling parser: typed content blocks parsed by a single state machine.

Inkling output format (every marker is a dedicated special token)::

    <|message_model|><|content_thinking|>...reasoning...<|end_message|>
    <|message_model|><|content_text|>...visible text...<|end_message|>
    <|message_model|><|content_invoke_tool_json|>
        {"name":"get_weather","args":{"city":"SF"}}<|end_message|>

Blocks are self-describing and may repeat in any order; sampling may
also end a block with the standalone ``<|content_model_end_sampling|>``
token. The tool-call payload is a single JSON object whose ``name`` is
extracted by the engine's name-from-args path and whose ``args`` object
is carved out of the wrapper by :func:`_inkling_arg_converter`.

Note the terminal *labels*: ``THINK_START``/``THINK_END`` are what the
engine keys its reasoning plumbing on (``is_reasoning_end``,
``count_reasoning_tokens``, initial-state seeding), so ``<|end_message|>``
is labelled ``THINK_END`` here even though it ends every block kind —
the transition table, not the label, carries the semantics.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
END_MESSAGE = "<|end_message|>"

INKLING_SPECIAL_TOKENS = (
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_TOOL_ERROR,
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
)


def _inkling_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the ``args`` object out of the tool-call JSON wrapper.

    Why a converter at all: the engine's ``tool_args_json`` machinery
    treats the *entire* TOOL_ARGS text as the tool arguments, but Inkling's
    payload is the ``{"name":...,"args":{...}}`` wrapper — without a
    converter, ``_compute_arg_delta`` streams the wrapper verbatim into
    the OpenAI ``arguments`` field (``converter is None -> raw delta``,
    unconditionally; ``stream_arg_deltas=False`` does not stop it).
    :func:`extract_json_member_object` returns a prefix-stable verbatim
    substring, which the engine's argument-delta diffing requires.
    """
    span = extract_json_member_object(raw_args, "args")
    if span is None:
        # No args value yet (streaming) or none at all (treat as empty).
        return "" if partial else "{}"
    return span


@functools.cache
def inkling_config() -> ParserEngineConfig:
    terminals = {
        "MSG_MODEL": MESSAGE_MODEL,
        "TEXT_START": CONTENT_TEXT,
        "THINK_START": CONTENT_THINKING,
        "THINK_END": END_MESSAGE,
        "END_SAMPLING": CONTENT_MODEL_END_SAMPLING,
        "TOOL_START": CONTENT_INVOKE_TOOL_JSON,
        "TOOL_TEXT": CONTENT_INVOKE_TOOL_TEXT,
        "TOOL_ERROR": CONTENT_TOOL_ERROR,
    }
    transitions: dict[tuple[ParserState, str], Transition] = {
        # ── Between blocks / inside a text block ──────────────────────
        (ParserState.CONTENT, "MSG_MODEL"): Transition(
            ParserState.MESSAGE_HEADER,
            (),
        ),
        # Opening a block that renders as visible content proves no reasoning
        # is still open. Confirming that here is what lets DelegatingParser
        # hand the following blocks to the tool pass: the reasoning pass only
        # leaves its reasoning phase on an explicit reasoning-end event, and a
        # response with no thinking block never emits one otherwise.
        (ParserState.CONTENT, "TEXT_START"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        (ParserState.CONTENT, "THINK_START"): Transition(
            ParserState.REASONING,
            (EventType.REASONING_START,),
        ),
        # A tool block confirms the same boundary, and is the one opener that
        # can start a turn with no visible block ahead of it.
        (ParserState.CONTENT, "TOOL_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        ),
        # Raw / error tool blocks render as visible text.
        (ParserState.CONTENT, "TOOL_TEXT"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        (ParserState.CONTENT, "TOOL_ERROR"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        # The optional function name between the model-role and content-kind
        # markers is metadata, not visible assistant content.
        (ParserState.MESSAGE_HEADER, "MSG_MODEL"): Transition(
            ParserState.MESSAGE_HEADER,
            (),
        ),
        (ParserState.MESSAGE_HEADER, "TEXT_START"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        (ParserState.MESSAGE_HEADER, "THINK_START"): Transition(
            ParserState.REASONING,
            (EventType.REASONING_START,),
        ),
        (ParserState.MESSAGE_HEADER, "TOOL_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        ),
        (ParserState.MESSAGE_HEADER, "TOOL_TEXT"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        (ParserState.MESSAGE_HEADER, "TOOL_ERROR"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
        # ── Inside a thinking block ───────────────────────────────────
        (ParserState.REASONING, "THINK_START"): Transition(
            ParserState.REASONING,
            (),
        ),
        # Defensive: tool call opening while a thinking block is still
        # unclosed.
        (ParserState.REASONING, "TOOL_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        ),
    }
    # Block-end terminals behave identically regardless of label. A
    # closed tool block returns to CONTENT (Inkling has no section wrapper;
    # blocks of any kind may follow), which also keeps the block-kind
    # and role tokens out of the engine's tool-terminal set so the
    # skip_tool_parsing reasoning pass still classifies reasoning.
    for end in ("THINK_END", "END_SAMPLING"):
        transitions[(ParserState.CONTENT, end)] = Transition(
            ParserState.CONTENT,
            (),
        )
        transitions[(ParserState.REASONING, end)] = Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        )
        transitions[(ParserState.TOOL_ARGS, end)] = Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        )
        transitions[(ParserState.MESSAGE_HEADER, end)] = Transition(
            ParserState.CONTENT,
            (EventType.TEXT_CHUNK,),
        )

    return ParserEngineConfig(
        name="inkling",
        # Normal generation continues after a prompt-prefilled
        # `<|message_model|>`. Non-streaming parsing receives only the generated
        # suffix, so begin in the corresponding message-header state as well.
        initial_state=ParserState.MESSAGE_HEADER,
        terminals=terminals,
        # Inkling content-kind markers are the grammar. When the engine is
        # used through DelegatingParser, the reasoning pass can hand the tool
        # pass reconstructed text whose token-id slice no longer contains the
        # content-kind marker that starts the tool block, so keep Inkling on
        # the text grammar instead of token-id-only terminal matching.
        token_id_terminals={},
        transitions=transitions,
        arg_converter=_inkling_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class InklingParser(ParserEngine):
    CONFIG_NAME = "inkling"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", inkling_config())
        super().__init__(tokenizer, tools, **kwargs)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Seed the initial parsing state from the prompt tail.

        Mirrors the Rust parser's ``initialize()``: scanning the prompt
        backwards, the last relevant special token decides whether generation
        continues inside a thinking block, a text block, a model message
        header, or between blocks.
        """
        vocab = self.vocab
        thinking_id = vocab.get(CONTENT_THINKING)
        text_id = vocab.get(CONTENT_TEXT)
        model_id = vocab.get(MESSAGE_MODEL)
        special_ids = {vocab[text] for text in INKLING_SPECIAL_TOKENS if text in vocab}
        for token_id in reversed(prompt_token_ids):
            if token_id == thinking_id:
                self._engine.reset(initial_state=ParserState.REASONING)
                self._streaming_initialized = True
                return
            if token_id == text_id:
                self._engine.reset(initial_state=ParserState.CONTENT)
                self._streaming_initialized = True
                return
            if token_id == model_id:
                self._engine.reset(initial_state=ParserState.MESSAGE_HEADER)
                self._streaming_initialized = True
                return
            if token_id in special_ids:
                break

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        vocab = self.vocab
        thinking_id = vocab.get(CONTENT_THINKING)
        text_id = vocab.get(CONTENT_TEXT)
        model_id = vocab.get(MESSAGE_MODEL)
        end_sampling_id = vocab.get(CONTENT_MODEL_END_SAMPLING)
        for token_id in reversed(input_ids):
            if token_id in (thinking_id, model_id):
                return False
            if token_id in (text_id, end_sampling_id):
                return True
        return False

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        vocab = self.vocab
        thinking_id = vocab.get(CONTENT_THINKING)
        end_ids = {
            token_id
            for token_id in (
                vocab.get(END_MESSAGE),
                vocab.get(CONTENT_MODEL_END_SAMPLING),
            )
            if token_id is not None
        }
        in_reasoning = False
        count = 0
        for token_id in token_ids:
            if token_id == thinking_id:
                in_reasoning = True
                continue
            if token_id in end_ids:
                in_reasoning = False
                continue
            if in_reasoning:
                count += 1
        return count

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        reasoning, content, tool_call_info = super()._single_pass_parse(
            text, token_ids, initial_state=initial_state
        )
        # The engine defers content that follows tool-call events within a
        # single pass; Inkling allows text blocks after tool-call blocks, so
        # flush the trailing text (matching the Rust unified parser).
        if self._deferred_content:
            trailing = self._deferred_content
            self._deferred_content = ""
            content = self._strip_content_whitespace(
                (content or "") + trailing,
                tool_call_info.tools_called,
            )
            tool_call_info = ExtractedToolCallInformation(
                tools_called=tool_call_info.tools_called,
                tool_calls=tool_call_info.tool_calls,
                content=content,
            )
        return reasoning, content, tool_call_info

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
        # Inkling wraps arguments under "args" rather than "arguments".
        for key in ("args", "arguments", "parameters"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return val
                return json.dumps(val, ensure_ascii=False)
        return None
