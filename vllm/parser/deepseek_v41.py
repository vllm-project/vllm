# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls."""

import functools
from dataclasses import replace

from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _param_patterns,
    deepseek_v4_config,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
# Spelling variants of the invoke opener seen in production output: the
# model sometimes emits the tool name as a bare parameter-like tag instead
# of an ``invoke`` tag, with or without the DSML marker.
DSML_INVOKE_PREFIX_VARIANTS = (
    DSML_INVOKE_PREFIX,
    '<｜DSML｜ parameter name="',
    '<parameter name="',
)
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    terminal_overrides = {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    return replace(
        config,
        name="deepseek_v41",
        terminals={
            **config.terminals,
            **terminal_overrides,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX_VARIANTS,
        },
        token_id_terminals={
            key: terminal_overrides.get(key, value)
            for key, value in config.token_id_terminals.items()
        },
        transitions={
            **config.transitions,
            # A drifted call block may omit both the tool-call block opener and
            # the reasoning end marker: close reasoning and open the tool call.
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
        },
        arg_converter=functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)
