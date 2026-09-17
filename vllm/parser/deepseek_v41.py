# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls."""

import functools
from dataclasses import replace

import regex as re

from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    deepseek_v4_config,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*?)"
    r"(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter\s+name=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*)$",
    re.DOTALL,
)


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
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            key: terminal_overrides.get(key, value)
            for key, value in config.token_id_terminals.items()
        },
        arg_converter=functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)
