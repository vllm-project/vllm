# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo's compact XML parameters contain verbatim string values."""

import functools
import json
from dataclasses import replace

from vllm.parser.qwen3 import (
    _PARAM_RE,
    _PARTIAL_PARAM_RE,
    CHATML_TURN_BOUNDARIES,
    Qwen3Parser,
    _legacy_param_value_extender,
    qwen3_config,
)


def _mimo_arg_converter(raw_args: str, partial: bool) -> str:
    params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(raw_args)}
    if partial:
        match = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
        if match and match.group(1):
            params[match.group(1)] = match.group(2)
    return json.dumps(params, ensure_ascii=False)


class MiMoParser(Qwen3Parser):
    def __init__(self, tokenizer, tools=None, **kwargs):
        thinking = (kwargs.get("chat_template_kwargs") or {}).get(
            "enable_thinking", True
        )
        kwargs.setdefault(
            "parser_engine_config",
            replace(
                qwen3_config(
                    thinking=thinking,
                    name="mimo",
                    turn_boundary_tokens=CHATML_TURN_BOUNDARIES,
                    # MiMo keeps the flat next-opener recovery with verbatim
                    # values; the nested parameter states are qwen3-only.
                    nested_param_values=False,
                ),
                arg_converter=_mimo_arg_converter,
                arg_value_extender=functools.partial(
                    _legacy_param_value_extender, trim_newlines=False
                ),
            ),
        )
        super().__init__(tokenizer, tools, **kwargs)
