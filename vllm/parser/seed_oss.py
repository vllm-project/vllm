# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""seed_oss parser for tool calls and reasoning.

seed_oss shares the Qwen3 XML grammar exactly; only the wrapper token
strings and the turn-boundary tokens differ::

    <think>      -> <seed:think>
    </think>     -> </seed:think>
    <tool_call>  -> <seed:tool_call>
    </tool_call> -> </seed:tool_call>
    <|im_start|> -> <seed:bos>
    <|im_end|>   -> <seed:eos>

``<function=...>`` and ``<parameter=...>`` are byte-identical, so the
entire transition table and ``_qwen3_arg_converter`` are inherited from
:class:`Qwen3Parser` unchanged.
"""

from __future__ import annotations

from vllm.parser.qwen3 import Qwen3Parser


class SeedOssParser(Qwen3Parser):
    CONFIG_NAME = "seed_oss"
    THINK_START = "<seed:think>"
    THINK_END = "</seed:think>"
    TOOL_START = "<seed:tool_call>"
    TOOL_END = "</seed:tool_call>"
    TURN_BOUNDARIES = frozenset(("<seed:bos>", "<seed:eos>"))
