# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""seed_oss parser for tool calls and reasoning.

seed_oss shares the Qwen3 XML grammar exactly; only the four wrapper
token strings differ::

    <think>      -> <seed:think>
    </think>     -> </seed:think>
    <tool_call>  -> <seed:tool_call>
    </tool_call> -> </seed:tool_call>

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
