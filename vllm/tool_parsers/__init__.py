# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.utils.tool_parser_registry import BUILTIN_TOOL_PARSERS

__all__ = ["ToolParser", "ToolParserManager"]


"""
Register a lazy module mapping.

Example:
    ToolParserManager.register_lazy_module(
        name="kimi_k2",
        module_path="vllm.tool_parsers.kimi_k2_parser",
        class_name="KimiK2ToolParser",
    )
"""


def register_lazy_tool_parsers():
    for name, (file_name, class_name) in BUILTIN_TOOL_PARSERS.items():
        module_path = f"vllm.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
