# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings


def __getattr__(name: str):
    if name == "ToolParser":
        from vllm.tool_parsers import ToolParser

        warnings.warn(
            "`vllm.entrypoints.openai.tool_parsers.ToolParser` has been moved to "
            "`vllm.tool_parsers.ToolParser`. "
            "The old name will be removed in v0.14.",
            DeprecationWarning,
            stacklevel=2,
        )

        return ToolParser
    if name == "ToolParserManager":
        from vllm.tool_parsers import ToolParserManager

        warnings.warn(
            "`vllm.entrypoints.openai.tool_parsers.ToolParserManager` "
            "has been moved to `vllm.tool_parsers.ToolParserManager`. "
            "The old name will be removed in v0.14.",
            DeprecationWarning,
            stacklevel=2,
        )

        return ToolParserManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
