# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

warnings.warn(
    "Importing ToolParserManager from 'vllm.tool_parsers.manager' is deprecated "
    "and will be removed in a future release. "
    "Please import from 'vllm.tools.parsers.manager' instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["ToolParser", "ToolParserManager"]
