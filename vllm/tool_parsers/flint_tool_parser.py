# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.flint import FlintParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


class FlintToolParser(FlintParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
    # Constrain auto tool calls with the shallow tag by default.
    default_tool_strict_level = ToolStrictLevel.FUNCTION
