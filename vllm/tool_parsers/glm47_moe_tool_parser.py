# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
    # Constrain auto tool calls with the shallow tag by default.
    default_tool_strict_level = ToolStrictLevel.FUNCTION
