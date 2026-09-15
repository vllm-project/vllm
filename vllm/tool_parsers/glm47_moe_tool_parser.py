# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
    # GLM renders tool schemas verbatim into the prompt and follows the
    # property order when emitting arguments; putting required fields first
    # keeps the model from omitting them once it starts on optional fields.
    # The strict-mode grammar (xgrammar glm_4_7) enforces the same order.
    reorder_tool_schema_required_first = True
