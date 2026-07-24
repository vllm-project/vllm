# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    # Honor VLLM_ENFORCE_STRICT_TOOL_CALLING via __init_subclass__ (strict=0 -> True).
    structural_tag_model = "glm_4_7"
