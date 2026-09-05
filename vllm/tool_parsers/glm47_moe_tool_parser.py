# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    """Legacy tool-parser name backed by the GLM-4.7 parser engine."""
