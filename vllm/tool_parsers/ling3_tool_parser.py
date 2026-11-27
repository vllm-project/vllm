# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.ling3 import Ling3ParserToolAdapter


class Ling3ToolParser(Ling3ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
