# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import MinimaxM2ParserToolAdapter


class MinimaxM2ToolParser(MinimaxM2ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "minimax"
