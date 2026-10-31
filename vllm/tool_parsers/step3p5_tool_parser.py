# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import Step3p5ParserToolAdapter
from vllm.tool_parsers.qwen3_engine_tool_parser import (
    UnwrappedQwenXmlStructuralTagMixin,
)


class Step3p5ToolParser(
    UnwrappedQwenXmlStructuralTagMixin,
    Step3p5ParserToolAdapter,  # type: ignore[valid-type, misc]
):
    structural_tag_model = "qwen_3_coder"
