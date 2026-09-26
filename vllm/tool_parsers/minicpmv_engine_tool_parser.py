# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import MiniCPMVParserToolAdapter


class MiniCPMVEngineToolParser(MiniCPMVParserToolAdapter):  # type: ignore[valid-type, misc]
    # MiniCPM-V emits the same XML tool call syntax as Qwen3-Coder.
    structural_tag_model = "qwen_3_coder"
