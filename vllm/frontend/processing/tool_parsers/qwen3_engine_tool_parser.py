# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.frontend.processing.parser.engine.registered_adapters import Qwen3ParserToolAdapter


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"
