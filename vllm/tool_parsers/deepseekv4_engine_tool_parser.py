# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import DeepSeekV4ParserToolAdapter


class DeepSeekV4EngineToolParser(DeepSeekV4ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v4"
