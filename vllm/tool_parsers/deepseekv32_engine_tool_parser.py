# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import DeepSeekV32ParserToolAdapter


class DeepSeekV32EngineToolParser(DeepSeekV32ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v3_2"
