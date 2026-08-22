# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import DeepSeekV32ParserToolAdapter


class DeepSeekV32EngineToolParser(DeepSeekV32ParserToolAdapter):  # type: ignore[valid-type, misc]
    """Legacy tool-parser name backed by the DeepSeek V3.2 parser engine."""
