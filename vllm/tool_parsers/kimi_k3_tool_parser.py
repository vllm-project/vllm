# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import KimiK3ParserToolAdapter


class KimiK3ToolParser(KimiK3ParserToolAdapter):  # type: ignore[valid-type, misc]
    """Tool-parser compatibility entry point backed by KimiK3Parser."""

    structural_tag_model = "kimi_k3"
