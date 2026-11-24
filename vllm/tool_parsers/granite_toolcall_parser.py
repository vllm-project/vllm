# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import GraniteToolCallParserToolAdapter


class GraniteToolCallToolParser(GraniteToolCallParserToolAdapter):  # type: ignore[valid-type, misc]
    # Structural-tag model whose grammar matches Granite 4.2's XML tool-call
    # format; one of the keys in SUPPORTED_STRUCTURAL_TAG_MODELS. Only consulted
    # under VLLM_ENFORCE_STRICT_TOOL_CALLING.
    structural_tag_model = "qwen_3_coder"
