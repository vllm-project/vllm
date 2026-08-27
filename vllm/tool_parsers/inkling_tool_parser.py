# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import InklingParserToolAdapter


class InklingEngineToolParser(InklingParserToolAdapter):  # type: ignore[valid-type, misc]
    # An Inkling structural tag IS now registered (structural_tag_registry.py), so
    # named/required tool choice can be constrained on the model's real wire format
    # (<|message_model|><|content_invoke_tool_json|>{...}<|end_message|>) instead of
    # silently falling back to "auto".
    structural_tag_model = "inkling"
    supports_required_and_named = True
