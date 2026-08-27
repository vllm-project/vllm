# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import InklingParserToolAdapter


class InklingEngineToolParser(InklingParserToolAdapter):  # type: ignore[valid-type, misc]
    # An Inkling structural tag is registered in structural_tag_registry.py, so
    # named/required tool choice is constrained on the model's real wire format
    # (<|message_model|><|content_invoke_tool_json|>{...}<|end_message|>) instead
    # of silently falling back to "auto".
    #
    # Do NOT also set supports_required_and_named: AbstractToolParser's
    # __init_subclass__ forces it to False whenever structural_tag_model is set
    # and VLLM_ENFORCE_STRICT_TOOL_CALLING is on. That is the intended design --
    # the structural tag supersedes the generic required/named JSON path -- so
    # setting it here is both redundant and silently overwritten.
    structural_tag_model = "inkling"
