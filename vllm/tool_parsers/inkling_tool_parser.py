# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import InklingParserToolAdapter


class InklingEngineToolParser(InklingParserToolAdapter):  # type: ignore[valid-type, misc]
    # The structural tag is registered in structural_tag_registry.py. Without it,
    # named and required tool choice fall back to "auto".
    #
    # Do not also set supports_required_and_named. __init_subclass__ forces it to
    # False when structural_tag_model is set, so the value here has no effect.
    structural_tag_model = "inkling"
