# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import InklingParserToolAdapter


class InklingEngineToolParser(InklingParserToolAdapter):  # type: ignore[valid-type, misc]
    # No Inkling structural-tag grammar is wired up yet; fall back to auto
    # parsing for named/required tool choice.
    structural_tag_model = None
    supports_required_and_named = False
