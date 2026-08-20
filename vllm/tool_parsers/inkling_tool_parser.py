# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import InklingParserToolAdapter


class InklingEngineToolParser(InklingParserToolAdapter):  # type: ignore[valid-type, misc]
    """Legacy tool-parser name backed by the Inkling parser engine."""
