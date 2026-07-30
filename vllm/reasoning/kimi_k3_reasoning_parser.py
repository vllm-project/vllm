# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import KimiK3ParserReasoningAdapter


class KimiK3ReasoningParser(KimiK3ParserReasoningAdapter):  # type: ignore[valid-type, misc]
    """Reasoning-parser compatibility entry point backed by KimiK3Parser."""
