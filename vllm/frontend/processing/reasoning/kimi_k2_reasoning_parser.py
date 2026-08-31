# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.frontend.processing.parser.engine.registered_adapters import (
    KimiK2ParserReasoningAdapter,
)

KimiK2ReasoningParser = KimiK2ParserReasoningAdapter

__all__ = ["KimiK2ReasoningParser"]
