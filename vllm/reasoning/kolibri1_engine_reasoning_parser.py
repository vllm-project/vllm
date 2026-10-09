# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.kolibri1 import Kolibri1Parser


class Kolibri1ParserReasoningAdapter(ParserEngineReasoningAdapter):
    """Reasoning only: Kolibri 1 emits Hermes tool calls, which the
    ``kolibri1`` tool parser handles."""

    _parser_engine_cls = Kolibri1Parser


__all__ = ["Kolibri1ParserReasoningAdapter"]
