# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.minimax_m3 import MiniMaxM3Parser


class MiniMaxM3ReasoningParser(ParserEngineReasoningAdapter):
    """Reasoning parser adapter for MiniMax M3 explicit thinking blocks."""

    _parser_engine_cls = MiniMaxM3Parser

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser_engine.is_reasoning_end_streaming(
            list(input_ids), tuple(delta_ids)
        )
