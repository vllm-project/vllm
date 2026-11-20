# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.hy_v4 import HYV4Parser


class HYV4ReasoningParser(ParserEngineReasoningAdapter):
    _parser_engine_cls = HYV4Parser
    _parser_engine: HYV4Parser

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self._parser_engine.is_reasoning_end_streaming(input_ids, delta_ids)
