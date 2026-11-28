# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.parser.engine.registered_adapters import MuseParserReasoningAdapter
from vllm.parser.muse_grammar import muse_answer_structural_tag

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MuseEngineReasoningParser(MuseParserReasoningAdapter):  # type: ignore[valid-type, misc]
    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Constrain Muse's surfaced `to=user` answer to the user's format."""
        request = super().adjust_request(request)
        structured_outputs = request.extract_structured_outputs()
        if structured_outputs is None:
            return request

        from vllm.parser.abstract_parser import structured_outputs_to_format

        answer = structured_outputs_to_format(structured_outputs)
        if answer is None:
            return request

        tag = muse_answer_structural_tag(answer)
        request.structured_outputs = replace(
            structured_outputs,
            json=None,
            regex=None,
            choice=None,
            grammar=None,
            json_object=None,
            structural_tag=json.dumps(tag.model_dump()),
        )
        if hasattr(request, "text"):
            request.text = None
        else:
            request.response_format = None
        request._grammar_from_parser = True
        return request

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser_engine.is_reasoning_end_streaming(input_ids, delta_ids)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self._parser_engine.count_reasoning_tokens(token_ids)
