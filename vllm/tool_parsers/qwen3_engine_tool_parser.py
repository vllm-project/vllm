# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from vllm.parser.engine.parser_engine_config import ParserState


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"

    @staticmethod
    def _has_tool_history(request) -> bool:
        for message in getattr(request, "messages", []) or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "tool":
                return True
            if message.get("role") == "assistant" and message.get("tool_calls"):
                return True
        return False

    def adjust_request(self, request):
        if (
            os.getenv("VLLM_QWEN3_FORCE_FIRST_AUTO_TOOL_REQUIRED", "0") == "1"
            and getattr(request, "tools", None)
            and getattr(request, "tool_choice", None) == "auto"
            and not self._has_tool_history(request)
        ):
            request.tool_choice = "required"
        return super().adjust_request(request)

    def extract_tool_calls(self, model_output, request):
        return self._parser_engine.extract_tool_calls(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        engine = self._parser_engine
        engine.initialize_streaming(initial_state=ParserState.REASONING)
        return engine.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
