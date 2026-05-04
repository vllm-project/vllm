# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class DeepSeekV4ReasoningParser(DeepSeekV3ReasoningParser):
    """
    DeepSeek V4 reasoning parser.

    DeepSeek V4 can start DSML tool calls directly after reasoning. Treating the
    tool-call block as content lets the DeepSeek V4 tool parser handle it.
    """

    tool_call_start_token = "<｜DSML｜tool_calls>"

    def _r1_parser(self) -> DeepSeekR1ReasoningParser | None:
        if isinstance(self._parser, DeepSeekR1ReasoningParser):
            return self._parser
        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        r1_parser = self._r1_parser()
        if r1_parser is None:
            return self._parser.extract_reasoning(model_output, request)

        reasoning, content = r1_parser.extract_reasoning(model_output, request)
        if reasoning is None or self.tool_call_start_token not in reasoning:
            return reasoning, content

        reasoning, _, tool_content = reasoning.partition(self.tool_call_start_token)
        content = self.tool_call_start_token + tool_content + (content or "")
        return reasoning, content
