# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import regex as re

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_ACTION_START = "<|action_start|>"
_PLUGIN = "<|plugin|>"
_ACTION_END = "<|action_end|>"

_ACTION_BLOCK_RE = re.compile(
    rf"{re.escape(_ACTION_START)}\s*{re.escape(_PLUGIN)}\s*.*?\s*"
    rf"{re.escape(_ACTION_END)}",
    re.DOTALL,
)


def _trim_newlines(text: str) -> str:
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _thinking_disabled(
    request: "ChatCompletionRequest | ResponsesRequest",
) -> bool:
    chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
    if (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.get("enable_thinking") is False
    ):
        return True

    # OpenAIBaseModel still allows extra top-level fields, so keep a defensive
    # fallback for callers that pass enable_thinking directly.
    return getattr(request, "enable_thinking", None) is False


class InternS1ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        thinking_disabled = _thinking_disabled(request)
        start_token = self.start_token
        end_token = self.end_token

        prefix = ""
        body = model_output
        if start_token in model_output:
            prefix, _, body = model_output.partition(start_token)

        if end_token not in body:
            if thinking_disabled:
                return None, model_output or None
            return super().extract_reasoning(model_output, request)

        reasoning, _, suffix = body.partition(end_token)
        action_blocks = list(_ACTION_BLOCK_RE.finditer(reasoning))
        if not action_blocks:
            if thinking_disabled:
                reasoning_text = _trim_newlines(reasoning)
                visible_content = f"{prefix}{suffix}"
                return reasoning_text or None, visible_content or None
            return super().extract_reasoning(model_output, request)

        reasoning_parts: list[str] = []
        hoisted_actions: list[str] = []
        cursor = 0
        for match in action_blocks:
            reasoning_parts.append(reasoning[cursor : match.start()])
            hoisted_actions.append(match.group(0))
            cursor = match.end()
        reasoning_parts.append(reasoning[cursor:])

        reasoning_text = _trim_newlines("".join(reasoning_parts))
        content = f"{prefix}{''.join(hoisted_actions)}{suffix}"

        if thinking_disabled:
            return (
                reasoning_text or None,
                content if content else reasoning_text or None,
            )

        return (
            reasoning_text or None,
            content if content else None,
        )
