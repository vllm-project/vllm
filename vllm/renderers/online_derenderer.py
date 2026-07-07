# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.base.serving import resolve_token_id_placeholder
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import ToolCall
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.parser import Parser, ParserManager
from vllm.renderers import BaseRenderer
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OnlineDerenderer:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.request_logger = request_logger

        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.parser: type[Parser] | None = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=model_config.model,
            is_harmony=self.use_harmony,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: ChatTemplateContentFormatOption = (
            chat_template_content_format
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.trust_request_chat_template = trust_request_chat_template

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

    async def derender_chat(
        self,
        generate_response: GenerateResponse,
        chat_request: ChatCompletionRequest | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        tokenizer = self.renderer.get_tokenizer()
        choices: list[ChatCompletionResponseChoice] = []

        for choice in generate_response.choices:
            if not choice.token_ids:
                raise ValueError(f"choice {choice.index} has empty or null token_ids")

            resolved_logprobs = (
                _resolve_logprobs(choice.logprobs, tokenizer)
                if choice.logprobs is not None
                else None
            )

            if self.parser is not None and chat_request is not None:
                # Parser path: decode with special tokens preserved
                # so the parser can see markers like </think>,
                # <tool_call>, or Harmony channel tokens.
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=False
                )

                chat_template_kwargs: dict[str, Any] = {}
                if not self.use_harmony:
                    chat_template_kwargs = (
                        chat_request.build_chat_params(
                            self.chat_template,
                            self.chat_template_content_format,
                        )
                        .with_defaults(self.default_chat_template_kwargs)
                        .chat_template_kwargs
                    )

                parser = self.parser(
                    tokenizer,
                    chat_request.tools,
                    chat_template_kwargs=chat_template_kwargs,
                )
                reasoning, content, tool_calls = parser.parse(
                    decoded_text,
                    chat_request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=choice.token_ids,
                )

                if not getattr(chat_request, "include_reasoning", True):
                    reasoning = None

                tc_items = (
                    [
                        ToolCall(
                            id=random_uuid(),
                            function=tc,
                        )
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else []
                )
                auto_tools_called = (
                    bool(tc_items)
                    and bool(chat_request.tools)
                    and (
                        chat_request.tool_choice == "auto"
                        or chat_request.tool_choice is None
                    )
                    and self.enable_auto_tools
                )
                is_required_tool_choice = chat_request.tool_choice == "required"

                message = ChatMessage(
                    role="assistant",
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tc_items,
                )
                finish_reason = (
                    "tool_calls"
                    if auto_tools_called
                    or (
                        is_required_tool_choice
                        and bool(tc_items)
                        and choice.finish_reason == "stop"
                    )
                    else choice.finish_reason
                )
            else:
                # No parser: plain detokenization.
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=True
                )
                message = ChatMessage(role="assistant", content=decoded_text)
                finish_reason = choice.finish_reason

            choices.append(
                ChatCompletionResponseChoice(
                    index=choice.index,
                    message=message,
                    logprobs=resolved_logprobs,
                    finish_reason=finish_reason,
                )
            )

        return choices

    async def derender_completion(
        self,
        generate_responses: list[GenerateResponse],
        prompt_tokens: list[int] | None = None,
    ) -> tuple[list[CompletionResponseChoice], int, int]:
        n = len(generate_responses)
        prompt_tokens_list: list[int] = (
            prompt_tokens if prompt_tokens is not None else [0] * n
        )

        tokenizer = self.renderer.get_tokenizer()
        choices: list[CompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        index = 0

        for gen, pt in zip(generate_responses, prompt_tokens_list):
            for choice in gen.choices:
                if not choice.token_ids:
                    raise ValueError(
                        f"choice {choice.index} in response {gen.request_id} "
                        "has empty or null token_ids"
                    )

                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=True
                )
                completion_logprobs = None
                if choice.logprobs is not None:
                    resolved = _resolve_logprobs(choice.logprobs, tokenizer)
                    completion_logprobs = _convert_chat_logprobs_to_completion_logprobs(
                        resolved
                    )
                choices.append(
                    CompletionResponseChoice(
                        index=index,
                        text=decoded_text,
                        finish_reason=choice.finish_reason,
                        logprobs=completion_logprobs,
                    )
                )
                total_completion_tokens += len(choice.token_ids)
                index += 1
            total_prompt_tokens += pt

        return choices, total_prompt_tokens, total_completion_tokens


def _parse_token_id_placeholder(token: str) -> int | None:
    """Extract token ID from a 'token_id:N' placeholder string."""
    if not token.startswith("token_id:"):
        return None
    try:
        return int(token[len("token_id:") :])
    except ValueError:
        return None


def _correct_decoded_token(
    token_id: int, context_token_ids: list[int], tokenizer: TokenizerLike
) -> str:
    """Use preceding tokens as context to fix U+FFFD from byte-fallback.

    Mirrors LogprobsProcessor._correct_decoded_token in v1/engine/logprobs.py.
    """
    max_ctx = min(len(context_token_ids), 4)

    for num_ctx in range(1, max_ctx + 1):
        context = context_token_ids[-num_ctx:]
        full_decoded = tokenizer.decode(context + [token_id])

        if full_decoded.endswith("�"):
            continue

        clean_end = len(context)
        for j in range(len(context) - 1, -1, -1):
            if tokenizer.decode([context[j]]).endswith("�"):
                clean_end = j
            else:
                break

        clean_prefix = tokenizer.decode(context[:clean_end]) if clean_end > 0 else ""

        if full_decoded.startswith(clean_prefix):
            return full_decoded[len(clean_prefix) :]

        common_len = 0
        for a, b in zip(clean_prefix, full_decoded):
            if a != b:
                break
            common_len += 1
        return full_decoded[common_len:]

    return ""


def _resolve_logprobs(
    logprobs: ChatCompletionLogProbs, tokenizer: TokenizerLike
) -> ChatCompletionLogProbs:
    """Resolve token_id:N placeholders in a ChatCompletionLogProbs object."""
    if logprobs.content is None:
        return logprobs

    context_token_ids: list[int] = []
    resolved_content = []

    for entry in logprobs.content:
        token_str, token_bytes = resolve_token_id_placeholder(entry.token, tokenizer)
        sampled_id = _parse_token_id_placeholder(entry.token)

        if token_str.endswith("�") and sampled_id is not None:
            token_str = _correct_decoded_token(sampled_id, context_token_ids, tokenizer)
            token_bytes = list(token_str.encode("utf-8"))

        resolved_top = []
        for top in entry.top_logprobs:
            top_str, top_bytes = resolve_token_id_placeholder(top.token, tokenizer)
            top_id = _parse_token_id_placeholder(top.token)
            if top_str.endswith("�") and top_id is not None:
                top_str = _correct_decoded_token(top_id, context_token_ids, tokenizer)
                top_bytes = list(top_str.encode("utf-8"))
            resolved_top.append(
                top.model_copy(update={"token": top_str, "bytes": top_bytes})
            )

        resolved_content.append(
            entry.model_copy(
                update={
                    "token": token_str,
                    "bytes": token_bytes,
                    "top_logprobs": resolved_top,
                }
            )
        )

        if sampled_id is not None:
            context_token_ids.append(sampled_id)

    return ChatCompletionLogProbs(content=resolved_content)


def _convert_chat_logprobs_to_completion_logprobs(
    logprobs: ChatCompletionLogProbs,
) -> CompletionLogProbs:
    """Convert ChatCompletionLogProbs (per-token objects) to CompletionLogProbs
    (parallel flat lists) as required by the /v1/completions response schema."""
    if logprobs.content is None:
        return CompletionLogProbs()

    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs_list: list[dict[str, float] | None] = []
    text_offset: list[int] = []

    offset = 0
    for entry in logprobs.content:
        text_offset.append(offset)
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs_list.append(
            {t.token: t.logprob for t in entry.top_logprobs}
            if entry.top_logprobs
            else None
        )
        offset += len(entry.token)

    return CompletionLogProbs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs_list,
    )
