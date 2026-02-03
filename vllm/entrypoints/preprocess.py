# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared preprocessing functions for chat and completion requests.

These functions extract the preprocessing logic from serving classes
so it can be shared between different serving layers (OpenAI HTTP, gRPC, etc.).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.inputs import EmbedsPrompt, TokensPrompt
from vllm.renderers import BaseRenderer, ChatParams, TokenizeParams, merge_kwargs
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.tool_parsers import ToolParser


class RendererRequest(Protocol):
    """Protocol for requests that can build tokenization parameters."""

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    """Protocol for chat requests that can build chat and tokenization parameters."""

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


async def preprocess_chat(
    renderer: BaseRenderer,
    model_config: ModelConfig,
    request: RendererChatRequest,
    messages: list[ChatCompletionMessageParam],
    default_template: str | None,
    default_template_content_format: ChatTemplateContentFormatOption,
    default_template_kwargs: dict[str, Any] | None,
    tool_dicts: list[dict[str, Any]] | None = None,
    tool_parser: Callable[[TokenizerLike], "ToolParser"] | None = None,
) -> tuple[list[ConversationMessage], list[TokensPrompt | EmbedsPrompt]]:
    """
    Preprocess chat messages for inference.

    This is the shared preprocessing logic used by both OpenAI HTTP serving
    and gRPC serving layers.

    Args:
        renderer: The BaseRenderer instance
        model_config: The model configuration
        request: Request object implementing RendererChatRequest protocol
        messages: List of chat messages
        default_template: Default chat template
        default_template_content_format: Content format for chat template
        default_template_kwargs: Default kwargs for chat template
        tool_dicts: Tool definitions
        tool_parser: Optional tool parser factory

    Returns:
        Tuple of (conversation, engine_prompts)
    """
    from vllm.tokenizers.mistral import MistralTokenizer

    default_template_kwargs = merge_kwargs(
        default_template_kwargs,
        dict(
            tools=tool_dicts,
            tokenize=isinstance(renderer.tokenizer, MistralTokenizer),
        ),
    )

    tok_params = request.build_tok_params(model_config)
    chat_params = request.build_chat_params(
        default_template, default_template_content_format
    ).with_defaults(default_template_kwargs)

    conversation, prompt = await renderer.render_messages_async(messages, chat_params)
    engine_prompt = await renderer.tokenize_prompt_async(prompt, tok_params)

    extra_items = {
        k: v
        for k in ("mm_processor_kwargs", "cache_salt")
        if (v := getattr(request, k, None)) is not None
    }
    engine_prompt.update(extra_items)  # type: ignore

    # tool parsing is done only if a tool_parser has been set and if
    # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
    # is set, we want to prevent parsing a tool_call hallucinated by the LLM
    if tool_parser is not None:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        tool_choice = getattr(request, "tool_choice", "none")
        if tool_choice != "none":
            if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                msg = (
                    "Tool usage is only supported for Chat Completions API "
                    "or Responses API requests."
                )
                raise NotImplementedError(msg)

            # TODO: Update adjust_request to accept ResponsesRequest
            tokenizer = renderer.get_tokenizer()
            request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

    return conversation, [engine_prompt]


async def preprocess_completion(
    renderer: BaseRenderer,
    model_config: ModelConfig,
    request: RendererRequest,
    prompt_input: str | list[str] | list[int] | list[list[int]] | None,
    prompt_embeds: bytes | list[bytes] | None,
) -> list[TokensPrompt | EmbedsPrompt]:
    """
    Preprocess completion inputs for inference.

    This is the shared preprocessing logic used by both OpenAI HTTP serving
    and gRPC serving layers.

    Args:
        renderer: The BaseRenderer instance
        model_config: The model configuration
        request: Request object implementing RendererRequest protocol
        prompt_input: Text or token inputs
        prompt_embeds: Embedding inputs

    Returns:
        List of tokenized engine prompts
    """
    tok_params = request.build_tok_params(model_config)

    in_prompts = await renderer.render_completions_async(prompt_input, prompt_embeds)
    engine_prompts = await renderer.tokenize_prompts_async(in_prompts, tok_params)

    extra_items = {
        k: v
        for k in ("mm_processor_kwargs", "cache_salt")
        if (v := getattr(request, k, None)) is not None
    }
    for prompt in engine_prompts:
        prompt.update(extra_items)  # type: ignore

    return engine_prompts
