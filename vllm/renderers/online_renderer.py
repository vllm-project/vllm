# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

from openai_harmony import Message as OpenAIMessage

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, ToolCall
from vllm.entrypoints.openai.engine.serving import resolve_token_id_placeholder
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.openai.parser.harmony_utils import (
    build_harmony_preamble,
    extract_instructions_from_messages,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import (
    EngineInput,
    PromptType,
    SingletonPrompt,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.parser import Parser, ParserManager
from vllm.renderers import BaseRenderer, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser
from vllm.utils.mistral import mt as _mt

logger = init_logger(__name__)


class OnlineRenderer:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        model_registry: OpenAIModelRegistry,
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
        self.model_registry = model_registry
        self.request_logger = request_logger

        self.trust_request_chat_template = trust_request_chat_template
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

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

    async def render_chat(
        self,
        request: ChatCompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.
        """
        tokenizer = self.renderer.tokenizer

        tool_parser = self.parser.tool_parser_cls if self.parser is not None else None

        if is_mistral_tokenizer(tokenizer):
            # because of issues with pydantic we need to potentially
            # re-serialize the tool_calls field of the request
            _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]
            _mt.truncate_tool_call_ids(request)  # type: ignore[arg-type]
            _mt.validate_request_params(request)

        # Check if tool parsing is unavailable (common condition)
        tool_parsing_unavailable = (
            tool_parser is None
            and not is_mistral_tokenizer(tokenizer)
            and not self.use_harmony
        )

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (
            None,
            "none",
        ):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                # "required" or named tool requires tool parser
                return self.create_error_response(
                    f'tool_choice="{request.tool_choice}" requires '
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        if not self.use_harmony:
            # Common case.
            error_check_ret = self.validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

            conversation, engine_inputs = await self.preprocess_chat(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=self.default_chat_template_kwargs,
                tool_dicts=tool_dicts,
                parser=self.parser,
                skip_mm_cache=skip_mm_cache,
            )
        else:
            # For GPT-OSS.
            should_include_tools = tool_dicts is not None
            conversation, engine_inputs = self._make_request_with_harmony(
                request, should_include_tools
            )

        return conversation, engine_inputs

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
        should_include_tools: bool = True,
    ):
        """Build Harmony (GPT-OSS) messages and engine prompt from a chat request."""
        messages: list[OpenAIMessage] = []

        # because of issues with pydantic we need to potentially
        # re-serialize the tool_calls field of the request
        # for more info: see comment in `maybe_serialize_tool_calls`
        _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]

        chat_messages = list(request.messages)
        instructions, chat_messages = extract_instructions_from_messages(chat_messages)

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        if (reasoning_effort := request.reasoning_effort) == "none":
            raise ValueError(f"Harmony does not support {reasoning_effort=}")
        tools = request.tools if should_include_tools else None
        messages.extend(
            build_harmony_preamble(
                instructions=instructions,
                tools=tools,  # type: ignore[arg-type]
                reasoning_effort=reasoning_effort,
                with_custom_tools=should_include_tools,
            )
        )

        # Add remaining conversation messages.
        messages.extend(parse_chat_inputs_to_harmony_messages(chat_messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=request.cache_salt)

        return messages, [engine_input]

    async def render_completion(
        self,
        request: CompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput] | ErrorResponse:
        """Core preprocessing logic for completion requests (no model/engine check).

        Called directly by render_completion_request and delegated to by
        OpenAIServingCompletion.render_completion_request after its engine-aware checks.
        """
        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response("suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
            skip_mm_cache=skip_mm_cache,
        )

        return engine_inputs

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        """Copied from OpenAIServing._validate_chat_template."""
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    async def preprocess_completion(
        self,
        request: Any,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput]:
        """Copied from OpenAIServing._preprocess_completion."""
        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))
        return await self.preprocess_cmpl(request, prompts, skip_mm_cache=skip_mm_cache)

    async def preprocess_cmpl(
        self,
        request: Any,
        prompts: Sequence[PromptType | bytes],
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput]:
        """Copied from OpenAIServing._preprocess_cmpl."""
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

    async def preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: type[Parser] | None = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        """Copied from OpenAIServing._preprocess_chat."""
        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=(
                    is_mistral_tokenizer(renderer.tokenizer)
                    or self.model_config.enable_prompt_embeds
                ),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        (conversation,), (engine_input,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        #
        # Exception: Mistral grammar-capable tokenizers always call
        # adjust_request — even for tool_choice="none" — so that the grammar
        # factory can prevent special-token leakage.
        if parser is not None:
            tokenizer = renderer.get_tokenizer()
            tool_parser = parser.tool_parser_cls
            tool_choice = getattr(request, "tool_choice", "none")
            is_mistral_grammar_eligible = (
                tool_parser is not None
                and is_mistral_tool_parser(tool_parser)
                and is_mistral_tokenizer(tokenizer)
                and tokenizer.supports_grammar
            )
            should_adjust_request = (
                parser.reasoning_parser_cls is not None
                or tool_choice != "none"
                or is_mistral_grammar_eligible
            )
            if should_adjust_request:
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported "
                        "for Chat Completions API or Responses API requests, "
                        f"but got {type(request).__name__}"
                    )
                    raise NotImplementedError(msg)
                request = parser(
                    tokenizer,
                    request.tools,
                    model_config=self.model_config,
                    chat_template_kwargs=chat_params.chat_template_kwargs,
                ).adjust_request(
                    request=request,
                )

        return conversation, [engine_input]

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

                message = ChatMessage(
                    role="assistant",
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tc_items,
                )
            else:
                # No parser: plain detokenization.
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=True
                )
                message = ChatMessage(role="assistant", content=decoded_text)

            choices.append(
                ChatCompletionResponseChoice(
                    index=choice.index,
                    message=message,
                    logprobs=resolved_logprobs,
                    finish_reason=choice.finish_reason,
                )
            )

        return choices

    async def derender_completion(
        self,
        generate_responses: list[GenerateResponse],
        prompt_tokens: list[int] | None = None,
    ) -> (list[CompletionResponseChoice], int, int):

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
                    return self.create_error_response(
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
