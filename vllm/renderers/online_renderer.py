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
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.parser.harmony_utils import (
    build_harmony_preamble,
    extract_instructions_from_messages,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
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
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser
from vllm.utils.mistral import mt as _mt

logger = init_logger(__name__)


class OnlineRenderer:
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
        """Copied from GenerateBaseServing._validate_chat_template."""
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
        """Copied from GenerateBaseServing._preprocess_completion."""
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
        """Copied from GenerateBaseServing._preprocess_cmpl."""
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
        """Copied from GenerateBaseServing._preprocess_chat."""
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
