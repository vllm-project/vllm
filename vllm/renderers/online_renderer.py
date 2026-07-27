# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, TypeAlias

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.tool import Mcp, Tool
from openai_harmony import Message as OpenAIMessage
from openai_harmony import ToolNamespaceConfig

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.parser.harmony_utils import (
    BUILTIN_TOOL_TO_MCP_SERVER_LABEL,
    build_harmony_preamble,
    extract_instructions_from_messages,
    get_user_message,
    has_custom_tools,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.harmony import (
    construct_harmony_previous_input_messages,
    response_input_to_harmony,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
    construct_tool_dicts,
    extract_tool_types,
)
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
from vllm.renderers import BaseRenderer, ChatParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser
from vllm.utils.mistral import mt as _mt

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.mcp.tool_server import ToolServer


ResponsesPreviousMessages: TypeAlias = (
    list[ChatCompletionMessageParam] | list[OpenAIMessage]
)


@dataclass(frozen=True)
class ResponsesRenderResult:
    messages: ResponsesPreviousMessages
    engine_input: EngineInput


def _extract_allowed_tools_from_mcp_requests(
    tools: list[Tool],
) -> dict[str, list[str] | None]:
    allowed_tools_map: dict[str, list[str] | None] = {}
    for tool in tools:
        if not isinstance(tool, Mcp):
            continue

        allowed_tools = None
        if isinstance(tool.allowed_tools, list):
            allowed_tools = tool.allowed_tools
        elif tool.allowed_tools is not None:
            allowed_tools = tool.allowed_tools.tool_names
        if allowed_tools is not None and "*" in allowed_tools:
            allowed_tools = None
        allowed_tools_map[tool.server_label] = allowed_tools
    return allowed_tools_map


def _reused_prompt_token_ids(request: Any) -> list[int] | None:
    """Pop prompt token ids forwarded for decode-side reuse, if any.

    Disaggregated serving carries the prefill stage's ids in
    ``kv_transfer_params`` so the decode stage can skip re-tokenizing. Removing
    the key keeps the id list out of the engine's sampling metadata.
    """
    kv = getattr(request, "kv_transfer_params", None)
    if not isinstance(kv, dict):
        return None
    return kv.pop("prompt_token_ids", None) or None


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

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    async def render_chat(
        self,
        request: ChatCompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.

        Decode-side token reuse (ids forwarded in ``kv_transfer_params``) is
        handled deeper, in ``preprocess_chat`` / ``_make_request_with_harmony``,
        so it skips only templating and tokenization while tool-choice
        validation and ``adjust_request`` still run and the output is
        detokenized (text-out).
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
                if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
                    tool_choice_desc = f'function "{request.tool_choice.function.name}"'
                else:
                    tool_choice_desc = f'"{request.tool_choice}"'
                return self.create_error_response(
                    f"tool_choice={tool_choice_desc} requires "
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
            if self.parser is not None:
                # HarmonyParser doesn't need chat_template_kwargs
                # TODO: Unify adjust_request() call with non-harmony branch
                self.parser(
                    self.renderer.get_tokenizer(),
                    request.tools,
                    model_config=self.model_config,
                ).adjust_request(request=request)

            should_include_tools = tool_dicts is not None
            conversation, engine_inputs = self._make_request_with_harmony(
                request, should_include_tools
            )

        return conversation, engine_inputs

    async def render_responses(
        self,
        request: ResponsesRequest,
        *,
        previous_messages: ResponsesPreviousMessages | None = None,
        previous_response_outputs: list[ResponseOutputItem] | None = None,
        tool_server: "ToolServer | None" = None,
        skip_mm_cache: bool = False,
    ) -> ResponsesRenderResult | ErrorResponse:
        """Render a Responses request using only explicitly supplied history."""
        if self.use_harmony:
            return self._render_responses_with_harmony(
                request,
                previous_messages=previous_messages,
                previous_response_outputs=previous_response_outputs,
                tool_server=tool_server,
            )

        if previous_messages is not None and any(
            isinstance(message, OpenAIMessage) for message in previous_messages
        ):
            return self.create_error_response(
                "Non-Harmony Responses history must use chat messages.",
                err_type="invalid_request_error",
                param="previous_response_id",
            )

        tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=list(previous_messages) if previous_messages is not None else None,
            prev_response_output=(
                list(previous_response_outputs)
                if previous_response_outputs is not None
                else None
            ),
        )
        chat_template_kwargs = (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )
        _, engine_inputs = await self.preprocess_chat(
            request,
            messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=self.parser,
            skip_mm_cache=skip_mm_cache,
        )
        return self._responses_render_result(messages, engine_inputs)

    def _render_responses_with_harmony(
        self,
        request: ResponsesRequest,
        *,
        previous_messages: ResponsesPreviousMessages | None,
        previous_response_outputs: list[ResponseOutputItem] | None,
        tool_server: "ToolServer | None",
    ) -> ResponsesRenderResult | ErrorResponse:
        if request.tool_choice not in ("auto", "none"):
            return self.create_error_response(
                "Only 'auto' or 'none' tool_choice is supported "
                "in response API with Harmony",
                err_type="invalid_request_error",
                param="tool_choice",
            )
        if previous_messages is not None and any(
            not isinstance(message, OpenAIMessage) for message in previous_messages
        ):
            return self.create_error_response(
                "Harmony Responses history must use Harmony messages.",
                err_type="invalid_request_error",
                param="previous_response_id",
            )

        messages: list[OpenAIMessage] = []
        request_input = request.input
        if previous_messages is None:
            tool_types = extract_tool_types(request.tools)
            with_custom_tools = has_custom_tools(tool_types)
            instructions = request.instructions
            if instructions is None and isinstance(request_input, list):
                instructions, request_input = extract_instructions_from_messages(
                    request_input
                )
            descriptions = self._get_harmony_builtin_tool_descriptions(
                request.tools,
                tool_types,
                tool_server,
            )
            messages.extend(
                build_harmony_preamble(
                    instructions=instructions,
                    tools=request.tools if with_custom_tools else None,
                    reasoning_effort=(
                        request.reasoning.effort if request.reasoning else None
                    ),
                    with_custom_tools=with_custom_tools,
                    **descriptions,
                )
            )
            messages.extend(construct_harmony_previous_input_messages(request))
        else:
            messages.extend(previous_messages)

        previous_outputs = list(previous_response_outputs or ())
        try:
            if isinstance(request_input, str):
                if request_input or not request.previous_input_messages:
                    messages.append(get_user_message(request_input))
            else:
                for response_message in request_input:
                    new_message = response_input_to_harmony(
                        response_message,
                        previous_outputs,
                    )
                    if new_message is not None:
                        messages.append(new_message)
                    if isinstance(response_message, ResponseFunctionToolCall):
                        previous_outputs.append(response_message)
        except ValueError as exc:
            return self.create_error_response(
                str(exc),
                err_type="invalid_request_error",
                param="input",
            )

        return ResponsesRenderResult(
            messages=messages,
            engine_input=self.render_responses_harmony_messages(
                messages,
                cache_salt=request.cache_salt,
            ),
        )

    def _get_harmony_builtin_tool_descriptions(
        self,
        tools: list[Tool],
        tool_types: set[str],
        tool_server: "ToolServer | None",
    ) -> dict[str, ToolNamespaceConfig | None]:
        allowed_tools = _extract_allowed_tools_from_mcp_requests(tools)
        descriptions: dict[str, ToolNamespaceConfig | None] = {}
        for server_name, request_name in BUILTIN_TOOL_TO_MCP_SERVER_LABEL.items():
            description = (
                tool_server.get_tool_description(
                    server_name,
                    allowed_tools.get(request_name),
                )
                if request_name in tool_types
                and tool_server is not None
                and tool_server.has_tool(server_name)
                else None
            )
            descriptions[f"{server_name}_description"] = description
        return descriptions

    @staticmethod
    def render_responses_harmony_messages(
        messages: list[OpenAIMessage],
        *,
        cache_salt: str | None,
    ) -> EngineInput:
        arrival_time = time.time()
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=cache_salt)
        engine_input["arrival_time"] = arrival_time
        return engine_input

    def _responses_render_result(
        self,
        messages: list[ChatCompletionMessageParam],
        engine_inputs: list[EngineInput],
    ) -> ResponsesRenderResult | ErrorResponse:
        if len(engine_inputs) != 1:
            return self.create_error_response(
                f"Expected exactly 1 engine prompt, got {len(engine_inputs)}"
            )
        return ResponsesRenderResult(
            messages=messages,
            engine_input=engine_inputs[0],
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
        should_include_tools: bool = True,
    ):
        """Build Harmony (GPT-OSS) messages and engine prompt from a chat request."""
        reuse_ids = _reused_prompt_token_ids(request)
        if reuse_ids:
            # Decode-side token reuse: feed the forwarded ids straight to the
            # engine. Harmony has no adjust_request hook to preserve.
            engine_input = tokens_input(reuse_ids, cache_salt=request.cache_salt)
            return [], [engine_input]

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

        reuse_ids = _reused_prompt_token_ids(request)
        if reuse_ids:
            # Decode-side token reuse: feed the forwarded ids straight to the
            # engine, skipping templating and tokenization. ``messages`` are not
            # tokenized, so conversation is empty. The adjust_request tail below
            # still runs.
            conversation: list[ConversationMessage] = []
            engine_input = tokens_input(
                reuse_ids, cache_salt=getattr(request, "cache_salt", None)
            )
        else:
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
