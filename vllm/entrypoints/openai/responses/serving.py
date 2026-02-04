# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack
from copy import copy
from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import Final

import jinja2
from fastapi import Request
from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterToolCallParam,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    response_function_web_search,
    response_text_delta_event,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Mcp, Tool
from openai_harmony import Message as OpenAIHarmonyMessage
from pydantic import TypeAdapter

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    ErrorType,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    construct_harmony_previous_input_messages,
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    has_custom_tools,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.context import (
    ConversationContext,
    HarmonyContext,
    ParsableContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.protocol import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputOutputMessage,
    ResponseReasoningPartAddedEvent,
    ResponseReasoningPartDoneEvent,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
    construct_tool_dicts,
    extract_tool_types,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import EmbedsPrompt, TokensPrompt
from vllm.inputs.parse import get_prompt_len
from vllm.logger import init_logger
from vllm.logprobs import Logprob as SampleLogprob
from vllm.logprobs import SampleLogprobs
from vllm.outputs import CompletionOutput
from vllm.parser import ParserManager
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = init_logger(__name__)


@dataclass
class HarmonyStreamingState:
    """Mutable state for harmony streaming event processing."""

    current_content_index: int = -1
    current_output_index: int = 0
    current_item_id: str = ""
    sent_output_item_added: bool = False
    is_first_function_call_delta: bool = False

    def reset_for_new_item(self) -> None:
        """Reset state when expecting a new output item."""
        self.current_output_index += 1
        self.sent_output_item_added = False
        self.is_first_function_call_delta = False


def _extract_allowed_tools_from_mcp_requests(
    tools: list[Tool],
) -> dict[str, list[str] | None]:
    """
    Extract allowed_tools mapping from MCP tool requests.

    Returns a dictionary mapping server_label to allowed_tools list.
    Handles both list format and McpAllowedToolsMcpToolFilter object format.

    Special handling:
    - If allowed_tools is None, returns None (allows all tools)
    - If allowed_tools contains "*", returns None (allows all tools)
    - Otherwise, returns the list of specific tool names

    This function can be reused for both harmony and non-harmony MCP calls.
    """
    allowed_tools_map: dict[str, list[str] | None] = {}
    for tool in tools:
        if not isinstance(tool, Mcp):
            continue

        # allowed_tools can be a list or an object with tool_names
        # Extract the actual list of tool names
        allowed_tools_val = None
        if tool.allowed_tools is not None:
            if isinstance(tool.allowed_tools, list):
                allowed_tools_val = tool.allowed_tools
            elif hasattr(tool.allowed_tools, "tool_names"):
                # It's an McpAllowedToolsMcpToolFilter object
                allowed_tools_val = tool.allowed_tools.tool_names

        # Normalize "*" to None (both mean "allow all tools")
        if allowed_tools_val is not None and "*" in allowed_tools_val:
            allowed_tools_val = None

        allowed_tools_map[tool.server_label] = allowed_tools_val
    return allowed_tools_map


class OpenAIServingResponses(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        tool_server: ToolServer | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_log_outputs = enable_log_outputs

        # Set up the unified parser - either a unified parser or fall back to
        # separate parsers accessed through the parser interface
        self.parser = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage

        self.default_sampling_params = self.model_config.get_diff_sampling_param()

        # If False (default), the "store" option is (silently) ignored and the
        # response is not stored. If True, the response is stored in memory.
        # NOTE(woosuk): This may not be intuitive for users, as the default
        # behavior in OpenAI's Responses API is to store the response, but
        # vLLM's default behavior is not.
        self.enable_store = envs.VLLM_ENABLE_RESPONSES_API_STORE
        if self.enable_store:
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled. This may "
                "cause a memory leak since we never remove responses from "
                "the store."
            )

        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            logger.warning(
                "For gpt-oss, we ignore --enable-auto-tool-choice "
                "and always enable tool use."
            )
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Handle tool call ID type for Kimi K2 (supporting test mocking via overrides)
        hf_overrides = getattr(self.model_config, "hf_overrides", None)
        if self.model_config.hf_text_config.model_type == "kimi_k2" or (
            isinstance(hf_overrides, dict)
            and hf_overrides.get("model_type") == "kimi_k2"
        ):
            self.tool_call_id_type = "kimi_k2"
        else:
            self.tool_call_id_type = "random"

        self.enable_auto_tools = enable_auto_tools
        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove responses from the store.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove messages from the store.
        self.msg_store: dict[str, list[ChatCompletionMessageParam]] = {}

        # HACK(wuhang): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove events from the store.
        self.event_store: dict[
            str, tuple[deque[StreamingResponsesResponse], asyncio.Event]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

        self.tool_server = tool_server

    def _validate_generator_input(
        self,
        engine_prompt: TokensPrompt | EmbedsPrompt,
    ) -> ErrorResponse | None:
        """Add validations to the input to the generator here."""
        prompt_len = get_prompt_len(engine_prompt)
        if self.max_model_len <= prompt_len:
            error_message = (
                f"The engine prompt length {prompt_len} "
                f"exceeds the max_model_len {self.max_model_len}. "
                "Please reduce prompt."
            )
            return self.create_error_response(
                err_type=ErrorType.INVALID_REQUEST_ERROR,
                message=error_message,
                status_code=HTTPStatus.BAD_REQUEST,
                param="input",
            )

        return None

    def _validate_create_responses_input(
        self, request: ResponsesRequest
    ) -> ErrorResponse | None:
        if self.use_harmony and request.is_include_output_logprobs():
            return self.create_error_response(
                err_type=ErrorType.INVALID_REQUEST_ERROR,
                message="logprobs are not supported with gpt-oss models",
                status_code=HTTPStatus.BAD_REQUEST,
                param="logprobs",
            )
        if request.store and not self.enable_store and request.background:
            return self.create_error_response(
                err_type=ErrorType.INVALID_REQUEST_ERROR,
                message=(
                    "This vLLM engine does not support `store=True` and "
                    "therefore does not support the background mode. To "
                    "enable these features, set the environment variable "
                    "`VLLM_ENABLE_RESPONSES_API_STORE=1` when launching "
                    "the vLLM server."
                ),
                status_code=HTTPStatus.BAD_REQUEST,
                param="background",
            )
        if request.previous_input_messages and request.previous_response_id:
            return self.create_error_response(
                err_type=ErrorType.INVALID_REQUEST_ERROR,
                message="Only one of `previous_input_messages` and "
                "`previous_response_id` can be set.",
                status_code=HTTPStatus.BAD_REQUEST,
                param="previous_response_id",
            )
        return None

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Request | None = None,
    ) -> (
        AsyncGenerator[StreamingResponsesResponse, None]
        | ResponsesResponse
        | ErrorResponse
    ):
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret
        maybe_validation_error = self._validate_create_responses_input(request)
        if maybe_validation_error is not None:
            return maybe_validation_error

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.store and not self.enable_store:
            # Disable the store option.
            # NOTE(woosuk): Although returning an error is possible, we opted
            # to implicitly disable store and process the request anyway, as
            # we assume most users do not intend to actually store the response
            # (i.e., their request's `store=True` just because it's the default
            # value).
            request.store = False

        # Handle the previous response ID.
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            lora_request = self._maybe_get_adapters(request)
            model_name = self.models.model_name(lora_request)

            if self.use_harmony:
                messages, engine_prompts = self._make_request_with_harmony(
                    request, prev_response
                )
            else:
                messages, engine_prompts = await self._make_request(
                    request, prev_response
                )

        except (
            ValueError,
            TypeError,
            RuntimeError,
            jinja2.TemplateError,
            NotImplementedError,
        ) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[ConversationContext, None]] = []

        builtin_tool_list: list[str] = []
        if self.tool_server is not None:
            if self.tool_server.has_tool("browser"):
                builtin_tool_list.append("browser")
            if self.tool_server.has_tool("python"):
                builtin_tool_list.append("python")
            if self.tool_server.has_tool("container"):
                builtin_tool_list.append("container")

        if self.tool_server is not None:
            available_tools = builtin_tool_list
        else:
            assert len(builtin_tool_list) == 0
            available_tools = []
        try:
            renderer = self.engine_client.renderer
            tokenizer = renderer.get_tokenizer()

            for engine_prompt in engine_prompts:
                maybe_error = self._validate_generator_input(engine_prompt)
                if maybe_error is not None:
                    return maybe_error

                default_max_tokens = get_max_tokens(
                    self.max_model_len,
                    request,
                    engine_prompt,
                    self.default_sampling_params,
                )

                sampling_params = request.to_sampling_params(
                    default_max_tokens, self.default_sampling_params
                )
                tok_params = request.build_tok_params(self.model_config)

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                context: ConversationContext
                if self.use_harmony:
                    if request.stream:
                        context = StreamingHarmonyContext(messages, available_tools)
                    else:
                        context = HarmonyContext(messages, available_tools)
                else:
                    if envs.VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT:
                        # This is a feature in development for parsing
                        # tokens during generation instead of at the end
                        context = ParsableContext(
                            response_messages=messages,
                            tokenizer=tokenizer,
                            reasoning_parser_cls=self.parser.reasoning_parser_cls
                            if self.parser
                            else None,
                            request=request,
                            tool_parser_cls=self.parser.tool_parser_cls
                            if self.parser
                            else None,
                            available_tools=available_tools,
                            chat_template=self.chat_template,
                            chat_template_content_format=self.chat_template_content_format,
                        )
                    else:
                        context = SimpleContext()

                if self.parser and self.parser.reasoning_parser_cls is not None:
                    reasoning_parser = self.parser.reasoning_parser_cls(tokenizer)
                    if (
                        isinstance(
                            struct_out := sampling_params.structured_outputs,
                            StructuredOutputsParams,
                        )
                        and struct_out.all_non_structural_tag_constraints_none()
                    ):
                        sampling_params.structured_outputs = replace(
                            struct_out,
                            structural_tag=reasoning_parser.prepare_structured_tag(
                                struct_out.structural_tag, self.tool_server
                            ),
                        )
                generator = self._generate_with_builtin_tools(
                    request_id=request.request_id,
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    tok_params=tok_params,
                    context=context,
                    lora_request=lora_request,
                    priority=request.priority,
                    trace_headers=trace_headers,
                )
                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        assert len(generators) == 1
        (result_generator,) = generators

        # Store the input messages.
        if request.store:
            self.msg_store[request.request_id] = messages

        if request.background:
            created_time = int(time.time())
            response = ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="queued",
                usage=None,
            )
            async with self.response_store_lock:
                self.response_store[response.id] = response

            # Run the request in the background.
            if request.stream:
                task = asyncio.create_task(
                    self._run_background_request_stream(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{request.request_id}",
                )
            else:
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{response.id}",
                )

            # For cleanup.
            response_id = response.id
            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda _: self.background_tasks.pop(response_id, None)
            )

            if request.stream:
                return self.responses_background_stream_generator(request.request_id)
            return response

        if request.stream:
            return self.responses_stream_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )

        try:
            return await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except Exception as e:
            return self.create_error_response(e)

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
    ):
        tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
        # Construct the input messages.
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=self.msg_store.get(prev_response.id) if prev_response else None,
            prev_response_output=prev_response.output if prev_response else None,
        )

        _, engine_prompts = await self._preprocess_chat(
            request,
            messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=None,
            tool_dicts=tool_dicts,
            tool_parser=self.parser.tool_parser_cls if self.parser else None,
        )
        return messages, engine_prompts

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in response API with Harmony"
            )

        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [engine_prompt]

    async def _initialize_tool_sessions(
        self,
        request: ResponsesRequest,
        context: ConversationContext,
        exit_stack: AsyncExitStack,
    ):
        # we should only initialize the tool session if the request needs tools
        if len(request.tools) == 0:
            return
        mcp_tools = {
            tool.server_label: tool for tool in request.tools if tool.type == "mcp"
        }
        await context.init_tool_sessions(
            self.tool_server, exit_stack, request.request_id, mcp_tools
        )

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
    ) -> ErrorResponse | ResponsesResponse:
        if created_time is None:
            created_time = int(time.time())

        async with AsyncExitStack() as exit_stack:
            try:
                await self._initialize_tool_sessions(request, context, exit_stack)
                async for _ in result_generator:
                    pass
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")
            except ValueError as e:
                return self.create_error_response(e)

        # NOTE: Implementation of status is still WIP, but for now
        # we guarantee that if the status is not "completed", it is accurate.
        # "completed" is implemented as the "catch-all" for now.
        status: ResponseStatus = "completed"

        input_messages: ResponseInputOutputMessage | None = None
        output_messages: ResponseInputOutputMessage | None = None
        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            if request.enable_response_messages:
                input_messages = context.messages[: context.num_init_messages]
                output_messages = context.messages[context.num_init_messages :]
            num_tool_output_tokens = context.num_tool_output_tokens
            if len(output) > 0:
                if context.finish_reason == "length":
                    status = "incomplete"
                elif context.finish_reason == "abort":
                    status = "cancelled"
                else:
                    self._raise_if_error(context.finish_reason, request.request_id)
            else:
                status = "incomplete"
        elif isinstance(context, ParsableContext):
            output = context.parser.make_response_output_items_from_parsable_context()

            if request.enable_response_messages:
                input_messages = context.input_messages
                output_messages = context.output_messages

            # TODO: Calculate usage.
            # assert final_res.prompt_token_ids is not None
            num_tool_output_tokens = 0

            # Check finish reason from the parser
            if context.parser.finish_reason == "length":
                status = "incomplete"
        else:
            assert isinstance(context, SimpleContext)
            # Use final_output which has accumulated text/token_ids/logprobs
            final_res = context.final_output
            assert final_res is not None
            assert len(final_res.outputs) == 1
            final_output = final_res.outputs[0]

            # finish_reason='error' indicates retryable internal error
            self._raise_if_error(final_output.finish_reason, request.request_id)

            # Check if generation was stopped due to max_tokens
            if final_output.finish_reason == "length":
                status = "incomplete"

            output = self._make_response_output_items(request, final_output, tokenizer)

            if request.enable_response_messages:
                input_messages = context.input_messages
                output_messages = context.output_messages

            # Calculate usage.
            assert final_res.prompt_token_ids is not None
            num_tool_output_tokens = 0

        assert isinstance(context, (SimpleContext, HarmonyContext, ParsableContext))
        num_prompt_tokens = context.num_prompt_tokens
        num_generated_tokens = context.num_output_tokens
        num_cached_tokens = context.num_cached_tokens
        num_reasoning_tokens = context.num_reasoning_tokens

        usage = ResponseUsage(
            input_tokens=num_prompt_tokens,
            output_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=num_cached_tokens,
                input_tokens_per_turn=[
                    turn.input_tokens for turn in context.all_turn_metrics
                ],
                cached_tokens_per_turn=[
                    turn.cached_input_tokens for turn in context.all_turn_metrics
                ],
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=num_reasoning_tokens,
                tool_output_tokens=num_tool_output_tokens,
                output_tokens_per_turn=[
                    turn.output_tokens for turn in context.all_turn_metrics
                ],
                tool_output_tokens_per_turn=[
                    turn.tool_output_tokens for turn in context.all_turn_metrics
                ],
            ),
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            input_messages=input_messages,
            output_messages=output_messages,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status=status,
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it.
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response
        return response

    def _is_mcp_tool_by_namespace(self, recipient: str | None) -> bool:
        """
        Determine if a tool call is an MCP tool based on recipient prefix.

        - Tools starting with "functions." are function calls
        - Everything else is an MCP tool
        """
        if recipient is None:
            return False

        # Function calls have "functions." prefix
        # Everything else is an MCP tool
        return not recipient.startswith("functions.")

    _TOOL_NAME_TO_MCP_SERVER_LABEL: Final[dict[str, str]] = {
        "python": "code_interpreter",
        "container": "container",
        "browser": "web_search_preview",
    }

    def _topk_logprobs(
        self,
        logprobs: dict[int, SampleLogprob],
        top_logprobs: int,
        tokenizer: TokenizerLike,
    ) -> list[LogprobTopLogprob]:
        """Returns the top-k logprobs from the logprobs dictionary."""
        out = []
        for i, (token_id, _logprob) in enumerate(logprobs.items()):
            if i >= top_logprobs:
                break
            text = self._get_decoded_token(
                logprob=_logprob,
                token_id=token_id,
                tokenizer=tokenizer,
                return_as_token_id=self.return_tokens_as_token_ids,
            )
            out.append(
                LogprobTopLogprob(
                    token=text,
                    logprob=max(_logprob.logprob, -9999.0),
                    bytes=list(text.encode("utf-8", errors="replace")),
                )
            )
        return out

    def _create_response_logprobs(
        self,
        token_ids: Sequence[int],
        logprobs: SampleLogprobs | None,
        tokenizer: TokenizerLike,
        top_logprobs: int | None = None,
    ) -> list[Logprob]:
        assert logprobs is not None, "logprobs must be provided"
        assert len(token_ids) == len(logprobs), (
            "token_ids and logprobs.token_ids must have the same length"
        )
        out = []
        for i, token_id in enumerate(token_ids):
            logprob = logprobs[i]
            token_logprob = logprob[token_id]
            text = self._get_decoded_token(
                logprob=token_logprob,
                token_id=token_id,
                tokenizer=tokenizer,
                return_as_token_id=self.return_tokens_as_token_ids,
            )
            out.append(
                Logprob(
                    token=text,
                    logprob=max(token_logprob.logprob, -9999.0),
                    bytes=list(text.encode("utf-8", errors="replace")),
                    top_logprobs=(
                        self._topk_logprobs(
                            logprob, top_logprobs=top_logprobs, tokenizer=tokenizer
                        )
                        if top_logprobs
                        else []
                    ),
                )
            )
        return out

    def _create_stream_response_logprobs(
        self,
        token_ids: Sequence[int],
        logprobs: SampleLogprobs | None,
        tokenizer: TokenizerLike,
        top_logprobs: int | None = None,
    ) -> list[response_text_delta_event.Logprob]:
        lgs = self._create_response_logprobs(
            token_ids=token_ids,
            logprobs=logprobs,
            tokenizer=tokenizer,
            top_logprobs=top_logprobs,
        )
        return [
            response_text_delta_event.Logprob(
                token=lg.token,
                logprob=lg.logprob,
                top_logprobs=[
                    response_text_delta_event.LogprobTopLogprob(
                        token=tl.token, logprob=tl.logprob
                    )
                    for tl in lg.top_logprobs
                ],
            )
            for lg in lgs
        ]

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: CompletionOutput,
        tokenizer: TokenizerLike,
    ) -> list[ResponseOutputItem]:
        if self.parser and self.parser.reasoning_parser_cls:
            try:
                reasoning_parser = self.parser.reasoning_parser_cls(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                raise e

            reasoning, content = reasoning_parser.extract_reasoning(
                final_output.text, request=request
            )
        else:
            reasoning = None
            content = final_output.text

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            output_text = ""
            if content:
                output_text = content
            elif reasoning:
                output_text = f"[reasoning: {reasoning}]"

            if output_text:
                self.request_logger.log_outputs(
                    request_id=request.request_id,
                    outputs=output_text,
                    output_token_ids=final_output.token_ids,
                    finish_reason=final_output.finish_reason,
                    is_streaming=False,
                    delta=False,
                )

        reasoning_item = None
        message_item = None
        if reasoning:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,  # NOTE: Only the last output item has status.
            )
        tool_calls, content = self._parse_tool_calls_from_content(
            request=request,
            tokenizer=tokenizer,
            content=content,
            enable_auto_tools=self.enable_auto_tools,
            tool_parser_cls=self.parser.tool_parser_cls if self.parser else None,
        )

        if content or (self.use_harmony and tool_calls):
            res_text_part = None
            if content:
                res_text_part = ResponseOutputText(
                    text=content,
                    annotations=[],  # TODO
                    type="output_text",
                    logprobs=(
                        self._create_response_logprobs(
                            token_ids=final_output.token_ids,
                            logprobs=final_output.logprobs,
                            tokenizer=tokenizer,
                            top_logprobs=request.top_logprobs,
                        )
                        if request.is_include_output_logprobs()
                        else None
                    ),
                )
            message_item = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[res_text_part] if res_text_part else [],
                role="assistant",
                status="completed",
                type="message",
            )
        outputs = []

        if reasoning_item:
            outputs.append(reasoning_item)
        if message_item:
            outputs.append(message_item)
        if tool_calls:
            # We use a simple counter for history_tool_call_count because
            # we don't track the history of tool calls in the Responses API yet.
            # This means that the tool call index will start from 0 for each
            # request.
            tool_call_items = []
            for history_tool_call_cnt, tool_call in enumerate(tool_calls):
                tool_call_items.append(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=tool_call.id
                        if tool_call.id
                        else make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tool_call.name,
                            idx=history_tool_call_cnt,
                        ),
                        type="function_call",
                        status="completed",
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )
            outputs.extend(tool_call_items)
        return outputs

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ) -> list[ResponseOutputItem]:
        output_items: list[ResponseOutputItem] = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _extract_system_message_from_request(self, request) -> str | None:
        system_msg = None
        if not isinstance(request.input, str):
            for response_msg in request.input:
                if (
                    isinstance(response_msg, dict)
                    and response_msg.get("role") == "system"
                ):
                    system_msg = response_msg.get("content")
                    break
        return system_msg

    def _construct_harmony_system_input_message(
        self, request: ResponsesRequest, with_custom_tools: bool, tool_types: set[str]
    ) -> OpenAIHarmonyMessage:
        model_identity = self._extract_system_message_from_request(request)

        reasoning_effort = request.reasoning.effort if request.reasoning else None

        # Extract allowed_tools from MCP tool requests
        allowed_tools_map = _extract_allowed_tools_from_mcp_requests(request.tools)

        # Get filtered tool descriptions first.
        # If get_tool_description returns None (due to filtering), the tool is disabled.
        browser_description = (
            self.tool_server.get_tool_description(
                "browser", allowed_tools_map.get("web_search_preview")
            )
            if "web_search_preview" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("browser")
            else None
        )
        python_description = (
            self.tool_server.get_tool_description(
                "python", allowed_tools_map.get("code_interpreter")
            )
            if "code_interpreter" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("python")
            else None
        )
        container_description = (
            self.tool_server.get_tool_description(
                "container", allowed_tools_map.get("container")
            )
            if "container" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("container")
            else None
        )

        sys_msg = get_system_message(
            model_identity=model_identity,
            reasoning_effort=reasoning_effort,
            browser_description=browser_description,
            python_description=python_description,
            container_description=container_description,
            instructions=request.instructions,
            with_custom_tools=with_custom_tools,
        )
        return sys_msg

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
    ) -> list[OpenAIHarmonyMessage]:
        messages: list[OpenAIHarmonyMessage] = []
        if prev_response is None:
            # New conversation.
            tool_types = extract_tool_types(request.tools)
            with_custom_tools = has_custom_tools(tool_types)

            sys_msg = self._construct_harmony_system_input_message(
                request, with_custom_tools, tool_types
            )
            messages.append(sys_msg)
            if with_custom_tools:
                dev_msg = get_developer_message(
                    instructions=request.instructions, tools=request.tools
                )
                messages.append(dev_msg)
            messages += construct_harmony_previous_input_messages(request)

        else:
            # Continue the previous conversation.
            # FIXME(woosuk): Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message. Note that this also removes these messages from the
            # msg_store.
            if len(prev_msgs) > 0:
                last_msg = prev_msgs[-1]
                assert isinstance(last_msg, OpenAIHarmonyMessage)
                if last_msg.channel == "final":
                    prev_final_msg_idx = -1
                    for i in range(len(prev_msgs) - 2, -1, -1):
                        prev_msg_i = prev_msgs[i]
                        assert isinstance(prev_msg_i, OpenAIHarmonyMessage)
                        if prev_msg_i.channel == "final":
                            prev_final_msg_idx = i
                            break
                    recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                    del prev_msgs[prev_final_msg_idx + 1 :]
                    for msg in recent_turn_msgs:
                        assert isinstance(msg, OpenAIHarmonyMessage)
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                new_msg = parse_response_input(response_msg, prev_outputs)
                if new_msg.author.role != "system":
                    messages.append(new_msg)

                # User passes in a tool call request and its output. We need
                # to add the tool call request to prev_outputs so that the
                # parse_response_input can find the tool call request when
                # parsing the tool call output.
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request_stream(
        self,
        request: ResponsesRequest,
        *args,
        **kwargs,
    ):
        event_deque: deque[StreamingResponsesResponse] = deque()
        new_event_signal = asyncio.Event()
        self.event_store[request.request_id] = (event_deque, new_event_signal)
        response = None
        try:
            generator = self.responses_stream_generator(request, *args, **kwargs)
            async for event in generator:
                event_deque.append(event)
                new_event_signal.set()  # Signal new event available
        except GenerationError as e:
            response = self._convert_generation_error_to_response(e)
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(e)
        finally:
            new_event_signal.set()

        if response is not None and isinstance(response, ErrorResponse):
            # If the request has failed, update the status to "failed".
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        *args,
        **kwargs,
    ):
        try:
            response = await self.responses_full_generator(request, *args, **kwargs)
        except GenerationError as e:
            response = self._convert_generation_error_to_response(e)
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(e)

        if isinstance(response, ErrorResponse):
            # If the request has failed, update the status to "failed".
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def responses_background_stream_generator(
        self,
        response_id: str,
        starting_after: int | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        if response_id not in self.event_store:
            raise VLLMValidationError(
                f"Unknown response_id: {response_id}",
                parameter="response_id",
                value=response_id,
            )

        event_deque, new_event_signal = self.event_store[response_id]
        start_index = 0 if starting_after is None else starting_after + 1
        current_index = start_index

        while True:
            new_event_signal.clear()

            # Yield existing events from start_index
            while current_index < len(event_deque):
                event = event_deque[current_index]
                yield event
                if getattr(event, "type", "unknown") == "response.completed":
                    return
                current_index += 1

            await new_event_signal.wait()

    async def retrieve_responses(
        self,
        response_id: str,
        starting_after: int | None,
        stream: bool | None,
    ) -> (
        ErrorResponse
        | ResponsesResponse
        | AsyncGenerator[StreamingResponsesResponse, None]
    ):
        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)

        if stream:
            return self.responses_background_stream_generator(
                response_id,
                starting_after,
            )
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> ErrorResponse | ResponsesResponse:
        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type=ErrorType.INVALID_REQUEST_ERROR,
                    message="Cannot cancel a synchronous response.",
                    param="response_id",
                )

            # Update the status to "cancelled".
            response.status = "cancelled"

        # Abort the request.
        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_not_found_error(self, response_id: str) -> ErrorResponse:
        return self.create_error_response(
            err_type=ErrorType.INVALID_REQUEST_ERROR,
            message=f"Response with id '{response_id}' not found.",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    def _make_store_not_supported_error(self) -> ErrorResponse:
        return self.create_error_response(
            err_type=ErrorType.INVALID_REQUEST_ERROR,
            message=(
                "`store=True` (default) is not supported. Please set "
                "`store=False` in Responses API or set "
                "`VLLM_ENABLE_RESPONSES_API_STORE=1` in the env var when "
                "starting the vLLM server."
            ),
            status_code=HTTPStatus.BAD_REQUEST,
            param="store",
        )

    async def _process_simple_streaming_events(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int,
        _increment_sequence_number_and_return: Callable[
            [StreamingResponsesResponse], StreamingResponsesResponse
        ],
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        current_content_index = 0
        current_output_index = 0
        current_item_id = ""
        reasoning_parser = None
        if self.parser and self.parser.reasoning_parser_cls:
            reasoning_parser = self.parser.reasoning_parser_cls(tokenizer)
        previous_text = ""
        previous_token_ids: list[int] = []
        first_delta_sent = False
        previous_delta_messages: list[DeltaMessage] = []
        async for ctx in result_generator:
            assert isinstance(ctx, SimpleContext)
            if ctx.last_output is None:
                continue
            if ctx.last_output.outputs:
                output = ctx.last_output.outputs[0]
                # finish_reason='error' indicates a retryable error
                self._raise_if_error(output.finish_reason, request.request_id)
                if reasoning_parser:
                    delta_message = reasoning_parser.extract_reasoning_streaming(
                        previous_text=previous_text,
                        current_text=previous_text + output.text,
                        delta_text=output.text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=previous_token_ids + output.token_ids,
                        delta_token_ids=output.token_ids,
                    )
                else:
                    delta_message = DeltaMessage(
                        content=output.text,
                    )
                previous_text += output.text
                previous_token_ids += output.token_ids
                if not delta_message:
                    continue
                if not first_delta_sent:
                    current_item_id = str(uuid.uuid4())
                    if delta_message.reasoning:
                        yield _increment_sequence_number_and_return(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                    else:
                        yield _increment_sequence_number_and_return(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                    yield _increment_sequence_number_and_return(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            content_index=current_content_index,
                            part=ResponseOutputText(
                                type="output_text",
                                text="",
                                annotations=[],
                                logprobs=[],
                            ),
                        )
                    )
                    current_content_index += 1
                    first_delta_sent = True
                # todo(kebe7jun) tool call support

                # check delta message and previous delta message are
                # same as content or reasoning content
                if (
                    previous_delta_messages
                    and previous_delta_messages[-1].reasoning is not None
                    and delta_message.content is not None
                ):
                    # from reasoning to normal content, send done
                    # event for reasoning
                    reason_content = "".join(
                        pm.reasoning
                        for pm in previous_delta_messages
                        if pm.reasoning is not None
                    )
                    yield _increment_sequence_number_and_return(
                        ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=current_item_id,
                            sequence_number=-1,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            text=reason_content,
                        )
                    )
                    current_content_index = 0
                    reasoning_item = ResponseReasoningItem(
                        type="reasoning",
                        content=[
                            ResponseReasoningTextContent(
                                text=reason_content,
                                type="reasoning_text",
                            ),
                        ],
                        status="completed",
                        id=current_item_id,
                        summary=[],
                    )
                    yield _increment_sequence_number_and_return(
                        ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=reasoning_item,
                        )
                    )
                    yield _increment_sequence_number_and_return(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=ResponseOutputMessage(
                                id=current_item_id,
                                type="message",
                                role="assistant",
                                content=[],
                                status="in_progress",
                            ),
                        )
                    )
                    current_output_index += 1
                    current_item_id = str(uuid.uuid4())
                    yield _increment_sequence_number_and_return(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            content_index=current_content_index,
                            part=ResponseOutputText(
                                type="output_text",
                                text="",
                                annotations=[],
                                logprobs=[],
                            ),
                        )
                    )
                    current_content_index += 1
                    # reset previous delta messages
                    previous_delta_messages = []

                if delta_message.reasoning is not None:
                    yield _increment_sequence_number_and_return(
                        ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=delta_message.reasoning,
                        )
                    )
                elif delta_message.content is not None:
                    yield _increment_sequence_number_and_return(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=delta_message.content,
                            logprobs=(
                                self._create_stream_response_logprobs(
                                    token_ids=output.token_ids,
                                    logprobs=output.logprobs,
                                    tokenizer=tokenizer,
                                    top_logprobs=request.top_logprobs,
                                )
                                if request.is_include_output_logprobs()
                                else []
                            ),
                        )
                    )
                current_content_index += 1

                previous_delta_messages.append(delta_message)
        if previous_delta_messages:
            if previous_delta_messages[-1].reasoning is not None:
                reason_content = "".join(
                    pm.reasoning
                    for pm in previous_delta_messages
                    if pm.reasoning is not None
                )
                yield _increment_sequence_number_and_return(
                    ResponseReasoningTextDoneEvent(
                        type="response.reasoning_text.done",
                        item_id=current_item_id,
                        sequence_number=-1,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        text=reason_content,
                    )
                )
                current_content_index += 1
                reasoning_item = ResponseReasoningItem(
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=reason_content,
                            type="reasoning_text",
                        ),
                    ],
                    status="completed",
                    id=current_item_id,
                    summary=[],
                )
                yield _increment_sequence_number_and_return(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item=reasoning_item,
                    )
                )
            elif previous_delta_messages[-1].content is not None:
                final_content = "".join(
                    pm.content
                    for pm in previous_delta_messages
                    if pm.content is not None
                )
                yield _increment_sequence_number_and_return(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        text=final_content,
                        logprobs=[],
                        item_id=current_item_id,
                    )
                )
                current_content_index += 1
                part = ResponseOutputText(
                    text=final_content,
                    type="output_text",
                    annotations=[],
                )
                yield _increment_sequence_number_and_return(
                    ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=current_item_id,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        part=part,
                    )
                )
                current_content_index += 1
                item = ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[
                        part,
                    ],
                    status="completed",
                    id=current_item_id,
                    summary=[],
                )
                yield _increment_sequence_number_and_return(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item=item,
                    )
                )

    def _emit_function_call_done_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when a function call completes."""
        function_name = previous_item.recipient[len("functions.") :]
        events = []
        events.append(
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                arguments=previous_item.content[0].text,
                name=function_name,
                item_id=state.current_item_id,
                output_index=state.current_output_index,
                sequence_number=-1,
            )
        )
        function_call_item = ResponseFunctionToolCall(
            type="function_call",
            arguments=previous_item.content[0].text,
            name=function_name,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            sequence_number=-1,
            call_id=f"fc_{random_uuid()}",
            status="completed",
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=function_call_item,
            )
        )
        return events

    def _emit_mcp_call_done_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when an MCP tool call completes."""
        server_label = self._TOOL_NAME_TO_MCP_SERVER_LABEL.get(
            previous_item.recipient, previous_item.recipient
        )
        events = []
        events.append(
            ResponseMcpCallArgumentsDoneEvent(
                type="response.mcp_call_arguments.done",
                arguments=previous_item.content[0].text,
                name=previous_item.recipient,
                item_id=state.current_item_id,
                output_index=state.current_output_index,
                sequence_number=-1,
            )
        )
        events.append(
            ResponseMcpCallCompletedEvent(
                type="response.mcp_call.completed",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=McpCall(
                    type="mcp_call",
                    arguments=previous_item.content[0].text,
                    name=previous_item.recipient,
                    id=state.current_item_id,
                    server_label=server_label,
                    status="completed",
                ),
            )
        )
        return events

    def _emit_reasoning_done_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when a reasoning (analysis) item completes."""
        content = ResponseReasoningTextContent(
            text=previous_item.content[0].text,
            type="reasoning_text",
        )
        reasoning_item = ResponseReasoningItem(
            type="reasoning",
            content=[content],
            status="completed",
            id=state.current_item_id,
            summary=[],
        )
        events = []
        events.append(
            ResponseReasoningTextDoneEvent(
                type="response.reasoning_text.done",
                item_id=state.current_item_id,
                sequence_number=-1,
                output_index=state.current_output_index,
                content_index=state.current_content_index,
                text=previous_item.content[0].text,
            )
        )
        events.append(
            ResponseReasoningPartDoneEvent(
                type="response.reasoning_part.done",
                sequence_number=-1,
                item_id=state.current_item_id,
                output_index=state.current_output_index,
                content_index=state.current_content_index,
                part=content,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=reasoning_item,
            )
        )
        return events

    def _emit_text_output_done_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when a final text output item completes."""
        text_content = ResponseOutputText(
            type="output_text",
            text=previous_item.content[0].text,
            annotations=[],
        )
        events = []
        events.append(
            ResponseTextDoneEvent(
                type="response.output_text.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                content_index=state.current_content_index,
                text=previous_item.content[0].text,
                logprobs=[],
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseContentPartDoneEvent(
                type="response.content_part.done",
                sequence_number=-1,
                item_id=state.current_item_id,
                output_index=state.current_output_index,
                content_index=state.current_content_index,
                part=text_content,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=ResponseOutputMessage(
                    id=state.current_item_id,
                    type="message",
                    role="assistant",
                    content=[text_content],
                    status="completed",
                ),
            )
        )
        return events

    def _emit_previous_item_done_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit done events for the previous item when expecting a new start."""
        if previous_item.recipient is not None:
            # Deal with tool call
            if previous_item.recipient.startswith("functions."):
                return self._emit_function_call_done_events(previous_item, state)
            elif (
                self._is_mcp_tool_by_namespace(previous_item.recipient)
                and state.current_item_id is not None
                and state.current_item_id.startswith("mcp_")
            ):
                return self._emit_mcp_call_done_events(previous_item, state)
        elif previous_item.channel == "analysis":
            return self._emit_reasoning_done_events(previous_item, state)
        elif previous_item.channel == "final":
            return self._emit_text_output_done_events(previous_item, state)
        return []

    def _emit_final_channel_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for final channel text delta streaming."""
        events = []
        if not state.sent_output_item_added:
            state.sent_output_item_added = True
            state.current_item_id = f"msg_{random_uuid()}"
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=ResponseOutputMessage(
                        id=state.current_item_id,
                        type="message",
                        role="assistant",
                        content=[],
                        status="in_progress",
                    ),
                )
            )
            state.current_content_index += 1
            events.append(
                ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item_id=state.current_item_id,
                    content_index=state.current_content_index,
                    part=ResponseOutputText(
                        type="output_text",
                        text="",
                        annotations=[],
                        logprobs=[],
                    ),
                )
            )
        events.append(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=-1,
                content_index=state.current_content_index,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                delta=ctx.last_content_delta,
                # TODO, use logprobs from ctx.last_request_output
                logprobs=[],
            )
        )
        return events

    def _emit_analysis_channel_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for analysis channel reasoning delta streaming."""
        events = []
        if not state.sent_output_item_added:
            state.sent_output_item_added = True
            state.current_item_id = f"msg_{random_uuid()}"
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=ResponseReasoningItem(
                        type="reasoning",
                        id=state.current_item_id,
                        summary=[],
                        status="in_progress",
                    ),
                )
            )
            state.current_content_index += 1
            events.append(
                ResponseReasoningPartAddedEvent(
                    type="response.reasoning_part.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item_id=state.current_item_id,
                    content_index=state.current_content_index,
                    part=ResponseReasoningTextContent(
                        text="",
                        type="reasoning_text",
                    ),
                )
            )
        events.append(
            ResponseReasoningTextDeltaEvent(
                type="response.reasoning_text.delta",
                item_id=state.current_item_id,
                output_index=state.current_output_index,
                content_index=state.current_content_index,
                delta=ctx.last_content_delta,
                sequence_number=-1,
            )
        )
        return events

    def _emit_mcp_tool_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
        recipient: str,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for MCP tool delta streaming."""
        server_label = self._TOOL_NAME_TO_MCP_SERVER_LABEL.get(recipient, recipient)
        events = []
        if not state.sent_output_item_added:
            state.sent_output_item_added = True
            state.current_item_id = f"mcp_{random_uuid()}"
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=McpCall(
                        type="mcp_call",
                        id=state.current_item_id,
                        name=recipient,
                        arguments="",
                        server_label=server_label,
                        status="in_progress",
                    ),
                )
            )
            events.append(
                ResponseMcpCallInProgressEvent(
                    type="response.mcp_call.in_progress",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item_id=state.current_item_id,
                )
            )
        events.append(
            ResponseMcpCallArgumentsDeltaEvent(
                type="response.mcp_call_arguments.delta",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                delta=ctx.last_content_delta,
            )
        )
        return events

    def _emit_code_interpreter_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for code interpreter delta streaming."""
        events = []
        if not state.sent_output_item_added:
            state.sent_output_item_added = True
            state.current_item_id = f"tool_{random_uuid()}"
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=ResponseCodeInterpreterToolCallParam(
                        type="code_interpreter_call",
                        id=state.current_item_id,
                        code=None,
                        container_id="auto",
                        outputs=None,
                        status="in_progress",
                    ),
                )
            )
            events.append(
                ResponseCodeInterpreterCallInProgressEvent(
                    type="response.code_interpreter_call.in_progress",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item_id=state.current_item_id,
                )
            )
        events.append(
            ResponseCodeInterpreterCallCodeDeltaEvent(
                type="response.code_interpreter_call_code.delta",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                delta=ctx.last_content_delta,
            )
        )
        return events

    def _emit_mcp_prefix_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for MCP prefix (mcp.*) delta streaming."""
        events = []
        if not state.sent_output_item_added:
            state.sent_output_item_added = True
            state.current_item_id = f"mcp_{random_uuid()}"
            mcp_name = ctx.parser.current_recipient[len("mcp.") :]

            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=McpCall(
                        type="mcp_call",
                        id=state.current_item_id,
                        name=mcp_name,
                        arguments="",
                        server_label=mcp_name,
                        status="in_progress",
                    ),
                )
            )
            events.append(
                ResponseMcpCallInProgressEvent(
                    type="response.mcp_call.in_progress",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item_id=state.current_item_id,
                )
            )

        events.append(
            ResponseMcpCallArgumentsDeltaEvent(
                type="response.mcp_call_arguments.delta",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                delta=ctx.last_content_delta,
            )
        )
        return events

    def _emit_content_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for content delta streaming based on channel type."""
        if not ctx.last_content_delta:
            return []

        if (
            ctx.parser.current_channel == "final"
            and ctx.parser.current_recipient is None
        ):
            return self._emit_final_channel_delta_events(ctx, state)
        elif (
            ctx.parser.current_channel == "analysis"
            and ctx.parser.current_recipient is None
        ):
            return self._emit_analysis_channel_delta_events(ctx, state)
        # built-in tools will be triggered on the analysis channel
        # However, occasionally built-in tools will
        # still be output to commentary.
        elif (
            ctx.parser.current_channel == "commentary"
            or ctx.parser.current_channel == "analysis"
        ) and ctx.parser.current_recipient is not None:
            recipient = ctx.parser.current_recipient
            # Check for function calls first - they have their own event handling
            if recipient.startswith("functions."):
                return self._emit_function_call_delta_events(ctx, state)
            is_mcp_tool = self._is_mcp_tool_by_namespace(recipient)
            if is_mcp_tool:
                return self._emit_mcp_tool_delta_events(ctx, state, recipient)
            else:
                return self._emit_code_interpreter_delta_events(ctx, state)
        elif (
            (
                ctx.parser.current_channel == "commentary"
                or ctx.parser.current_channel == "analysis"
            )
            and ctx.parser.current_recipient is not None
            and ctx.parser.current_recipient.startswith("mcp.")
        ):
            return self._emit_mcp_prefix_delta_events(ctx, state)

        return []

    def _emit_browser_tool_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for browser tool calls (web search)."""
        function_name = previous_item.recipient[len("browser.") :]
        parsed_args = json.loads(previous_item.content[0].text)
        action = None

        if function_name == "search":
            action = response_function_web_search.ActionSearch(
                type="search",
                query=parsed_args["query"],
            )
        elif function_name == "open":
            action = response_function_web_search.ActionOpenPage(
                type="open_page",
                # TODO: translate to url
                url=f"cursor:{parsed_args.get('cursor', '')}",
            )
        elif function_name == "find":
            action = response_function_web_search.ActionFind(
                type="find",
                pattern=parsed_args["pattern"],
                # TODO: translate to url
                url=f"cursor:{parsed_args.get('cursor', '')}",
            )
        else:
            raise ValueError(f"Unknown function name: {function_name}")

        state.current_item_id = f"tool_{random_uuid()}"
        events = []
        events.append(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=response_function_web_search.ResponseFunctionWebSearch(
                    # TODO: generate a unique id for web search call
                    type="web_search_call",
                    id=state.current_item_id,
                    action=action,
                    status="in_progress",
                ),
            )
        )
        events.append(
            ResponseWebSearchCallInProgressEvent(
                type="response.web_search_call.in_progress",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseWebSearchCallSearchingEvent(
                type="response.web_search_call.searching",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        # enqueue
        events.append(
            ResponseWebSearchCallCompletedEvent(
                type="response.web_search_call.completed",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=ResponseFunctionWebSearch(
                    type="web_search_call",
                    id=state.current_item_id,
                    action=action,
                    status="completed",
                ),
            )
        )
        return events

    def _emit_mcp_tool_completion_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when an MCP tool completes during assistant action turn."""
        recipient = previous_item.recipient
        server_label = self._TOOL_NAME_TO_MCP_SERVER_LABEL.get(recipient, recipient)
        events = []
        events.append(
            ResponseMcpCallArgumentsDoneEvent(
                type="response.mcp_call_arguments.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                arguments=previous_item.content[0].text,
                name=recipient,
            )
        )
        events.append(
            ResponseMcpCallCompletedEvent(
                type="response.mcp_call.completed",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=McpCall(
                    type="mcp_call",
                    id=state.current_item_id,
                    name=recipient,
                    arguments=previous_item.content[0].text,
                    server_label=server_label,
                    status="completed",
                ),
            )
        )
        return events

    def _emit_code_interpreter_completion_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when code interpreter completes."""
        events = []
        events.append(
            ResponseCodeInterpreterCallCodeDoneEvent(
                type="response.code_interpreter_call_code.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                code=previous_item.content[0].text,
            )
        )
        events.append(
            ResponseCodeInterpreterCallInterpretingEvent(
                type="response.code_interpreter_call.interpreting",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseCodeInterpreterCallCompletedEvent(
                type="response.code_interpreter_call.completed",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=ResponseCodeInterpreterToolCallParam(
                    type="code_interpreter_call",
                    id=state.current_item_id,
                    code=previous_item.content[0].text,
                    container_id="auto",
                    outputs=[],
                    status="completed",
                ),
            )
        )
        return events

    def _emit_mcp_prefix_completion_events(
        self,
        previous_item,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events when an MCP prefix tool (mcp.*) completes."""
        mcp_name = previous_item.recipient[len("mcp.") :]
        events = []
        events.append(
            ResponseMcpCallArgumentsDoneEvent(
                type="response.mcp_call_arguments.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                arguments=previous_item.content[0].text,
                name=mcp_name,
            )
        )
        events.append(
            ResponseMcpCallCompletedEvent(
                type="response.mcp_call.completed",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
        events.append(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=McpCall(
                    type="mcp_call",
                    id=state.current_item_id,
                    name=mcp_name,
                    arguments=previous_item.content[0].text,
                    server_label=mcp_name,
                    status="completed",
                ),
            )
        )
        return events

    def _emit_tool_action_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for tool action turn."""
        if not ctx.is_assistant_action_turn() or len(ctx.parser.messages) == 0:
            return []

        events = []
        previous_item = ctx.parser.messages[-1]

        # Handle browser tool
        if (
            self.tool_server is not None
            and self.tool_server.has_tool("browser")
            and previous_item.recipient is not None
            and previous_item.recipient.startswith("browser.")
        ):
            events.extend(self._emit_browser_tool_events(previous_item, state))

        # Handle tool completion
        if (
            self.tool_server is not None
            and previous_item.recipient is not None
            and state.current_item_id is not None
            and state.sent_output_item_added
        ):
            recipient = previous_item.recipient
            # Handle MCP prefix tool completion first
            if recipient.startswith("mcp."):
                events.extend(
                    self._emit_mcp_prefix_completion_events(previous_item, state)
                )
            else:
                # Handle other MCP tool and code interpreter completion
                is_mcp_tool = self._is_mcp_tool_by_namespace(
                    recipient
                ) and state.current_item_id.startswith("mcp_")
                if is_mcp_tool:
                    events.extend(
                        self._emit_mcp_tool_completion_events(previous_item, state)
                    )
                else:
                    events.extend(
                        self._emit_code_interpreter_completion_events(
                            previous_item, state
                        )
                    )

        return events

    def _emit_function_call_delta_events(
        self,
        ctx: StreamingHarmonyContext,
        state: HarmonyStreamingState,
    ) -> list[StreamingResponsesResponse]:
        """Emit events for developer function calls on commentary channel."""
        if not (
            ctx.parser.current_channel == "commentary"
            and ctx.parser.current_recipient
            and ctx.parser.current_recipient.startswith("functions.")
        ):
            return []

        events = []
        if state.is_first_function_call_delta is False:
            state.is_first_function_call_delta = True
            fc_name = ctx.parser.current_recipient[len("functions.") :]
            state.current_item_id = f"fc_{random_uuid()}"
            tool_call_item = ResponseFunctionToolCall(
                name=fc_name,
                type="function_call",
                id=state.current_item_id,
                call_id=f"call_{random_uuid()}",
                arguments="",
                status="in_progress",
            )
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=-1,
                    output_index=state.current_output_index,
                    item=tool_call_item,
                )
            )
        # Always emit the delta (including on first call)
        events.append(
            ResponseFunctionCallArgumentsDeltaEvent(
                item_id=state.current_item_id,
                delta=ctx.last_content_delta,
                output_index=state.current_output_index,
                sequence_number=-1,
                type="response.function_call_arguments.delta",
            )
        )
        return events

    async def _process_harmony_streaming_events(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int,
        _increment_sequence_number_and_return: Callable[
            [StreamingResponsesResponse], StreamingResponsesResponse
        ],
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        state = HarmonyStreamingState()

        async for ctx in result_generator:
            assert isinstance(ctx, StreamingHarmonyContext)

            # finish_reason='error' indicates a retryable error
            self._raise_if_error(ctx.finish_reason, request.request_id)

            if ctx.is_expecting_start():
                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    for event in self._emit_previous_item_done_events(
                        previous_item, state
                    ):
                        yield _increment_sequence_number_and_return(event)
                state.reset_for_new_item()

            # Stream the output of a harmony message
            for event in self._emit_content_delta_events(ctx, state):
                yield _increment_sequence_number_and_return(event)

            # Stream tool call outputs
            for event in self._emit_tool_action_events(ctx, state):
                yield _increment_sequence_number_and_return(event)

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _increment_sequence_number_and_return(
            event: StreamingResponsesResponse,
        ) -> StreamingResponsesResponse:
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            return event

        async with AsyncExitStack() as exit_stack:
            if self.use_harmony:
                # TODO: in streaming, we noticed this bug:
                # https://github.com/vllm-project/vllm/issues/25697
                await self._initialize_tool_sessions(request, context, exit_stack)
                processer = self._process_harmony_streaming_events
            else:
                processer = self._process_simple_streaming_events
            # TODO Hanchen make sampling params to include the structural tag

            initial_response = ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="in_progress",
                usage=None,
            ).model_dump()
            yield _increment_sequence_number_and_return(
                ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=-1,
                    response=initial_response,
                )
            )
            yield _increment_sequence_number_and_return(
                ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=-1,
                    response=initial_response,
                )
            )

            try:
                async for event_data in processer(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                    _increment_sequence_number_and_return,
                ):
                    yield event_data
            except GenerationError as e:
                error_json = self._convert_generation_error_to_streaming_response(e)
                yield _increment_sequence_number_and_return(
                    TypeAdapter(StreamingResponsesResponse).validate_json(error_json)
                )
                return

            async def empty_async_generator():
                # A hack to trick Python to think this is a generator but
                # in fact it immediately returns.
                if False:
                    yield

            final_response = await self.responses_full_generator(
                request,
                sampling_params,
                empty_async_generator(),
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time=created_time,
            )
            yield _increment_sequence_number_and_return(
                ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=-1,
                    response=final_response,
                )
            )
