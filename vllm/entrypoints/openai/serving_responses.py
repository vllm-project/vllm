# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack
from copy import copy
from http import HTTPStatus
from typing import Any, Callable, Final, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
from fastapi import Request
from openai import BaseModel

# yapf conflicts with isort for this block
# yapf: disable
from openai.types.responses import (ResponseCreatedEvent,
                                    ResponseFunctionToolCall,
                                    ResponseInProgressEvent,
                                    ResponseOutputItem,
                                    ResponseOutputItemDoneEvent,
                                    ResponseOutputMessage, ResponseOutputText,
                                    ResponseReasoningItem,
                                    ResponseReasoningTextDeltaEvent,
                                    ResponseReasoningTextDoneEvent)
# yapf: enable
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai_harmony import Message as OpenAIHarmonyMessage

from vllm import envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from vllm.entrypoints.logger import RequestLogger

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              RequestResponseMetadata,
                                              ResponsesRequest,
                                              ResponsesResponse, ResponseUsage)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.tool_server import MCPToolServer, ToolServer
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

# Note: Using local type definitions instead of openai.types.responses
# since those may not be available in all openai package versions


class OpenAIServingResponses(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        tool_server: Optional[ToolServer] = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        use_harmony: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_log_outputs = enable_log_outputs

        self.reasoning_parser: Optional[Callable[[AnyTokenizer], ReasoningParser]] = (
            None
        )
        if reasoning_parser:
            try:
                self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
                assert self.reasoning_parser is not None
            except Exception as e:
                raise TypeError(f"{reasoning_parser=} has not been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default chat sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )

        # If False (default), the "store" option is (silently) ignored and the
        # response is not stored. If True, the response is stored in memory.
        # NOTE(woosuk): This may not be intuitive for users, as the default
        # behavior in OpenAI's Responses API is to store the response, but
        # vLLM's default behavior is not.
        self.enable_store = False
        if self.enable_store:
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled. This may "
                "cause a memory leak since we never remove responses from "
                "the store."
            )

        self.use_harmony =use_harmony
        print(f"JAY DEBUG: self.use_harmony: {use_harmony}")
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

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                '"auto" tool choice has been enabled please note that while'
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored."
            )

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove responses from the store.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove messages from the store.
        self.msg_store: dict[str, list[ChatCompletionMessageParam]] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

        self.tool_server = tool_server

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.store and not self.enable_store:
            if request.background:
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message=(
                        "This vLLM engine does not support `store=True` and "
                        "therefore does not support the background mode. To "
                        "enable these features, set the environment variable "
                        "`VLLM_ENABLE_RESPONSES_API_STORE=1` when launching "
                        "the vLLM server."
                    ),
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            # Disable the store option.
            request.store = False

        # Handle the previous response ID.
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            lora_request = self._maybe_get_adapters(request)
            model_name = self._get_model_name(request.model, lora_request)
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                messages, request_prompts, engine_prompts = await self._make_request(
                    request, prev_response, tokenizer
                )

        except (
            ValueError,
            TypeError,
            RuntimeError,
            jinja2.TemplateError,
            NotImplementedError,
        ) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ["web_search_preview", "code_interpreter"]
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and "
                "streaming mode"
            )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[ConversationContext, None]] = []

        builtin_tool_list: list[str] = []
        if self.use_harmony and self.tool_server is not None:
            if self.tool_server.has_tool("browser"):
                builtin_tool_list.append("browser")
            if self.tool_server.has_tool("python"):
                builtin_tool_list.append("python")

        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    # TODO: initialize tool sessions lazily when the session
                    # is actually used.
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.new_session(tool_name)
                        )
                        for tool_name in builtin_tool_list
                    }
                    tool_sessions = {}
                    for tool_name in builtin_tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(builtin_tool_list) == 0
                    tool_sessions = {}

                for i, engine_prompt in enumerate(engine_prompts):
                    default_max_tokens = self.max_model_len - len(
                        engine_prompt["prompt_token_ids"]
                    )
                    sampling_params = request.to_sampling_params(
                        default_max_tokens, self.default_sampling_params
                    )

                    trace_headers = (
                        None
                        if raw_request is None
                        else await self._get_trace_headers(raw_request.headers)
                    )

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    generator = self._generate_with_builtin_tools(
                        request_id=request.request_id,
                        request_prompt=request_prompts[i],
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        context=context,
                        lora_request=lora_request,
                        priority=request.priority,
                        trace_headers=trace_headers,
                    )
                    generators.append(generator)
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

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
            except Exception as e:
                return self.create_error_response(str(e))
        return self.create_error_response("Should not reach here")

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        """Create request using Harmony format."""
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in " "response API with Harmony"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = {"prompt_token_ids": prompt_token_ids}
        return messages, [prompt_token_ids], [engine_prompt]

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[RequestOutput],
        context: ConversationContext,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        """Generate a complete response."""

        final_res: Optional[RequestOutput] = None
        async for res in result_generator:
            final_res = res
            context.append_output(res)

        if final_res is None:
            return self.create_error_response("No response generated")

        # Get the final output
        final_output = final_res.outputs[0]

        # Create response output items
        if self.use_harmony and isinstance(context, HarmonyContext):
            output_items = self._make_response_output_items_with_harmony(context)
        else:
            output_items = self._make_response_output_items(
                request, final_output, tokenizer
            )

        # Calculate usage
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = len(final_output.token_ids)
        total_tokens = prompt_tokens + completion_tokens

        usage = ResponseUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

        response = ResponsesResponse(
            id=request_metadata.request_id,
            model=model_name,
            output=output_items,
            usage=usage,
            created=created_time or int(time.time()),
        )

        return response

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[RequestOutput],
        context: ConversationContext,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""

        # For now, collect all results and return as single response
        # TODO: Implement true streaming
        final_res: Optional[RequestOutput] = None
        async for res in result_generator:
            final_res = res
            context.append_output(res)

        if final_res is None:
            yield f"data: {self.create_error_response('No response generated').model_dump_json()}\n\n"
            return

        # Get the final output
        final_output = final_res.outputs[0]

        # Create response output items
        if self.use_harmony and isinstance(context, HarmonyContext):
            output_items = self._make_response_output_items_with_harmony(context)
        else:
            output_items = self._make_response_output_items(
                request, final_output, tokenizer
            )

        # Calculate usage
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = len(final_output.token_ids)
        total_tokens = prompt_tokens + completion_tokens

        usage = ResponseUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

        response = ResponsesResponse(
            id=request_metadata.request_id,
            model=model_name,
            output=output_items,
            usage=usage,
            created=created_time or int(time.time()),
        )

        yield f"data: {response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: CompletionOutput,
        tokenizer: AnyTokenizer,
    ) -> list[dict[str, Any]]:
        """Create response output items from completion output."""
        # Simple implementation - convert completion text to output items
        output_text = {
            "text": final_output.text,
            "annotations": [],
            "type": "output_text",
            "logprobs": None,
        }

        text_item = {
            "id": f"msg_{random_uuid()}",
            "content": [output_text],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }

        return [text_item]

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ) -> list[dict[str, Any]]:
        """Create response output items using Harmony context."""
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[OpenAIHarmonyMessage]:
        """Construct input messages using Harmony format."""
        messages = []

        # Add system message
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None,
        )
        messages.append(sys_msg)

        # Add developer message
        dev_msg = get_developer_message()
        messages.append(dev_msg)

        # Add input messages from request
        if hasattr(request, "input") and request.input:
            for input_item in request.input:
                msg = parse_response_input(input_item, [])
                messages.append(msg)

        # Add previous response messages if available
        if prev_response and hasattr(prev_response, "output"):
            prev_msgs = []
            for output_item in prev_response.output:
                # Convert output items back to messages
                # This is a simplified conversion
                if hasattr(output_item, "content"):
                    for content in output_item.content:
                        if hasattr(content, "text"):
                            user_msg = get_user_message(content.text)
                            prev_msgs.append(user_msg)

            if prev_msgs:
                # Handle the case where the last message is final
                last_msg = prev_msgs[-1]
                if hasattr(last_msg, "channel") and last_msg.channel == "final":
                    prev_final_msg_idx = -1
                    for i in range(len(prev_msgs) - 2, -1, -1):
                        prev_msg_i = prev_msgs[i]
                        if (
                            hasattr(prev_msg_i, "channel")
                            and prev_msg_i.channel == "final"
                        ):
                            prev_final_msg_idx = i
                            break
                    recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                    del prev_msgs[prev_final_msg_idx + 1 :]
                    for msg in recent_turn_msgs:
                        if hasattr(msg, "channel") and msg.channel != "analysis":
                            prev_msgs.append(msg)
            messages.extend(prev_msgs)

        return messages
