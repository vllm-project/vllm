# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from copy import copy
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus
from typing import Callable, Final, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
from fastapi import Request
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
import openai.types.responses as openai_responses_tyes
from openai_harmony import Message as OpenAIMessage
from pydantic import BaseModel

from vllm import envs
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall)
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption)
from vllm.entrypoints.context import (ConversationContext, HarmonyContext,
                                      SimpleContext, StreamingHarmonyContext)
from vllm.entrypoints.harmony_utils import (
    get_developer_message, get_stop_tokens_for_assistant_actions,
    get_system_message, get_user_message, parse_output_message,
    parse_response_input, parse_response_output, render_for_completion)
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              InputTokensDetails,
                                              OutputTokensDetails,
                                              RequestResponseMetadata,
                                              ResponseContentPartDoneEvent,
                                              ResponseOutputItemDoneEvent,
                                              ResponseReasoningItem,
                                              ResponseReasoningTextContent,
                                              ResponseReasoningTextDeltaEvent,
                                              ResponseReasoningTextDoneEvent,
                                              ResponsesRequest,
                                              ResponsesResponse, ResponseUsage)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.tool import HarmonyBrowserTool, HarmonyPythonTool
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


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
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            enable_force_include_usage=enable_force_include_usage,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if reasoning_parser:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
                assert self.reasoning_parser is not None
            except Exception as e:
                raise TypeError(
                    f"{reasoning_parser=} has not been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

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
                "the store.")
        self.use_harmony = True
        self.supports_browsing = False
        self.supports_code_interpreter = False
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            logger.warning("For gpt-oss, we ignore --enable-auto-tool-choice "
                           "and always enable tool use.")
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions())

            self.browser_tool = HarmonyBrowserTool()
            self.supports_browsing = self.browser_tool.enabled
            self.python_tool = HarmonyPythonTool()
            self.supports_code_interpreter = self.python_tool.enabled

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")
            if not self.use_harmony:
                raise NotImplementedError("Auto tool choice is not supported "
                                          "yet unless using Harmony")

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove responses from the store.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove messages from the store.
        self.msg_store: dict[str, list[ChatCompletionMessageParam]] = {}
        # FIXME: This causes a memory leak since we never remove messages
        # from the store.
        self.msg_store: dict[str, Union[list[ChatCompletionMessageParam],
                                        list[OpenAIMessage]]] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

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
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
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
                        "the vLLM server."),
                    status_code=HTTPStatus.BAD_REQUEST,
                )
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
                    self._make_request_with_harmony(request, prev_response))
            else:
                messages, request_prompts, engine_prompts = (
                    await self._make_request(request, prev_response,
                                             tokenizer))

        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(
            request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"])
                sampling_params = request.to_sampling_params(
                    default_max_tokens, self.default_sampling_params)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                context: ConversationContext
                if self.use_harmony:
                    if request.stream:
                        context = StreamingHarmonyContext(
                            messages, self.browser_tool, self.python_tool)
                    else:
                        context = HarmonyContext(messages, self.browser_tool,
                                                 self.python_tool)
                else:
                    context = SimpleContext()
                generator = self._generate_with_builtin_tools(
                    request.request_id,
                    request_prompts[i],
                    engine_prompt,
                    sampling_params,
                    context,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

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
                lambda _: self.background_tasks.pop(response_id, None))
            return response

        if request.stream:
            return self.responses_stream_generator(
                request,
                sampling_params,
                result_generator,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                model_name,
                tokenizer,
                request_metadata,
            )

        try:
            result: Union[
                ErrorResponse,
                ResponsesResponse] = await self.responses_full_generator(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                )
            return result
        except Exception as e:
            return self.create_error_response(str(e))

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: AnyTokenizer,
    ):
        # Construct the input messages.
        messages = self._construct_input_messages(request, prev_response)
        _, request_prompts, engine_prompts = await self._preprocess_chat(
            request,
            tokenizer,
            messages,
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
        )
        return messages, request_prompts, engine_prompts

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in "
                "response API")
        messages = self._construct_input_messages_with_harmony(
            request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)
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
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # TODO: these are all 0 for now!
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None
            assert len(final_res.outputs) == 1
            final_output = final_res.outputs[0]

            output = self._make_response_output_items(request, final_output,
                                                      tokenizer)

            # Calculate usage.
            assert final_res.prompt_token_ids is not None
            num_prompt_tokens = len(final_res.prompt_token_ids)
            num_generated_tokens = len(final_output.token_ids)
            num_cached_tokens = final_res.num_cached_tokens
            num_reasoning_tokens = 0

        usage = ResponseUsage(
            input_tokens=num_prompt_tokens,
            output_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=num_cached_tokens),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=num_reasoning_tokens),
        )

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it.
                if (stored_response is None
                        or stored_response.status != "cancelled"):
                    self.response_store[response.id] = response
        return response

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: CompletionOutput,
        tokenizer: AnyTokenizer,
    ):
        if self.reasoning_parser:
            try:
                reasoning_parser = self.reasoning_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                return self.create_error_response(str(e))

            reasoning_content, content = (
                reasoning_parser.extract_reasoning_content(final_output.text,
                                                           request=request))
        else:
            reasoning_content = None
            content = final_output.text

        output_items = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                content=[ResponseReasoningTextContent(text=reasoning_content)],
                status=None,  # NOTE: Only the last output item has status.
            )
            output_items.append(reasoning_item)
        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        return output_items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append({
                "role": "system",
                "content": request.instructions,
            })

        # Prepend the conversation history.
        if prev_response is not None:
            # Add the previous messages.
            prev_msg = self.msg_store[prev_response.id]
            messages.extend(prev_msg)

            # Add the previous output.
            for output_item in prev_response.output:
                # NOTE: We skip the reasoning output of the previous response.
                if isinstance(output_item, ResponseReasoningItem):
                    continue
                for content in output_item.content:
                    messages.append({
                        "role": "assistant",
                        "content": content.text,
                    })

        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            messages.extend(request.input)  # type: ignore
        return messages

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[OpenAIMessage]:
        messages: list[OpenAIMessage] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = (request.reasoning.effort
                                if request.reasoning else None)
            tool_types = [tool.type for tool in request.tools]
            enable_browsing = ("web_search_preview" in tool_types
                               and self.supports_browsing)
            enable_code_interpreter = ("code_interpreter" in tool_types
                                       and self.supports_code_interpreter)
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                enable_browsing=enable_browsing,
                browser_tool=self.browser_tool,
                enable_python=enable_code_interpreter,
                python_tool=self.python_tool,
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions,
                                            request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME(woosuk): Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if len(prev_msgs) > 0 and hasattr(
                    prev_msgs[-1], 'channel'
            ) and prev_msgs[-1].channel == "final":  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if hasattr(prev_msgs[i], 'channel') and prev_msgs[
                            i].channel == "final":  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1:]
                del prev_msgs[prev_final_msg_idx + 1:]
                for msg in recent_turn_msgs:
                    if hasattr(
                            msg, 'channel'
                    ) and msg.channel != "analysis":  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Reponses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(
                    parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        *args,
        **kwargs,
    ):
        try:
            response = await self.responses_full_generator(
                request, *args, **kwargs)
        except Exception as e:
            logger.exception("Background request failed for %s",
                             request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ErrorResponse):
            # If the request has failed, update the status to "failed".
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a synchronous response.",
                )

            # Update the status to "cancelled".
            response.status = "cancelled"

        # Abort the request.
        if (task := self.background_tasks.get(response_id)):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled",
                                 response_id)
        return response

    def _make_invalid_id_error(self, response_id: str) -> ErrorResponse:
        return self.create_error_response(
            err_type="invalid_request_error",
            message=(f"Invalid 'response_id': '{response_id}'. "
                     "Expected an ID that begins with 'resp'."),
        )

    def _make_not_found_error(self, response_id: str) -> ErrorResponse:
        return self.create_error_response(
            err_type="invalid_request_error",
            message=f"Response with id '{response_id}' not found.",
            status_code=HTTPStatus.NOT_FOUND,
        )

    def _make_store_not_supported_error(self) -> ErrorResponse:
        return self.create_error_response(
            err_type="invalid_request_error",
            message=("`store=True` (default) is not supported. Please set "
                     "`store=False` in Responses API or set "
                     "`VLLM_ENABLE_RESPONSES_API_STORE=1` in the env var when "
                     "starting the vLLM server."),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event: BaseModel):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, 'sequence_number'):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, 'type', 'unknown')
            return (f"event: {event_type}\n"
                    f"data: {event.model_dump_json(indent=None)}\n\n")

        current_content_index = 0
        current_output_index = 0
        current_item_id = ""
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            ))
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            ))

        async for ctx in result_generator:

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            type="reasoning",
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text),
                            ],
                            status="completed",
                        )
                        yield _send_event(
                            ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            ))
                        yield _send_event(
                            ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=reasoning_item,
                            ))
                        yield _send_event(
                            ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            ))
                    elif previous_item.channel == "final":
                        text_content = ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            ))
                        yield _send_event(
                            openai_responses_types.
                            ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            ))
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            ))

            if ctx.parser.last_content_delta:
                if (ctx.parser.current_channel == "final"
                        and ctx.parser.current_recipient is None):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.
                                ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            ))
                        yield _send_event(
                            openai_responses_types.
                            ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=[],
                                ),
                            ))
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        ))
                elif (ctx.parser.current_channel == "analysis"
                      and ctx.parser.current_recipient is None):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.
                                ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            ))
                        yield _send_event(
                            openai_responses_types.
                            ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=[],
                                ),
                            ))
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        ))

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (self.browser_tool.enabled
                        and previous_item.recipient is not None
                        and previous_item.recipient.startswith("browser.")):
                    function_name = previous_item.recipient[len("browser."):]
                    action = None
                    parsed_args = json.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = (openai_responses_types.
                                  response_function_web_search.ActionSearch(
                                      type="search",
                                      query=parsed_args["query"],
                                  ))
                    elif function_name == "open":
                        action = (
                            openai_responses_types.
                            response_function_web_search.ActionOpenPage(
                                type="open_page",
                                # TODO: translate to url
                                url=f"cursor:{parsed_args.get('cursor', '')}",
                            ))
                    elif function_name == "find":
                        action = (
                            openai_responses_types.
                            response_function_web_search.ActionFind(
                                type="find",
                                pattern=parsed_args["pattern"],
                                # TODO: translate to url
                                url=f"cursor:{parsed_args.get('cursor', '')}",
                            ))
                    else:
                        raise ValueError(
                            f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.
                            response_function_web_search.
                            ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        ))
                    yield _send_event(
                        openai_responses_types.
                        ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))
                    yield _send_event(
                        openai_responses_types.
                        ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))

                    # enqueue
                    yield _send_event(
                        openai_responses_types.
                        ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.
                            ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        ))

                if (self.python_tool.enabled
                        and previous_item.recipient is not None
                        and previous_item.recipient.startswith("python")):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.
                            ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        ))
                    yield _send_event(
                        openai_responses_types.
                        ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.
                        ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text))
                    yield _send_event(
                        openai_responses_types.
                        ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))
                    yield _send_event(
                        openai_responses_types.
                        ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        ))
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.
                            ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        ))

        async def empty_async_generator():
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
        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=final_response.model_dump(),
            ))
