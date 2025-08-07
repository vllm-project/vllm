# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus
from typing import Callable, Final, Optional, Union

import jinja2
from fastapi import Request
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from vllm import envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption)
from vllm.entrypoints.context import ConversationContext, SimpleContext
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              ResponseReasoningItem,
                                              ResponsesRequest,
                                              ResponsesResponse, UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.tool_server import ToolServer
from vllm.logger import init_logger
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
        tool_server: Optional[ToolServer] = None,
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
        # Construct the input messages.
        messages = self._construct_input_messages(request, prev_response)

        try:
            lora_request = self._maybe_get_adapters(request)
            model_name = self._get_model_name(request.model, lora_request)
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            _, request_prompts, engine_prompts = await self._preprocess_chat(
                request,
                tokenizer,
                messages,
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
            )
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(
            request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[ConversationContext, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"])
                sampling_params = request.to_sampling_params(
                    default_max_tokens, self.default_sampling_params)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

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
            raise NotImplementedError("Streaming responses are not supported")

        try:
            return await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                model_name,
                tokenizer,
                request_metadata,
            )
        except Exception as e:
            return self.create_error_response(str(e))

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext],
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        if created_time is None:
            created_time = int(time.time())

        context: Optional[ConversationContext] = None
        try:
            async for context in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert context is not None
        assert isinstance(context, SimpleContext)
        final_res = context.last_output
        assert final_res is not None
        assert len(final_res.outputs) == 1
        final_output = final_res.outputs[0]

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

        output = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                text=reasoning_content,
                status=None,  # NOTE: Only the last output item has status.
            )
            output.append(reasoning_item)
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
            output.append(message)

        # Calculate usage.
        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = len(final_output.token_ids)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)
        request_metadata.final_usage_info = usage

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
                # NOTE: We skip the reasoning output.
                if isinstance(output_item, ResponseOutputMessage):
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
